//! The scene, in two halves. [`description`] is the typed, named object
//! schema and [`changeset`] its one edit path — what scene files, the pbrt
//! importer, and lookdev edits all speak. The private `prep` module joins
//! that model to the GPU residency below — meshes built into acceleration
//! structures, per-instance materials, emissive triangles and delta
//! lights with their sampling table, a thin-lens camera, and an equirect
//! environment. [`Scene::prep`]
//! builds a description fresh; `Scene::update` follows its accumulated
//! dirty state and rebuilds only what an edit touched. [`Scene::new`]
//! remains as the procedural build the furnace and estimator tests speak
//! (they need materials and environments no scene file can express).
//!
//! [`Scene::demo`] is the standing test subject — a grid of smooth-shaded
//! spheres sweeping roughness × metalness across a glossy floor, where
//! winding, handedness, shading-normal, or energy mistakes are instantly
//! visible, under a warm quad light and the bundled Kloofendal sky. It is
//! [`changeset::ChangeSet::demo`] prepped: the demo scene is data first.

pub mod changeset;
mod demo;
pub mod description;
mod prep;

use std::collections::{BTreeMap, HashMap};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::environment::Environment;
use crate::error::Result;
use crate::gpu::{AccelerationStructure, Buffer, Context, SampledImage, TlasInstance};
use crate::lights::{DeltaLight, LIGHT_NONE, TriangleLight};
use crate::material::Material;

/// Ray-visibility mask bits, matched by the mask each TLAS instance
/// carries. Camera rays trace with [`ray_mask::CAMERA`]; every other ray
/// (bounce, shadow) traces with [`ray_mask::ALL`], so a camera-invisible
/// instance still bounces light, casts shadows, and — when emissive —
/// illuminates. The full per-ray-type set (diffuse/glossy/shadow) is a
/// ledger deferral; today only the camera bit is real.
pub(crate) mod ray_mask {
    pub const CAMERA: u32 = 0x01;
    pub const ALL: u32 = 0xFF;
}

/// A triangle mesh on the host: tightly packed positions, matching shading
/// normals, plus index triples.
pub struct Mesh {
    /// Vertex positions, in meters, in object space.
    pub positions: Vec<Vec3>,
    /// Unit shading normals, one per position, in object space. Shading
    /// interpolates these across each triangle, which is what makes a
    /// coarse sphere render smooth; geometry that *should* look flat
    /// (planes, quads) carries its face normal at every vertex.
    pub normals: Vec<Vec3>,
    /// Counter-clockwise-outward index triples into `positions`.
    pub triangles: Vec<[u32; 3]>,
}

/// One thing in a scene: a mesh, where it stands, and what its surface is.
pub struct Object {
    /// The geometry, in object space.
    pub mesh: Mesh,
    /// Object-to-world placement. Must be invertible — normals and ray
    /// offsets transform through the inverse.
    pub transform: Mat4,
    /// The surface, constant across the mesh (per-face materials are M2+).
    pub material: Material,
}

/// One mesh resident on the GPU. The vertex, normal, and index buffers stay
/// alive past the BLAS build: the surface-shading kernel fetches triangle
/// corners from them to compute geometric normals and interpolate shading
/// normals.
struct GpuMesh {
    blas: AccelerationStructure,
    vertices: Buffer,
    normals: Buffer,
    indices: Buffer,
}

/// One entry of the geometry lookup table, indexed by instance custom index:
/// where the instance's triangles live plus its transforms — everything a
/// kernel needs to re-evaluate shading at a hit. Mirrors
/// `struct GeometryRecord` in `shaders/scene.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeometryRecord {
    positions: vk::DeviceAddress,
    normals: vk::DeviceAddress,
    indices: vk::DeviceAddress,
    /// Rows of the instance's 3×4 object-to-world transform — the same
    /// shape the TLAS instance itself carries.
    object_to_world: [[f32; 4]; 3],
    /// Rows of the inverse: normals transform through it, and the
    /// spawn-point error bounds need both directions.
    world_to_object: [[f32; 4]; 3],
    /// Index of the instance's *first* light record, or [`LIGHT_NONE`] —
    /// an emissive instance has one record per triangle, in primitive
    /// order, so a BSDF-sampled hit finds the pdf its MIS weight competes
    /// against at `light + primitive`.
    light: u32,
    _pad0: [u32; 3],
}

/// Every buffer the scene shares with the kernels, one address each, plus
/// the embedded environment tables — kernels carry a single pointer to
/// this table in their push constants. Mirrors `struct SceneTable` (with
/// its nested `struct Environment`) in `shaders/scene.slang` and
/// `shaders/environment.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneTable {
    geometry: vk::DeviceAddress,
    materials: vk::DeviceAddress,
    lights: vk::DeviceAddress,
    /// The closure's baked lookup tables ([`crate::tables`]) — static
    /// data, but reached through the scene table like everything else the
    /// kernels share, which keeps their push constants inside Vulkan's
    /// guaranteed 128 bytes.
    bsdf_tables: vk::DeviceAddress,
    env_marginal: vk::DeviceAddress,
    env_conditional: vk::DeviceAddress,
    env_pdfs: vk::DeviceAddress,
    env_width: u32,
    env_height: u32,
    /// p(next-event estimation samples the environment rather than the
    /// light list) — `selectProb` on the Slang side.
    env_select_prob: f32,
    _pad0: u32,
    light_count: u32,
    _pad1: u32,
}

/// The scene, resident on the GPU and ready to trace against.
pub struct Scene {
    // Declared before `meshes`: the TLAS dies before the BLASes its
    // instances reference.
    tlas: AccelerationStructure,
    /// The environment radiance image — the binding model's one texture,
    /// bound next to the TLAS at every scene-resource dispatch.
    environment: SampledImage,
    /// The one [`SceneTable`] every kernel reaches scene data through.
    table: Buffer,
    /// The buffers `table` points into, replaced piecewise as edits dirty
    /// them.
    resident: ResidentBuffers,
    /// Mesh residency by name — prep rebuilds only the names an edit
    /// dirtied. The procedural [`Scene::new`] path keys them by object
    /// index and never updates.
    meshes: BTreeMap<String, GpuMesh>,
    camera: Camera,
    /// The environment's dimensions and emitted power, retained so a
    /// light edit can rebuild the scene table (its selection probability
    /// weighs the light list against the environment) without reloading
    /// the image.
    env_size: (u32, u32),
    env_power: f64,
}

/// Every buffer the [`SceneTable`] points into: geometry records,
/// materials, light records, and the environment's three sampling tables.
/// Held to keep the residency alive; replaced piecewise by prep as edits
/// dirty them.
struct ResidentBuffers {
    geometry: Buffer,
    materials: Buffer,
    lights: Buffer,
    /// The closure's lookup tables — uploaded once at build and never
    /// dirtied (the data is embedded in the binary).
    bsdf_tables: Buffer,
    env_marginal: Buffer,
    env_conditional: Buffer,
    env_pdfs: Buffer,
}

impl Scene {
    /// Upload `objects` and build them into a traceable scene, lit by its
    /// emissive objects and `environment` — the procedural build the
    /// estimator tests speak. Production scenes go through
    /// [`Scene::prep`], which builds the same residency from a
    /// [`description::SceneDescription`].
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from upload or acceleration-structure builds.
    ///
    /// # Panics
    ///
    /// On an empty scene or a non-invertible object transform —
    /// programmer bugs.
    pub fn new(
        gpu: &Context,
        objects: &[Object],
        camera: Camera,
        environment: &Environment,
    ) -> Result<Self> {
        assert!(!objects.is_empty(), "a scene needs at least one object");
        let meshes = objects
            .iter()
            .enumerate()
            .map(|(index, object)| upload_mesh(gpu, &format!("scene.object{index}"), &object.mesh))
            .collect::<Result<Vec<GpuMesh>>>()?;
        // The light list: every triangle of every emissive object, in
        // world space. The procedural path has no delta lights — those
        // are description objects, exercised through prep.
        let triangle_lights: Vec<TriangleLight> = objects
            .iter()
            .enumerate()
            .filter(|(_, object)| object.material.emission != Vec3::ZERO)
            .flat_map(|(index, object)| {
                emissive_triangles(
                    &object.mesh.positions,
                    &object.mesh.triangles,
                    object.transform,
                    object.material.emission,
                    index as u32,
                )
            })
            .collect();
        let placements: Vec<Placement> = meshes
            .iter()
            .zip(objects)
            .map(|(mesh, object)| Placement {
                mesh,
                transform: object.transform,
                material: object.material,
                camera_visible: true,
            })
            .collect();
        let tlas = build_scene_tlas(gpu, &placements)?;
        let (geometry, materials, lights) =
            upload_instance_tables(gpu, &placements, &triangle_lights, &[])?;
        let env = upload_environment(gpu, environment)?;
        let resident = ResidentBuffers {
            geometry,
            materials,
            lights,
            bsdf_tables: crate::tables::upload(gpu)?,
            env_marginal: env.marginal,
            env_conditional: env.conditional,
            env_pdfs: env.pdfs,
        };
        let env_size = (environment.width(), environment.height());
        let light_count = triangle_lights.len() as u32;
        let table = upload_scene_table(
            gpu,
            &resident,
            env_size,
            select_probability(env.power, crate::lights::total_power(&triangle_lights, &[])),
            light_count,
        )?;

        Ok(Self {
            tlas,
            environment: env.image,
            table,
            resident,
            meshes: meshes
                .into_iter()
                .enumerate()
                .map(|(index, mesh)| (index.to_string(), mesh))
                .collect(),
            camera,
            env_size,
            env_power: env.power,
        })
    }

    /// The scene's TLAS, ready to bind for ray queries.
    #[must_use]
    pub fn tlas(&self) -> &AccelerationStructure {
        &self.tlas
    }

    /// The scene table: the one buffer of addresses kernels reach all
    /// shared scene data through (geometry records, materials, lights,
    /// environment tables).
    #[must_use]
    pub fn table(&self) -> &Buffer {
        &self.table
    }

    /// The environment radiance image, ready to bind next to the TLAS.
    #[must_use]
    pub fn environment(&self) -> &SampledImage {
        &self.environment
    }

    /// The scene's camera.
    #[must_use]
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Mutable camera access — the viewer's orbit control writes here
    /// between frames.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
}

/// A camera, described by where it sits, what it looks at, and which way
/// is up on screen — a pinhole unless it carries a [`Lens`].
#[derive(Clone, Copy)]
pub struct Camera {
    /// Eye position, meters.
    pub position: Vec3,
    /// The point the view axis passes through.
    pub look_at: Vec3,
    /// Which way is up on screen — the roll control. Usually world up
    /// ([`Vec3::Y`]); need not be perpendicular to the view axis, just
    /// not parallel to it.
    pub up: Vec3,
    /// Vertical field of view, degrees.
    pub vfov_degrees: f32,
    /// The thin lens, when depth of field is wanted; `None` is a pinhole
    /// (everything sharp).
    pub lens: Option<Lens>,
}

/// A thin lens: rays leave a disk instead of a point, and only the focal
/// plane images sharply. Raygen consumes it by scaling the [`RayBasis`]
/// to the focal plane — `position + forward + x·right + y·up` is then a
/// pixel's focal *point* — and re-aiming each ray from a sampled point on
/// the lens disk.
#[derive(Clone, Copy)]
pub struct Lens {
    /// Lens radius, meters; larger blurs out-of-focus geometry more.
    /// Zero is exactly a pinhole.
    pub aperture_radius: f32,
    /// Distance from the camera to the focal plane along the view axis,
    /// meters. Must be positive.
    pub focus_distance: f32,
}

/// A camera's ray-generation basis: the kernel builds each pixel's ray as
/// `normalize(forward + ndc.x · right + ndc.y · up)` with NDC in [-1, 1],
/// +y up. `forward` is unit length; `right` and `up` are scaled by the
/// field of view and aspect ratio.
pub struct RayBasis {
    /// Screen-right, scaled by `tan(vfov/2) · aspect`.
    pub right: Vec3,
    /// Screen-up, scaled by `tan(vfov/2)`.
    pub up: Vec3,
    /// Unit view direction.
    pub forward: Vec3,
}

impl Camera {
    /// The ray-generation basis for a target with the given aspect ratio
    /// (width / height).
    ///
    /// # Panics
    ///
    /// On a degenerate camera — `position == look_at`, or `up` parallel
    /// to the view axis. Both are programmer bugs: description-driven
    /// cameras were validated at apply, and the viewer's orbit control
    /// clamps away from the poles.
    #[must_use]
    pub fn basis(&self, aspect: f32) -> RayBasis {
        let forward = (self.look_at - self.position).normalize();
        assert!(forward.is_finite(), "camera position and look_at coincide");
        let right = forward.cross(self.up).normalize();
        assert!(right.is_finite(), "camera up is parallel to the view axis");
        let up = right.cross(forward);
        let half_height = (self.vfov_degrees.to_radians() / 2.0).tan();
        RayBasis {
            right: right * half_height * aspect,
            up: up * half_height,
            forward,
        }
    }
}

/// One instance as the GPU assembly reads it: the resident mesh it
/// places, where it stands, and its finished GPU material — what
/// [`Scene::new`] lowers objects into and prep lowers a description into,
/// so both build the same residency through the same helpers.
struct Placement<'a> {
    mesh: &'a GpuMesh,
    transform: Mat4,
    material: Material,
    /// Whether camera rays see it — lowered into the instance's TLAS
    /// visibility mask.
    camera_visible: bool,
}

/// Build the TLAS: one instance per placement, with `custom_index` =
/// position, so a hit leads back to the right geometry record and
/// material. A camera-invisible placement drops [`ray_mask::CAMERA`]
/// from its mask, so camera rays traverse past it while every other ray
/// still sees it.
fn build_scene_tlas(gpu: &Context, placements: &[Placement]) -> Result<AccelerationStructure> {
    let instances: Vec<TlasInstance> = placements
        .iter()
        .enumerate()
        .map(|(index, placement)| TlasInstance {
            blas: &placement.mesh.blas,
            transform: placement.transform,
            custom_index: index as u32,
            mask: if placement.camera_visible {
                ray_mask::ALL
            } else {
                ray_mask::ALL & !ray_mask::CAMERA
            } as u8,
            opaque: placement.material.opacity >= 1.0,
        })
        .collect();
    gpu.build_tlas("scene.tlas", &instances)
}

/// Upload the per-instance tables: geometry records (each carrying the
/// index of its instance's *first* light record, or [`LIGHT_NONE`]), the
/// material array, and the light records. An emissive instance's records
/// sit contiguously in primitive order, so a hit's own record is
/// `light + primitive`.
fn upload_instance_tables(
    gpu: &Context,
    placements: &[Placement],
    triangle_lights: &[TriangleLight],
    delta_lights: &[DeltaLight],
) -> Result<(Buffer, Buffer, Buffer)> {
    let light_records = crate::lights::build(triangle_lights, delta_lights);
    let light_index = |instance: u32| {
        triangle_lights
            .iter()
            .position(|light| light.instance == instance)
            .map_or(LIGHT_NONE, |index| index as u32)
    };
    let records: Vec<GeometryRecord> = placements
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            let inverse = placement.transform.inverse();
            assert!(
                inverse.is_finite(),
                "instance transform must be invertible, got {:?}",
                placement.transform
            );
            GeometryRecord {
                positions: placement.mesh.vertices.device_address(),
                normals: placement.mesh.normals.device_address(),
                indices: placement.mesh.indices.device_address(),
                object_to_world: transform_rows(placement.transform),
                world_to_object: transform_rows(inverse),
                light: light_index(index as u32),
                _pad0: [0; 3],
            }
        })
        .collect();
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let geometry = gpu.upload_buffer("scene.geometry", bytemuck::cast_slice(&records), usage)?;
    let materials: Vec<Material> = placements
        .iter()
        .map(|placement| placement.material)
        .collect();
    let materials =
        gpu.upload_buffer("scene.materials", bytemuck::cast_slice(&materials), usage)?;
    // Vulkan forbids empty buffers, so a lightless scene uploads one
    // zeroed record the kernels never read (the table says count 0).
    let padded = [Zeroable::zeroed()];
    let lights = gpu.upload_buffer(
        "scene.lights",
        bytemuck::cast_slice(if light_records.is_empty() {
            &padded
        } else {
            &light_records
        }),
        usage,
    )?;
    Ok((geometry, materials, lights))
}

/// Upload the [`SceneTable`] — the one buffer of addresses every kernel
/// reaches scene data through, rebuilt whenever anything it points at
/// moved.
fn upload_scene_table(
    gpu: &Context,
    resident: &ResidentBuffers,
    env_size: (u32, u32),
    env_select_prob: f32,
    light_count: u32,
) -> Result<Buffer> {
    let table = SceneTable {
        geometry: resident.geometry.device_address(),
        materials: resident.materials.device_address(),
        lights: resident.lights.device_address(),
        bsdf_tables: resident.bsdf_tables.device_address(),
        env_marginal: resident.env_marginal.device_address(),
        env_conditional: resident.env_conditional.device_address(),
        env_pdfs: resident.env_pdfs.device_address(),
        env_width: env_size.0,
        env_height: env_size.1,
        env_select_prob,
        _pad0: 0,
        light_count,
        _pad1: 0,
    };
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    gpu.upload_buffer("scene.table", bytemuck::bytes_of(&table), usage)
}

/// Unpack an emissive instance into per-triangle lights: every triangle
/// of the mesh, in primitive order, transformed to world space — the
/// order (and the one-record-per-triangle rule, degenerate triangles
/// included) is what lets a BSDF-sampled hit on the light find its own
/// record at `GeometryRecord.light + primitive`.
fn emissive_triangles(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    transform: Mat4,
    emission: Vec3,
    instance: u32,
) -> Vec<TriangleLight> {
    triangles
        .iter()
        .enumerate()
        .map(|(primitive, corners)| TriangleLight {
            corners: corners.map(|vertex| transform.transform_point3(positions[vertex as usize])),
            emission,
            instance,
            primitive: primitive as u32,
        })
        .collect()
}

/// The environment's GPU half: the radiance image, the three sampling
/// tables, and the emitted power the selection probability weighs.
struct GpuEnvironment {
    image: SampledImage,
    marginal: Buffer,
    conditional: Buffer,
    pdfs: Buffer,
    power: f64,
}

/// Upload the environment's image and sampling tables.
fn upload_environment(gpu: &Context, environment: &Environment) -> Result<GpuEnvironment> {
    let tables = environment.tables();
    let image = gpu.upload_sampled_image(
        "scene.environment",
        environment.width(),
        environment.height(),
        environment.texels(),
    )?;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    Ok(GpuEnvironment {
        image,
        marginal: gpu.upload_buffer(
            "scene.env.marginal",
            bytemuck::cast_slice(&tables.marginal),
            usage,
        )?,
        conditional: gpu.upload_buffer(
            "scene.env.conditional",
            bytemuck::cast_slice(&tables.conditional),
            usage,
        )?,
        pdfs: gpu.upload_buffer("scene.env.pdfs", bytemuck::cast_slice(&tables.pdfs), usage)?,
        power: tables.power,
    })
}

/// Weigh the environment against the light list: the power-proportional
/// probability that next-event estimation samples the environment rather
/// than the list. The environment weighs its luminance integral over the
/// sphere — a flux per unit receiver area, so the comparison implicitly
/// stands in a ~1 m² receiver — against [`crate::lights::total_power`]'s
/// per-kind flux measures. The approximations only steer noise: the MIS
/// weights stay exact whatever this probability is. The exact-0/exact-1
/// endpoints *are* load-bearing: the shader walks the light list
/// whenever its draw lands above `select_prob`, so a scene whose list is
/// powerless must pin it to 1, and a black environment (with no other
/// lights either) disables next-event estimation entirely.
fn select_probability(env_power: f64, light_power: f64) -> f32 {
    if light_power == 0.0 {
        f32::from(u8::from(env_power > 0.0))
    } else {
        (env_power / (env_power + light_power)) as f32
    }
}

/// The top three rows of an affine transform, in the kernels' `float4[3]`
/// row-major shape (glam matrices are column-major, hence the transpose).
fn transform_rows(transform: Mat4) -> [[f32; 4]; 3] {
    let rows = transform.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}

fn upload_mesh(gpu: &Context, name: &str, mesh: &Mesh) -> Result<GpuMesh> {
    assert_eq!(
        mesh.normals.len(),
        mesh.positions.len(),
        "a mesh needs one shading normal per vertex"
    );
    // BUILD_INPUT for the BLAS build; STORAGE + device address so the
    // shading kernel can fetch triangle corners afterwards.
    let usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER;
    let vertices = gpu.upload_buffer(
        &format!("{name}.vertices"),
        bytemuck::cast_slice(&mesh.positions),
        usage,
    )?;
    let normals = gpu.upload_buffer(
        &format!("{name}.normals"),
        bytemuck::cast_slice(&mesh.normals),
        usage,
    )?;
    let indices = gpu.upload_buffer(
        &format!("{name}.indices"),
        bytemuck::cast_slice(&mesh.triangles),
        usage,
    )?;
    let blas = gpu.build_blas(
        &format!("{name}.blas"),
        &vertices,
        mesh.positions.len() as u32,
        &indices,
        mesh.triangles.len() as u32,
    )?;
    Ok(GpuMesh {
        blas,
        vertices,
        normals,
        indices,
    })
}

/// A unit-radius icosphere: `subdivisions` rounds of 4-way face splits of an
/// icosahedron, every vertex re-projected onto the sphere. Yields 20·4ⁿ
/// triangles whose shading normals are the exact sphere normals, so the
/// surface shades smooth at any subdivision — only the silhouette betrays
/// the facets.
#[must_use]
pub fn icosphere(subdivisions: u32) -> Mesh {
    let mut mesh = icosahedron();
    for _ in 0..subdivisions {
        // Midpoints are shared by neighboring faces; the cache keeps the
        // mesh watertight instead of duplicating each edge's midpoint.
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut faces = Vec::with_capacity(mesh.triangles.len() * 4);
        for [a, b, c] in std::mem::take(&mut mesh.triangles) {
            let ab = midpoint(&mut mesh.positions, &mut midpoints, a, b);
            let bc = midpoint(&mut mesh.positions, &mut midpoints, b, c);
            let ca = midpoint(&mut mesh.positions, &mut midpoints, c, a);
            faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        mesh.triangles = faces;
    }
    // Every vertex lies on the unit sphere, where the exact normal at a
    // point is the point itself.
    mesh.normals = mesh.positions.clone();
    mesh
}

fn midpoint(
    positions: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    *cache.entry((a.min(b), a.max(b))).or_insert_with(|| {
        let index = positions.len() as u32;
        let mid = positions[a as usize].midpoint(positions[b as usize]);
        positions.push(mid.normalize());
        index
    })
}

/// The regular icosahedron with unit-radius vertices: three orthogonal
/// golden-ratio rectangles, faces wound counter-clockwise seen from outside.
fn icosahedron() -> Mesh {
    let phi = 1.0_f32.midpoint(5.0_f32.sqrt());
    let positions = [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ]
    .into_iter()
    .map(|p| Vec3::from(p).normalize())
    .collect::<Vec<Vec3>>();
    let normals = positions.clone();
    let triangles = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    Mesh {
        positions,
        normals,
        triangles,
    }
}

/// A square ground plane in the XZ plane at y = 0, spanning ±`half_extent`
/// meters, normal +Y.
#[must_use]
pub fn ground_plane(half_extent: f32) -> Mesh {
    let e = half_extent;
    Mesh {
        positions: vec![
            Vec3::new(-e, 0.0, -e),
            Vec3::new(-e, 0.0, e),
            Vec3::new(e, 0.0, e),
            Vec3::new(e, 0.0, -e),
        ],
        normals: vec![Vec3::Y; 4],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_counts_follow_the_subdivision_formula() {
        for n in 0..3 {
            let mesh = icosphere(n);
            assert_eq!(mesh.triangles.len(), 20 * 4_usize.pow(n));
            assert_eq!(mesh.positions.len(), 10 * 4_usize.pow(n) + 2);
        }
    }

    #[test]
    fn icosphere_vertices_lie_on_the_unit_sphere() {
        let mesh = icosphere(2);
        for position in &mesh.positions {
            assert!((position.length() - 1.0).abs() < 1e-6);
        }
        // …which makes each vertex its own exact shading normal.
        assert_eq!(mesh.normals, mesh.positions);
    }

    #[test]
    fn meshes_carry_one_shading_normal_per_vertex() {
        for mesh in [icosphere(0), icosphere(3), ground_plane(2.0)] {
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            for normal in mesh.normals {
                assert!((normal.length() - 1.0).abs() < 1e-6);
            }
        }
    }

    /// The renderer-breaking bug this scene exists to catch: for a convex
    /// solid centered at the origin, a counter-clockwise-outward face has
    /// its geometric normal pointing away from the origin.
    #[test]
    fn winding_is_counter_clockwise_outward() {
        let mesh = icosphere(1);
        for [a, b, c] in mesh.triangles {
            let (a, b, c) = (
                mesh.positions[a as usize],
                mesh.positions[b as usize],
                mesh.positions[c as usize],
            );
            assert!((b - a).cross(c - a).dot(a + b + c) > 0.0);
        }

        let plane = ground_plane(5.0);
        for [a, b, c] in plane.triangles {
            let (a, b, c) = (
                plane.positions[a as usize],
                plane.positions[b as usize],
                plane.positions[c as usize],
            );
            assert!((b - a).cross(c - a).y > 0.0);
        }
    }

    /// The ray basis must be orthogonal, oriented (up skyward, right = +X
    /// when looking down −Z), and scaled by fov and aspect — the kernel
    /// trusts it blindly.
    #[test]
    fn camera_basis_is_orthogonal_and_fov_scaled() {
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 5.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 90.0,
            lens: None,
        };
        let basis = camera.basis(2.0);
        assert!((basis.forward.length() - 1.0).abs() < 1e-6);
        assert!(basis.forward.dot(basis.right).abs() < 1e-6);
        assert!(basis.forward.dot(basis.up).abs() < 1e-6);
        assert!(basis.right.dot(basis.up).abs() < 1e-6);
        // tan(90° / 2) = 1, so |up| = 1 and |right| = aspect.
        assert!((basis.up.length() - 1.0).abs() < 1e-6);
        assert!((basis.right.length() - 2.0).abs() < 1e-6);
        assert!(basis.up.y > 0.0);
        assert!(basis.right.x > 0.0);
    }

    /// `up` carries roll: flipping it upside down while looking down −Z
    /// must flip the screen — up and right both negate. This is the
    /// orientation the format's camera op commits to (`up` is the roll
    /// control), so the basis has to honor it, not just world +Y.
    #[test]
    fn camera_up_carries_roll() {
        let level = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            vfov_degrees: 60.0,
            lens: None,
        };
        let inverted = Camera {
            up: -Vec3::Y,
            ..level
        };
        let (a, b) = (level.basis(1.0), inverted.basis(1.0));
        assert!((a.up + b.up).length() < 1e-6, "{} vs {}", a.up, b.up);
        assert!((a.right + b.right).length() < 1e-6);
        assert!((a.forward - b.forward).length() < 1e-6);
    }

    #[test]
    fn indices_stay_in_bounds() {
        let mesh = icosphere(2);
        let count = mesh.positions.len() as u32;
        assert!(mesh.triangles.iter().flatten().all(|&i| i < count));
    }

    /// Two BLASes and the TLAS build without errors on real hardware
    /// (validation complaints appear via the debug messenger in the log).
    #[test]
    fn demo_scene_builds() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        Scene::demo(&gpu).expect("demo scene should build");
    }
}
