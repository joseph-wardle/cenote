//! Scenes as the tracer consumes them: meshes built into acceleration
//! structures, per-instance materials, quad lights with their sampling
//! table, a pinhole camera, and an equirect environment. All geometry is
//! procedural and the only file input is the environment EXR (real scene
//! formats are M2's job); [`Scene::demo`] is the standing test subject — a
//! grid of deliberately faceted spheres sweeping roughness × metalness
//! over a glossy floor, where winding, handedness, or energy mistakes are
//! instantly visible, under a warm quad light and the bundled Kloofendal
//! sky.

use std::collections::HashMap;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::color::acescg_from_rec709;
use crate::environment::Environment;
use crate::error::Result;
use crate::gpu::{AccelerationStructure, Buffer, Context, SampledImage, TlasInstance};
use crate::lights::{LIGHT_NONE, QuadLight};
use crate::material::Material;

/// A triangle mesh on the host: tightly packed positions plus index triples.
pub struct Mesh {
    /// Vertex positions, in meters, in object space.
    pub positions: Vec<Vec3>,
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

/// One mesh resident on the GPU. The vertex and index buffers stay alive
/// past the BLAS build: the surface-shading kernel fetches triangle corners
/// from them to compute geometric normals.
struct GpuMesh {
    blas: AccelerationStructure,
    vertices: Buffer,
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
    indices: vk::DeviceAddress,
    /// Rows of the instance's 3×4 object-to-world transform — the same
    /// shape the TLAS instance itself carries.
    object_to_world: [[f32; 4]; 3],
    /// Rows of the inverse: normals transform through it, and the
    /// spawn-point error bounds need both directions.
    world_to_object: [[f32; 4]; 3],
    /// Index into the light list, or [`LIGHT_NONE`] — how a BSDF-sampled
    /// hit on a light finds the pdf its MIS weight competes against.
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
    env_marginal: vk::DeviceAddress,
    env_conditional: vk::DeviceAddress,
    env_pdfs: vk::DeviceAddress,
    env_width: u32,
    env_height: u32,
    /// p(next-event estimation samples the environment rather than a quad).
    env_selection: f32,
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
    /// The GPU material table the scene table points into — updated in
    /// place by [`Scene::set_material`].
    material_buffer: Buffer,
    /// Host copy of the material table, so an edit rewrites one entry.
    materials: Vec<Material>,
    /// The other buffers `table` points into: geometry records, light
    /// records, and the environment's three sampling tables.
    #[expect(dead_code, reason = "GPU residency: the buffers `table` points into")]
    resident: [Buffer; 5],
    #[expect(
        dead_code,
        reason = "GPU residency: the BLASes and the buffers the geometry records point into"
    )]
    meshes: Vec<GpuMesh>,
    camera: Camera,
}

impl Scene {
    /// Upload `objects` and build them into a traceable scene, lit by its
    /// emissive objects and `environment`.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from upload or acceleration-structure builds.
    ///
    /// # Panics
    ///
    /// On an empty scene, a non-invertible object transform, or an
    /// emissive object whose mesh is not a parallelogram quad (all M1
    /// lights are) — programmer bugs.
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
        // One instance per object, with custom_index = position in
        // `objects`, so a hit leads back to the right vertex data and
        // material.
        let instances: Vec<TlasInstance> = meshes
            .iter()
            .zip(objects)
            .enumerate()
            .map(|(index, (mesh, object))| TlasInstance {
                blas: &mesh.blas,
                transform: object.transform,
                custom_index: index as u32,
            })
            .collect();
        let tlas = gpu.build_tlas("scene.tlas", &instances)?;

        // The light list: every emissive object, validated and unpacked
        // into the world-space parallelogram next-event sampling draws
        // points from.
        let quads: Vec<QuadLight> = objects
            .iter()
            .enumerate()
            .filter(|(_, object)| object.material.emission != Vec3::ZERO)
            .map(|(index, object)| light_quad(object, index as u32))
            .collect();
        let light_records = crate::lights::build(&quads);
        let light_index = |instance: u32| {
            quads
                .iter()
                .position(|quad| quad.instance == instance)
                .map_or(LIGHT_NONE, |index| index as u32)
        };

        let records: Vec<GeometryRecord> = meshes
            .iter()
            .zip(objects)
            .enumerate()
            .map(|(index, (mesh, object))| {
                let inverse = object.transform.inverse();
                assert!(
                    inverse.is_finite(),
                    "object transform must be invertible, got {:?}",
                    object.transform
                );
                GeometryRecord {
                    positions: mesh.vertices.device_address(),
                    indices: mesh.indices.device_address(),
                    object_to_world: transform_rows(object.transform),
                    world_to_object: transform_rows(inverse),
                    light: light_index(index as u32),
                    _pad0: [0; 3],
                }
            })
            .collect();
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let geometry =
            gpu.upload_buffer("scene.geometry", bytemuck::cast_slice(&records), usage)?;
        let materials: Vec<Material> = objects.iter().map(|object| object.material).collect();
        let material_buffer =
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

        let env = upload_environment(gpu, environment, &quads)?;
        let table = SceneTable {
            geometry: geometry.device_address(),
            materials: material_buffer.device_address(),
            lights: lights.device_address(),
            env_marginal: env.marginal.device_address(),
            env_conditional: env.conditional.device_address(),
            env_pdfs: env.pdfs.device_address(),
            env_width: environment.width(),
            env_height: environment.height(),
            env_selection: env.selection,
            _pad0: 0,
            light_count: light_records.len() as u32,
            _pad1: 0,
        };
        let table = gpu.upload_buffer("scene.table", bytemuck::bytes_of(&table), usage)?;

        Ok(Self {
            tlas,
            environment: env.image,
            table,
            material_buffer,
            materials,
            resident: [geometry, lights, env.marginal, env.conditional, env.pdfs],
            meshes,
            camera,
        })
    }

    /// Grid columns: `specular_roughness` 0 → 1, left to right.
    const GRID_COLUMNS: usize = 5;
    /// Grid rows: `metalness` 0 → 1, bottom to top.
    const GRID_ROWS: usize = 3;

    /// The floor's instance index in [`Scene::demo`] — the grid's spheres
    /// come first, so this is also their count. The floor is the one
    /// uniform surface in the demo, which makes it the object the viewer's
    /// material sliders edit.
    pub const DEMO_FLOOR: usize = Self::GRID_COLUMNS * Self::GRID_ROWS;

    /// The demo scene: a terracotta material chart — a grid of spheres
    /// sweeping `specular_roughness` 0 → 1 left to right and `metalness`
    /// 0 → 1 bottom to top, the same base color read as a lacquered
    /// plastic's diffuse base in the bottom row and a conductor's F0 in
    /// the top — over a lightly glossy gray floor that mirrors it. A warm
    /// quad light overhead to the left is the key (its soft shadow and
    /// warm cast are what next-event estimation resolves), and the bundled
    /// Kloofendal sky (see `assets/README.md`) fills, backs, and reflects
    /// — its unclipped sun is the importance-sampling stress case the
    /// environment tables exist for.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from upload, decode, or acceleration-structure
    /// builds.
    pub fn demo(gpu: &Context) -> Result<Self> {
        let terracotta = acescg_from_rec709(Vec3::new(0.7, 0.22, 0.08));
        let mut objects: Vec<Object> = (0..Self::DEMO_FLOOR)
            .map(|index| {
                let (row, column) = (index / Self::GRID_COLUMNS, index % Self::GRID_COLUMNS);
                let sweep = |step: usize, steps: usize| step as f32 / (steps - 1) as f32;
                Object {
                    mesh: icosphere(2),
                    // The bottom row rests on the floor; the rest float
                    // above it — the standard material-chart layout.
                    transform: Mat4::from_translation(Vec3::new(
                        1.2 * (column as f32 - 2.0),
                        0.5 + 1.2 * row as f32,
                        0.0,
                    )) * Mat4::from_scale(Vec3::splat(0.5)),
                    material: Material::glossy(terracotta, 0.4, sweep(column, Self::GRID_COLUMNS))
                        .with_metalness(sweep(row, Self::GRID_ROWS)),
                }
            })
            .collect();
        objects.push(Object {
            // Large enough that the frame's bottom edge still lands on it
            // from the pulled-back camera below.
            mesh: ground_plane(12.0),
            transform: Mat4::IDENTITY,
            material: Material::glossy(acescg_from_rec709(Vec3::splat(0.65)), 0.1, 0.15),
        });
        // A 1.5 m × 1.5 m quad, up and off to the *left* — opposite the
        // HDRI's sun (up-right-behind, 48° elevation), so the spheres are
        // cross-lit warm/cool. Placed outside the default framing (above
        // the frame's top edge, ~y 4.8 at the quad's depth), and high
        // enough that the shadow it cuts out of the sunlight lands outside
        // the frame too, instead of reading as a dark artifact.
        objects.push(Object {
            mesh: ground_plane(0.75),
            transform: Mat4::from_translation(Vec3::new(-3.5, 5.4, 1.0)),
            material: Material::emitter(acescg_from_rec709(Vec3::new(1.0, 0.85, 0.6)) * 18.0),
        });
        // A little above the grid's center and pulled back far enough that
        // the whole chart fits a square frame (the goldens'), with the
        // floor and its reflections along the lower edge.
        let camera = Camera {
            position: Vec3::new(0.0, 2.6, 9.5),
            look_at: Vec3::new(0.0, 1.7, 0.0),
            vfov_degrees: 40.0,
        };
        // Loaded from the dev tree rather than embedded: at 4k the asset
        // is 43 MB, which no binary should carry. Real scene I/O (and
        // installable assets) is M2's job; this matches how shader hot
        // reload already finds its sources.
        let load = || -> Result<Environment> {
            let path =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(Self::DEMO_ENVIRONMENT);
            Environment::from_equirect_exr(&std::fs::read(path)?)
        };
        // The lib test suite builds this scene dozens of times per process,
        // and the 4k decode plus table build is seconds of debug-profile
        // CPU each — so tests share one copy. Outside tests a process
        // builds the demo once, and shouldn't pin ~200 MB of host-side
        // copies for its lifetime.
        #[cfg(test)]
        {
            static DEMO_SKY: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
            // Two steps rather than `get_or_init(load)`: the load can fail,
            // and only a loaded environment may be stored.
            let sky = if let Some(sky) = DEMO_SKY.get() {
                sky
            } else {
                let loaded = load()?;
                DEMO_SKY.get_or_init(|| loaded)
            };
            Self::new(gpu, &objects, camera, sky)
        }
        #[cfg(not(test))]
        {
            Self::new(gpu, &objects, camera, &load()?)
        }
    }

    /// The demo environment's path, relative to the crate root — the
    /// bundled Kloofendal sky (`assets/README.md` has provenance and
    /// encoding notes).
    pub const DEMO_ENVIRONMENT: &str = "assets/kloofendal_puresky.exr";

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

    /// The material of instance `index` — read, modify, and hand back to
    /// [`Scene::set_material`].
    ///
    /// # Panics
    ///
    /// If `index` is out of range — instance indices are fixed at build.
    #[must_use]
    pub fn material(&self, index: usize) -> Material {
        self.materials[index]
    }

    /// Replace instance `index`'s material, in place on the GPU — the
    /// viewer's sliders editing a surface between frames. The caller owns
    /// resetting any accumulation that no longer matches. Emission is
    /// *not* live-editable (the light list and its alias table are built
    /// at prep), so the edit must not change it.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from the upload.
    ///
    /// # Panics
    ///
    /// If `index` is out of range, or the edit changes `emission` —
    /// programmer bugs.
    pub fn set_material(&mut self, gpu: &Context, index: usize, material: Material) -> Result<()> {
        assert_eq!(
            self.materials[index].emission, material.emission,
            "emission is baked into the light list at prep"
        );
        self.materials[index] = material;
        gpu.update_buffer(&self.material_buffer, bytemuck::cast_slice(&self.materials))
    }
}

/// A pinhole camera, described by where it sits and what it looks at.
/// World up is +Y (crate convention); the view axis must not be vertical.
pub struct Camera {
    /// Eye position, meters.
    pub position: Vec3,
    /// The point the view axis passes through.
    pub look_at: Vec3,
    /// Vertical field of view, degrees.
    pub vfov_degrees: f32,
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
    /// On a degenerate camera — `position == look_at`, or a view axis
    /// parallel to world up. Both are programmer bugs.
    #[must_use]
    pub fn basis(&self, aspect: f32) -> RayBasis {
        let forward = (self.look_at - self.position).normalize();
        assert!(forward.is_finite(), "camera position and look_at coincide");
        let right = forward.cross(Vec3::Y).normalize();
        assert!(right.is_finite(), "camera view axis is vertical");
        let up = right.cross(forward);
        let half_height = (self.vfov_degrees.to_radians() / 2.0).tan();
        RayBasis {
            right: right * half_height * aspect,
            up: up * half_height,
            forward,
        }
    }
}

/// Unpack an emissive object into the world-space parallelogram all M1
/// lights must be, validating that the mesh really is one: next-event
/// estimation samples points on this analytic quad, so it has to describe
/// exactly the surface shadow rays trace against — a mismatch would let
/// the light-sampling and BSDF-sampling strategies disagree about where
/// the light is.
///
/// # Panics
///
/// If the mesh is not two triangles tiling a parallelogram.
fn light_quad(object: &Object, instance: u32) -> QuadLight {
    let mesh = &object.mesh;
    assert!(
        mesh.positions.len() == 4 && mesh.triangles.len() == 2,
        "M1 lights are parallelogram quads; instance {instance} has {} vertices, {} triangles",
        mesh.positions.len(),
        mesh.triangles.len()
    );
    // Vertex 0 is a corner of any parallelogram. Its opposite vertex is
    // the one that equals the sum of 0's two neighbors minus vertex 0;
    // trying each candidate identifies the layout.
    let p = &mesh.positions;
    let others = |opposite: usize| match opposite {
        1 => [2usize, 3],
        2 => [1, 3],
        _ => [1, 2],
    };
    let opposite = (1usize..4).find(|&candidate| {
        let [b, c] = others(candidate);
        let expected = p[b] + p[c] - p[0];
        let tolerance = 1e-4 * (p[b] - p[0]).length().max((p[c] - p[0]).length());
        (p[candidate] - expected).length() <= tolerance
    });
    let Some(opposite) = opposite else {
        panic!("emissive mesh of instance {instance} is not a parallelogram");
    };
    let [b, c] = others(opposite);

    // The two triangles must tile the quad: all four vertices used, and
    // the shared edge on one of the diagonals (two triangles sharing a
    // *side* would overlap instead of tiling).
    let mut used: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let shared: Vec<u32> = mesh.triangles[0]
        .iter()
        .filter(|vertex| mesh.triangles[1].contains(vertex))
        .copied()
        .collect();
    let on_diagonal =
        shared.len() == 2 && (shared.contains(&0) == shared.contains(&(opposite as u32)));
    assert!(
        used == [0, 1, 2, 3] && on_diagonal,
        "emissive mesh of instance {instance} does not tile its quad"
    );

    let world = |vertex: Vec3| object.transform.transform_point3(vertex);
    let corner = world(p[0]);
    QuadLight {
        corner,
        edge1: world(p[b]) - corner,
        edge2: world(p[c]) - corner,
        emission: object.material.emission,
        instance,
    }
}

/// The environment's GPU half: the radiance image, the three sampling
/// tables, and the next-event selection probability.
struct GpuEnvironment {
    image: SampledImage,
    marginal: Buffer,
    conditional: Buffer,
    pdfs: Buffer,
    selection: f32,
}

/// Upload the environment's image and sampling tables, and weigh it
/// against the quad lights: the power-proportional probability that
/// next-event estimation samples the environment rather than a quad.
/// Quads weigh π × luminance × area (one face's exitance-weighted flux);
/// the environment weighs its luminance integral over the sphere, which
/// is a flux per unit receiver area — the comparison implicitly stands in
/// a ~1 m² receiver, a heuristic that only noise, never correctness,
/// rides on. The exact-0/exact-1 endpoints *are* load-bearing: the shader
/// walks the quad list whenever its draw lands above `selection`, so a
/// lightless scene must pin it to 1, and a black environment (with no
/// quads either) disables next-event estimation entirely.
fn upload_environment(
    gpu: &Context,
    environment: &Environment,
    quads: &[QuadLight],
) -> Result<GpuEnvironment> {
    let tables = environment.tables();
    let quad_power = std::f64::consts::PI * crate::lights::total_power(quads);
    let selection = if quad_power == 0.0 {
        f32::from(u8::from(tables.power > 0.0))
    } else {
        (tables.power / (tables.power + quad_power)) as f32
    };
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
        selection,
    })
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
        indices,
    })
}

/// A unit-radius icosphere: `subdivisions` rounds of 4-way face splits of an
/// icosahedron, every vertex re-projected onto the sphere. Yields 20·4ⁿ
/// faceted triangles.
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
    .collect();
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
        for position in icosphere(2).positions {
            assert!((position.length() - 1.0).abs() < 1e-6);
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
            vfov_degrees: 90.0,
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
