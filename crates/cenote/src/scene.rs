//! Scenes as the tracer consumes them: meshes built into acceleration
//! structures, per-instance materials, quad lights with their sampling
//! table, a pinhole camera, and a constant sky. All geometry is procedural
//! and zero file I/O (real scene formats are M2's job); [`Scene::demo`] is
//! the standing test subject — a row of deliberately faceted spheres
//! sweeping metalness across a glossy floor, where winding, handedness, or
//! energy mistakes are instantly visible, under a warm quad light that
//! gives direct-light sampling something to find.

use std::collections::HashMap;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::color::acescg_from_rec709;
use crate::error::Result;
use crate::gpu::{AccelerationStructure, Buffer, Context, TlasInstance};
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

/// Every buffer the scene shares with the kernels, one address each —
/// kernels carry a single pointer to this table in their push constants.
/// Mirrors `struct SceneTable` in `shaders/scene.slang` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneTable {
    geometry: vk::DeviceAddress,
    materials: vk::DeviceAddress,
    lights: vk::DeviceAddress,
    light_count: u32,
    _pad0: u32,
}

/// The scene, resident on the GPU and ready to trace against.
pub struct Scene {
    // Declared before `meshes`: the TLAS dies before the BLASes its
    // instances reference.
    tlas: AccelerationStructure,
    /// The one [`SceneTable`] every kernel reaches scene data through.
    table: Buffer,
    /// The GPU material table the scene table points into — updated in
    /// place by [`Scene::set_material`].
    material_buffer: Buffer,
    /// Host copy of the material table, so an edit rewrites one entry.
    materials: Vec<Material>,
    /// The other buffers `table` points into: geometry records and light
    /// records.
    #[expect(dead_code, reason = "GPU residency: the buffers `table` points into")]
    resident: [Buffer; 2],
    #[expect(
        dead_code,
        reason = "GPU residency: the BLASes and the buffers the geometry records point into"
    )]
    meshes: Vec<GpuMesh>,
    camera: Camera,
    sky: Vec3,
}

impl Scene {
    /// Upload `objects` and build them into a traceable scene, lit by its
    /// emissive objects and a constant `sky` (radiance in `ACEScg`, every
    /// direction — the environment upgrades to an HDRI in M1 step 10).
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
    pub fn new(gpu: &Context, objects: &[Object], camera: Camera, sky: Vec3) -> Result<Self> {
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

        let table = SceneTable {
            geometry: geometry.device_address(),
            materials: material_buffer.device_address(),
            lights: lights.device_address(),
            light_count: light_records.len() as u32,
            _pad0: 0,
        };
        let table = gpu.upload_buffer("scene.table", bytemuck::bytes_of(&table), usage)?;

        Ok(Self {
            tlas,
            table,
            material_buffer,
            materials,
            resident: [geometry, lights],
            meshes,
            camera,
            sky,
        })
    }

    /// Instances `0..DEMO_SPHERES` of [`Scene::demo`] are its sphere row —
    /// the objects the viewer's material sliders edit.
    pub const DEMO_SPHERES: usize = 5;

    /// The demo scene: a row of [`Scene::DEMO_SPHERES`] terracotta spheres
    /// sweeping metalness 1 → 0 left to right — the same base color read
    /// as a conductor's F0 on the left and a lacquered plastic's diffuse
    /// base on the right — on a lightly glossy gray floor that mirrors
    /// them. A warm quad light overhead to the right is the key (its soft
    /// shadow and warm cast are what next-event estimation resolves) over
    /// a dim gray sky's ambient fill.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from upload or acceleration-structure builds.
    pub fn demo(gpu: &Context) -> Result<Self> {
        let terracotta = acescg_from_rec709(Vec3::new(0.7, 0.22, 0.08));
        let mut objects: Vec<Object> = (0..Self::DEMO_SPHERES)
            .map(|index| {
                let step = index as f32 / (Self::DEMO_SPHERES - 1) as f32;
                Object {
                    mesh: icosphere(2),
                    transform: Mat4::from_translation(Vec3::new(
                        1.2 * (index as f32 - 2.0),
                        0.5,
                        0.0,
                    )) * Mat4::from_scale(Vec3::splat(0.5)),
                    material: Material::glossy(terracotta, 0.4, 0.2).with_metalness(1.0 - step),
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
        // A 1.5 m × 1.5 m quad, up and off to the right — outside the
        // default framing, so it lights the scene without appearing in it.
        objects.push(Object {
            mesh: ground_plane(0.75),
            transform: Mat4::from_translation(Vec3::new(2.5, 3.5, 1.0)),
            material: Material::emitter(acescg_from_rec709(Vec3::new(1.0, 0.85, 0.6)) * 12.0),
        });
        // Above and behind, looking slightly down the row so the floor
        // (and its reflections) fill the lower frame.
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 8.5),
            look_at: Vec3::new(0.0, 0.5, 0.0),
            vfov_degrees: 40.0,
        };
        // Dim enough that the quad light reads as the key (soft shadow,
        // warm cast) over the sky's ambient fill. Neutral gray is the same
        // in ACEScg as in Rec.709 — no conversion to blur the exact sky
        // values tests probe for.
        Self::new(gpu, &objects, camera, Vec3::splat(0.4))
    }

    /// The scene's TLAS, ready to bind for ray queries.
    #[must_use]
    pub fn tlas(&self) -> &AccelerationStructure {
        &self.tlas
    }

    /// The scene table: the one buffer of addresses kernels reach all
    /// shared scene data through (geometry records, materials, lights).
    #[must_use]
    pub fn table(&self) -> &Buffer {
        &self.table
    }

    /// The constant sky radiance (`ACEScg`).
    #[must_use]
    pub fn sky(&self) -> Vec3 {
        self.sky
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
