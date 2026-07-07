//! The procedural M0 scene (decision D-008): a subdivided icosphere resting
//! on a ground plane — two BLASes, one TLAS with two instances, a fixed
//! pinhole camera, zero file I/O (real scene formats are M2's job). The
//! sphere is deliberately faceted: geometric normals rendered as color make
//! winding or handedness mistakes instantly visible.

use std::collections::HashMap;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::error::Result;
use crate::gpu::{AccelerationStructure, Buffer, Context, TlasInstance};

/// A triangle mesh on the host: tightly packed positions plus index triples.
pub struct Mesh {
    /// Vertex positions, in meters, in object space.
    pub positions: Vec<Vec3>,
    /// Counter-clockwise-outward index triples into `positions`.
    pub triangles: Vec<[u32; 3]>,
}

/// One mesh resident on the GPU. The vertex and index buffers stay alive
/// past the BLAS build: the primary kernel fetches triangle corners from
/// them to compute geometric normals (D-017).
struct GpuMesh {
    blas: AccelerationStructure,
    vertices: Buffer,
    indices: Buffer,
}

/// One entry of the geometry lookup table, indexed by instance custom index.
/// Mirrors `struct GeometryRecord` in `shaders/primary.slang` — the kernel
/// follows these addresses to fetch the hit triangle's corners (D-017).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeometryRecord {
    positions: vk::DeviceAddress,
    indices: vk::DeviceAddress,
}

/// The scene, resident on the GPU and ready to trace against.
pub struct Scene {
    // Declared before `meshes`: the TLAS dies before the BLASes its
    // instances reference.
    tlas: AccelerationStructure,
    /// One [`GeometryRecord`] per instance custom index.
    geometry: Buffer,
    #[expect(
        dead_code,
        reason = "GPU residency: the BLASes and the buffers `geometry` points into"
    )]
    meshes: Vec<GpuMesh>,
    camera: Camera,
}

impl Scene {
    /// Upload and build the M0 demo scene: a unit icosphere resting on a
    /// 10 m × 10 m ground plane at y = 0.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from upload or acceleration-structure builds.
    pub fn demo(gpu: &Context) -> Result<Self> {
        let meshes = vec![
            upload_mesh(gpu, "scene.sphere", &icosphere(2))?,
            upload_mesh(gpu, "scene.plane", &ground_plane(5.0))?,
        ];
        // custom_index = position in `meshes`, so a hit leads back to the
        // right vertex data.
        let instances = [
            TlasInstance {
                blas: &meshes[0].blas,
                transform: Mat4::from_translation(Vec3::Y),
                custom_index: 0,
            },
            TlasInstance {
                blas: &meshes[1].blas,
                transform: Mat4::IDENTITY,
                custom_index: 1,
            },
        ];
        let tlas = gpu.build_tlas("scene.tlas", &instances)?;

        let records: Vec<GeometryRecord> = meshes
            .iter()
            .map(|mesh| GeometryRecord {
                positions: mesh.vertices.device_address(),
                indices: mesh.indices.device_address(),
            })
            .collect();
        let geometry = gpu.upload_buffer(
            "scene.geometry",
            bytemuck::cast_slice(&records),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        // Slightly above and behind the sphere (center (0, 1, 0)), looking
        // down at it so the ground plane fills the lower frame.
        let camera = Camera {
            position: Vec3::new(0.0, 1.8, 5.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            vfov_degrees: 40.0,
        };

        Ok(Self {
            tlas,
            geometry,
            meshes,
            camera,
        })
    }

    /// The scene's TLAS, ready to bind for ray queries.
    #[must_use]
    pub fn tlas(&self) -> &AccelerationStructure {
        &self.tlas
    }

    /// The geometry lookup table: one `{positions, indices}` address pair
    /// per instance custom index (D-017).
    #[must_use]
    pub fn geometry(&self) -> &Buffer {
        &self.geometry
    }

    /// The scene's camera.
    #[must_use]
    pub fn camera(&self) -> &Camera {
        &self.camera
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
    /// parallel to world up. Both are programmer bugs (D-010).
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

fn upload_mesh(gpu: &Context, name: &str, mesh: &Mesh) -> Result<GpuMesh> {
    // BUILD_INPUT for the BLAS build; STORAGE + device address so the
    // primary kernel can fetch triangle corners afterwards (D-017).
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

    /// The renderer-breaking bug this scene exists to catch (D-008): for a
    /// convex solid centered at the origin, a counter-clockwise-outward face
    /// has its geometric normal pointing away from the origin.
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

    /// The step-6 checkpoint: two BLASes and the TLAS build without errors
    /// on real hardware (validation complaints appear via the debug
    /// messenger in the log).
    #[test]
    fn demo_scene_builds() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        Scene::demo(&gpu).expect("demo scene should build");
    }
}
