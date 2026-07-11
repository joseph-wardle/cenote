//! Procedural test solids: the icosphere and ground plane the furnace,
//! estimator, and demo scenes are built from. Self-contained geometry —
//! positions, shading normals, UVs, and winding — with no GPU or residency
//! dependency, so the correctness properties they must hold (vertices on
//! the sphere, counter-clockwise-outward winding, one normal per vertex)
//! live and are tested here, apart from the residency machinery in the
//! parent module.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use super::Mesh;

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
    // No parameterization — a seamless sphere unwrap isn't worth inventing
    // for a test solid; textured lookups on it read texel (0, 0).
    mesh.uvs = vec![Vec2::ZERO; mesh.positions.len()];
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
    let uvs = vec![Vec2::ZERO; positions.len()];
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
        uvs,
        triangles,
    }
}

/// A square ground plane in the XZ plane at y = 0, spanning ±`half_extent`
/// meters, normal +Y, unit UVs with v growing toward +z (image row 0 maps
/// to the far edge, matching the top-of-image = far convention a camera
/// looking down −z sees).
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
        uvs: vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
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

    #[test]
    fn indices_stay_in_bounds() {
        let mesh = icosphere(2);
        let count = mesh.positions.len() as u32;
        assert!(mesh.triangles.iter().flatten().all(|&i| i < count));
    }
}
