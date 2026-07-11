//! Procedural tessellation of pbrt's shapes into inline meshes: the
//! trianglemesh stream reader and the sphere/disk generators. Geometry
//! only — the mapper places and materials the result.

use cenote::scene::description::MeshSource;
use cenote::{Error, Result};
use glam::Vec3;

use crate::parse::Directive;

/// A `trianglemesh` shape's streams, verbatim in object space. `flip`
/// (trap 4's XOR) negates authored normals and reverses winding —
/// winding also drives derived normals, so orientation survives either
/// way.
pub(super) fn trianglemesh(directive: &Directive, flip: bool) -> Result<MeshSource> {
    let params = &directive.params;
    let triples = |name: &str, types: &[&str]| -> Result<Option<Vec<[f32; 3]>>> {
        let Some(param) = params.take(name, types)? else {
            return Ok(None);
        };
        let floats = param.as_floats()?;
        if floats.len() % 3 != 0 {
            return Err(Error::SceneFormat(format!(
                "{}: \"{name}\" needs whole (x, y, z) triples",
                param.location
            )));
        }
        Ok(Some(
            floats
                .chunks_exact(3)
                .map(|triple| [triple[0] as f32, triple[1] as f32, triple[2] as f32])
                .collect(),
        ))
    };
    let positions = triples("P", &["point3", "point"])?.ok_or_else(|| {
        Error::SceneFormat(format!(
            "{}: trianglemesh has no \"point3 P\"",
            directive.location
        ))
    })?;
    let mut normals = triples("N", &["normal", "normal3"])?;
    if flip && let Some(normals) = &mut normals {
        for normal in normals {
            *normal = normal.map(|component| -component);
        }
    }
    let uvs = match params.take("uv", &["point2", "float", "vector2"])? {
        Some(param) => {
            let floats = param.as_floats()?;
            if floats.len() % 2 != 0 {
                return Err(Error::SceneFormat(format!(
                    "{}: \"uv\" needs whole (u, v) pairs",
                    param.location
                )));
            }
            Some(
                floats
                    .chunks_exact(2)
                    .map(|pair| [pair[0] as f32, pair[1] as f32])
                    .collect(),
            )
        }
        None => None,
    };
    let triangles = match params.take("indices", &["integer"])? {
        Some(param) => {
            let values = param.as_floats()?;
            if values.len() % 3 != 0 {
                return Err(Error::SceneFormat(format!(
                    "{}: \"indices\" needs whole triangles",
                    param.location
                )));
            }
            let mut triangles = Vec::with_capacity(values.len() / 3);
            for triple in values.chunks_exact(3) {
                let mut triangle = [0u32; 3];
                for (corner, value) in triangle.iter_mut().zip(triple) {
                    if *value < 0.0 || *value > f64::from(u32::MAX) {
                        return Err(Error::SceneFormat(format!(
                            "{}: index {value} is out of range",
                            param.location
                        )));
                    }
                    *corner = *value as u32;
                }
                triangles.push(triangle);
            }
            triangles
        }
        // pbrt allows exactly one implicit triangle.
        None if positions.len() == 3 => vec![[0, 1, 2]],
        None => {
            return Err(Error::SceneFormat(format!(
                "{}: trianglemesh has no \"integer indices\"",
                directive.location
            )));
        }
    };
    let triangles = if flip {
        triangles.into_iter().map(|[a, b, c]| [a, c, b]).collect()
    } else {
        triangles
    };
    Ok(MeshSource::Inline {
        positions,
        normals,
        uvs,
        triangles,
    })
}

/// A pbrt sphere, tessellated: poles on the object-space z axis,
/// analytic normals, pbrt's parameterization for UVs (`u` around z,
/// `v = 0` at the +z pole). 32 rings × 64 segments keeps silhouettes
/// clean at corpus scales.
pub(super) fn sphere_mesh(radius: f32) -> MeshSource {
    const RINGS: u32 = 32;
    const SEGMENTS: u32 = 64;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for ring in 0..=RINGS {
        let v = ring as f32 / RINGS as f32;
        let theta = v * std::f32::consts::PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for segment in 0..=SEGMENTS {
            let u = segment as f32 / SEGMENTS as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
            positions.push((normal * radius).into());
            normals.push(normal.into());
            uvs.push([u, v]);
        }
    }
    let mut triangles = Vec::new();
    let row = SEGMENTS + 1;
    for ring in 0..RINGS {
        for segment in 0..SEGMENTS {
            let a = ring * row + segment;
            let b = a + row;
            // The two pole rows collapse to points; their degenerate
            // half of each quad is skipped.
            if ring != 0 {
                triangles.push([a, b, a + 1]);
            }
            if ring != RINGS - 1 {
                triangles.push([a + 1, b, b + 1]);
            }
        }
    }
    MeshSource::Inline {
        positions,
        normals: Some(normals),
        uvs: Some(uvs),
        triangles,
    }
}

/// A pbrt disk: radius `radius` in the plane `z = height`, facing +z,
/// pbrt's radial parameterization (`v = 1` at the center).
pub(super) fn disk_mesh(radius: f32, height: f32) -> MeshSource {
    const SEGMENTS: u32 = 64;
    let mut positions = vec![[0.0, 0.0, height]];
    let mut uvs = vec![[0.0, 1.0]];
    for segment in 0..=SEGMENTS {
        let u = segment as f32 / SEGMENTS as f32;
        let phi = u * std::f32::consts::TAU;
        positions.push([radius * phi.cos(), radius * phi.sin(), height]);
        uvs.push([u, 0.0]);
    }
    let triangles = (0..SEGMENTS)
        .map(|segment| [0, segment + 1, segment + 2])
        .collect();
    MeshSource::Inline {
        positions,
        normals: Some(vec![[0.0, 0.0, 1.0]; SEGMENTS as usize + 2]),
        uvs: Some(uvs),
        triangles,
    }
}
