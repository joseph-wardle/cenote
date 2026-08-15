//! Procedural tessellation of pbrt's shapes into inline meshes: the
//! trianglemesh stream reader and the sphere/disk generators. Geometry
//! only — the mapper places and materials the result.

use cenote::scene::description::{CurveBasis, CurveType, CurveWrap, MeshSource};
use cenote::scene::segment_count;
use cenote::{Error, Result};
use glam::Vec3;

use crate::parse::Directive;

/// One `Shape "curve"`, in the form a batch accumulates it: the control
/// vertices still in pbrt's flat `[x, y, z, …]`, the segments they imply,
/// and the two end widths.
///
/// pbrt spells one curve per `Shape` directive — bunny-fur is 1.5 million
/// of them in a row — so this is the unit the mapper coalesces, and it
/// borrows rather than copies.
pub(super) struct CurveShape<'a> {
    pub points: &'a [f64],
    pub segments: usize,
    pub width0: f32,
    pub width1: f32,
    /// Cubic bezier or b-spline; the key a batch cannot straddle.
    pub basis: CurveBasis,
}

/// Read one `Shape "curve"`.
///
/// The semantics are pbrt-v4's own (`shapes.cpp`, `Curve::Create`), and
/// the two that matter are that `width` is a **diameter** — the
/// intersector rejects on `ptCurveDist2 > Sqr(hitWidth) * 0.25` — and that
/// a segment's widths come out as `Lerp(seg / nSegments, width0, width1)`,
/// which is cenote's `varying` interpolation exactly. Nothing is
/// approximated: what cenote's curves cannot express is refused.
pub(super) fn curve(directive: &Directive) -> Result<CurveShape<'_>> {
    let params = &directive.params;
    let location = &directive.location;

    // pbrt elevates a quadratic to cubic on the way in. Cenote's bases are
    // `UsdGeomBasisCurves`', which has no quadratic row, and elevating
    // silently would be a different curve wearing the same control points.
    let degree = params.integer("degree")?.unwrap_or(3);
    if degree != 3 {
        return Err(Error::SceneFormat(format!(
            "{location}: curve \"degree\" {degree} is not supported — only cubic (3) curves \
             have a cenote equivalent"
        )));
    }
    let basis = match params.string("basis")?.unwrap_or("bezier") {
        "bezier" => CurveBasis::Bezier,
        "bspline" => CurveBasis::BSpline,
        other => {
            return Err(Error::SceneFormat(format!(
                "{location}: curve \"basis\" \"{other}\" is not a pbrt basis"
            )));
        }
    };
    // `flat` and `cylinder` are the *same geometry* in pbrt: one
    // ray-perpendicular projected hit test for both, differing only in the
    // shading `dpdv` they hand the BSDF, which a swept tube derives from
    // its own surface. `ribbon` genuinely differs — an oriented flat quad
    // driven by an authored `N`, which the sweep has no way to honour.
    match params.string("type")?.unwrap_or("flat") {
        "flat" | "cylinder" => {}
        "ribbon" => {
            return Err(Error::SceneFormat(format!(
                "{location}: curve \"type\" \"ribbon\" is not supported — an oriented ribbon is \
                 not a swept tube, and cenote renders every curve round"
            )));
        }
        other => {
            return Err(Error::SceneFormat(format!(
                "{location}: curve \"type\" \"{other}\" is not a pbrt curve type"
            )));
        }
    }
    // A hint to pbrt's own intersector, not a property of the curve.
    // Consumed so it does not read as a parameter this importer dropped.
    let _ = params.take("splitdepth", &["integer", "float"])?;

    let param = params
        .take("P", &["point3", "point"])?
        .ok_or_else(|| Error::SceneFormat(format!("{location}: curve has no \"point3 P\"")))?;
    let points = param.as_floats()?;
    if points.len() % 3 != 0 {
        return Err(Error::SceneFormat(format!(
            "{}: curve \"P\" needs whole (x, y, z) triples",
            param.location
        )));
    }
    let vertices = points.len() / 3;
    // The renderer's own table, so the importer cannot drift from it — and
    // resolved here, before the batch is touched, because a curve no span
    // fits would otherwise poison a batch of a million siblings with an
    // error naming none of them.
    let segments = segment_count(vertices, CurveType::Cubic, basis, CurveWrap::Nonperiodic)
        .map_err(|_| {
            Error::SceneFormat(format!(
                "{location}: a cubic {basis} curve cannot have {vertices} control vertices"
            ))
        })?;

    let width = params.float("width")?.unwrap_or(1.0);
    Ok(CurveShape {
        points,
        segments,
        width0: params.float("width0")?.unwrap_or(width),
        width1: params.float("width1")?.unwrap_or(width),
        basis,
    })
}

/// pbrt's flat `[x, y, z, …]` as the triples every cenote stream holds.
pub(super) fn triples(floats: &[f64]) -> impl Iterator<Item = [f32; 3]> {
    floats
        .chunks_exact(3)
        .map(|triple| [triple[0] as f32, triple[1] as f32, triple[2] as f32])
}

/// pbrt inverts `t` at every image-texture lookup (`st[1] = 1 - st[1]`
/// against top-row-first image memory); cenote samples `v` as stored,
/// texel row 0 at `v = 0`. Storing `1 - v` at import makes every
/// downstream lookup read the texel pbrt would.
fn flip_v(uvs: &mut [[f32; 2]]) {
    for uv in uvs {
        uv[1] = 1.0 - uv[1];
    }
}

/// A `trianglemesh` shape's streams, verbatim in object space — except
/// `v`, which lands pre-flipped (see `flip_v`). `flip` (trap 4's XOR)
/// negates authored normals and reverses winding — winding also drives
/// derived normals, so orientation survives either way.
pub(super) fn trianglemesh(directive: &Directive, flip: bool) -> Result<MeshSource> {
    let params = &directive.params;
    let triples_of = |name: &str, types: &[&str]| -> Result<Option<Vec<[f32; 3]>>> {
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
        Ok(Some(triples(floats).collect()))
    };
    let positions = triples_of("P", &["point3", "point"])?.ok_or_else(|| {
        Error::SceneFormat(format!(
            "{}: trianglemesh has no \"point3 P\"",
            directive.location
        ))
    })?;
    let mut normals = triples_of("N", &["normal", "normal3"])?;
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
            let mut uvs: Vec<[f32; 2]> = floats
                .chunks_exact(2)
                .map(|pair| [pair[0] as f32, pair[1] as f32])
                .collect();
            flip_v(&mut uvs);
            Some(uvs)
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

/// A `bilinearmesh` shape's one implicit patch as two triangles —
/// `None` when the mesh is more than that (explicit "indices" or extra
/// control points; the mapper warns and skips those). Corner order is
/// pbrt's (p00, p10, p01, p11); the (0,1,3)(0,3,2) split keeps each
/// triangle's winding normal on the patch's `dpdu × dpdv` side, and
/// absent authored UVs default to the patch parameterization. `flip` is
/// trap 4's `ReverseOrientation`, as in [`trianglemesh`].
pub(super) fn bilinearmesh(directive: &Directive, flip: bool) -> Result<Option<MeshSource>> {
    let params = &directive.params;
    if params.take("indices", &["integer"])?.is_some() {
        return Ok(None);
    }
    let positions_param = params.take("P", &["point3", "point"])?.ok_or_else(|| {
        Error::SceneFormat(format!(
            "{}: bilinearmesh has no \"point3 P\"",
            directive.location
        ))
    })?;
    let floats = positions_param.as_floats()?;
    if floats.len() != 12 {
        return Ok(None);
    }
    let positions: Vec<[f32; 3]> = floats
        .chunks_exact(3)
        .map(|triple| [triple[0] as f32, triple[1] as f32, triple[2] as f32])
        .collect();
    let mut uvs: Vec<[f32; 2]> = match params.take("uv", &["point2", "float", "vector2"])? {
        Some(param) => {
            let floats = param.as_floats()?;
            if floats.len() != 8 {
                return Err(Error::SceneFormat(format!(
                    "{}: bilinearmesh \"uv\" needs four (u, v) pairs",
                    param.location
                )));
            }
            floats
                .chunks_exact(2)
                .map(|pair| [pair[0] as f32, pair[1] as f32])
                .collect()
        }
        None => vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    };
    flip_v(&mut uvs);
    let triangles = if flip {
        vec![[0, 3, 1], [0, 2, 3]]
    } else {
        vec![[0, 1, 3], [0, 3, 2]]
    };
    Ok(Some(MeshSource::Inline {
        positions,
        normals: None,
        uvs: Some(uvs),
        triangles,
    }))
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
    flip_v(&mut uvs);
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
    flip_v(&mut uvs);
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
