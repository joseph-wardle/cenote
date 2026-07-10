//! Environment image conversion. pbrt-v4 infinite lights use *square
//! equal-area octahedral* images (Clarberg's mapping — the whole sphere
//! unfolded onto a square, +z at the center); cenote's environment is an
//! equirectangular EXR. The importer resamples one into the other at
//! import time, baking in what cenote's environment object can't carry:
//! the light's world orientation and its photometric scale. Sampling
//! wraps across the octahedron's seams exactly as pbrt's own
//! `WrapEqualAreaSquare` does, so seam texels filter correctly instead of
//! clamping.
//!
//! Image-less infinite lights (`"rgb L"` and no filename) become a tiny
//! constant equirect EXR — the environment always reads from a file, and
//! two texels of sky cost nothing.

use std::path::Path;

use cenote::{Error, Result};
use glam::{Mat3, Vec2, Vec3};

/// Output width cap: a resample never writes more than a 4096×2048
/// equirect (the same ceiling texture prep applies to its inputs).
const MAX_WIDTH: u32 = 4096;

/// Resample an equal-area octahedral EXR into the equirect EXR cenote
/// reads, with radiance scaled by `scale` (pbrt's photometric scale,
/// baked here because the environment object carries no scalar) and
/// directions mapped through `world_from_light` (the light's `CTM`
/// orientation, already conjugated into cenote's coordinate convention by
/// the caller). Values stay linear `Rec.709` — cenote's environment
/// loader owns the conversion to the working space.
///
/// # Errors
///
/// [`Error::Scene`] when the source doesn't read, isn't an EXR, isn't
/// square, or the orientation isn't invertible.
pub(crate) fn resample_octahedral(
    source: &Path,
    world_from_light: Mat3,
    scale: f32,
    out: &Path,
) -> Result<()> {
    let (width, height, texels) = cenote::output::read_exr(source).map_err(|error| {
        Error::Scene(format!(
            "infinite light image \"{}\": {error} (only EXR images import)",
            source.display()
        ))
    })?;
    if width != height {
        return Err(Error::Scene(format!(
            "infinite light image \"{}\" is {width}×{height} — pbrt-v4 sky images are square \
             equal-area octahedral (pbrt's `imgtool makeequiarea` converts)",
            source.display()
        )));
    }
    let light_from_world = world_from_light.inverse();
    if !light_from_world.is_finite() {
        return Err(Error::Scene(format!(
            "infinite light image \"{}\" has a non-invertible orientation",
            source.display()
        )));
    }

    let out_width = (2 * width).min(MAX_WIDTH);
    let out_height = out_width / 2;
    let mut pixels = Vec::with_capacity(out_width as usize * out_height as usize * 4);
    for y in 0..out_height {
        for x in 0..out_width {
            let u = (x as f32 + 0.5) / out_width as f32;
            let v = (y as f32 + 0.5) / out_height as f32;
            let direction = equirect_direction(u, v);
            let light = (light_from_world * direction).normalize();
            let radiance = sample_octahedral(&texels, width, sphere_to_square(light)) * scale;
            pixels.extend_from_slice(&[radiance.x, radiance.y, radiance.z, 1.0]);
        }
    }
    cenote::output::write_exr(out, out_width, out_height, &pixels)
}

/// Write the constant-radiance sky an image-less `infinite` light
/// becomes. `radiance` is linear `Rec.709`, photometric scale already
/// applied.
///
/// # Errors
///
/// Whatever the EXR write returns.
pub(crate) fn write_constant(radiance: [f32; 3], out: &Path) -> Result<()> {
    let texel = [radiance[0], radiance[1], radiance[2], 1.0];
    let pixels: Vec<f32> = texel.repeat(2);
    cenote::output::write_exr(out, 2, 1, &pixels)
}

/// The direction a cenote equirect texel looks along — the inverse of the
/// environment's documented convention: `v` is polar angle over π with
/// row 0 at the +Y zenith, and the image's horizontal center faces −Z.
fn equirect_direction(u: f32, v: f32) -> Vec3 {
    let phi = 2.0 * std::f32::consts::PI * (u - 0.5);
    let theta = std::f32::consts::PI * v;
    let (sin_theta, cos_theta) = theta.sin_cos();
    Vec3::new(sin_theta * phi.sin(), cos_theta, -sin_theta * phi.cos())
}

/// Clarberg's equal-area sphere→square mapping, matching pbrt-v4's
/// `EqualAreaSphereToSquare` (with an exact `atan` in place of its
/// polynomial): +z lands at the square's center, −z at its corners.
#[expect(
    clippy::many_single_char_names,
    reason = "the variables are named what Clarberg's paper names them"
)]
fn sphere_to_square(direction: Vec3) -> Vec2 {
    let x = direction.x.abs();
    let y = direction.y.abs();
    let z = direction.z.abs();
    let r = (1.0 - z).max(0.0).sqrt();
    let (a, b) = (x.max(y), x.min(y));
    let b = if a == 0.0 { 0.0 } else { b / a };
    let mut phi = b.atan() * std::f32::consts::FRAC_2_PI;
    if x < y {
        phi = 1.0 - phi;
    }
    let mut v = phi * r;
    let mut u = r - v;
    if direction.z < 0.0 {
        (u, v) = (1.0 - v, 1.0 - u);
    }
    u = u.copysign(direction.x);
    v = v.copysign(direction.y);
    Vec2::new(0.5 * (u + 1.0), 0.5 * (v + 1.0))
}

/// Mirror a filter tap back into the square across the octahedron's
/// seams — pbrt's `WrapEqualAreaSquare`, texel-coordinate flavored.
fn wrap(uv: Vec2) -> Vec2 {
    let Vec2 { mut x, mut y } = uv;
    if x < 0.0 {
        x = -x;
        y = 1.0 - y;
    } else if x > 1.0 {
        x = 2.0 - x;
        y = 1.0 - y;
    }
    if y < 0.0 {
        x = 1.0 - x;
        y = -y;
    } else if y > 1.0 {
        x = 1.0 - x;
        y = 2.0 - y;
    }
    Vec2::new(x, y)
}

/// Bilinear lookup in the square, seam-wrapped per tap.
fn sample_octahedral(texels: &[f32], width: u32, uv: Vec2) -> Vec3 {
    let size = width as f32;
    let px = uv.x * size - 0.5;
    let py = uv.y * size - 0.5;
    let (x0, y0) = (px.floor(), py.floor());
    let (fx, fy) = (px - x0, py - y0);
    let mut sum = Vec3::ZERO;
    for (dx, dy, weight) in [
        (0.0, 0.0, (1.0 - fx) * (1.0 - fy)),
        (1.0, 0.0, fx * (1.0 - fy)),
        (0.0, 1.0, (1.0 - fx) * fy),
        (1.0, 1.0, fx * fy),
    ] {
        let tap = wrap(Vec2::new((x0 + dx + 0.5) / size, (y0 + dy + 0.5) / size));
        let tx = ((tap.x * size) as u32).min(width - 1);
        let ty = ((tap.y * size) as u32).min(width - 1);
        let texel = ((ty * width + tx) * 4) as usize;
        sum += weight * Vec3::new(texels[texel], texels[texel + 1], texels[texel + 2]);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pbrt's forward mapping (`EqualAreaSquareToSphere`), transcribed
    /// for the tests: synthesizing octahedral fixtures and checking the
    /// round trip against the inverse the resampler actually uses.
    #[expect(
        clippy::many_single_char_names,
        reason = "the variables are named what Clarberg's paper names them"
    )]
    fn square_to_sphere(u: f32, v: f32) -> Vec3 {
        let u = 2.0 * u - 1.0;
        let v = 2.0 * v - 1.0;
        let (up, vp) = (u.abs(), v.abs());
        let signed_distance = 1.0 - (up + vp);
        let d = signed_distance.abs();
        let r = 1.0 - d;
        let phi = (if r == 0.0 { 1.0 } else { (vp - up) / r + 1.0 }) * std::f32::consts::FRAC_PI_4;
        let z = (1.0 - r * r).copysign(signed_distance);
        let cos_phi = phi.cos().copysign(u);
        let sin_phi = phi.sin().copysign(v);
        let scale = r * (2.0 - r * r).max(0.0).sqrt();
        Vec3::new(cos_phi * scale, sin_phi * scale, z)
    }

    #[test]
    fn the_octahedral_mapping_matches_pbrt_landmarks() {
        // +z is the square's center; the axes land on the diagonals'
        // midpoints; −z unfolds to the corners.
        assert!(sphere_to_square(Vec3::Z).abs_diff_eq(Vec2::splat(0.5), 1e-6));
        assert!(sphere_to_square(Vec3::X).abs_diff_eq(Vec2::new(1.0, 0.5), 1e-6));
        assert!(sphere_to_square(Vec3::NEG_X).abs_diff_eq(Vec2::new(0.0, 0.5), 1e-6));
        assert!(sphere_to_square(Vec3::Y).abs_diff_eq(Vec2::new(0.5, 1.0), 1e-6));
        assert!(sphere_to_square(Vec3::NEG_Y).abs_diff_eq(Vec2::new(0.5, 0.0), 1e-6));
    }

    #[test]
    fn the_mapping_round_trips_across_the_sphere() {
        for i in 0..32 {
            for j in 0..32 {
                let u = (f64::from(i) + 0.5) as f32 / 32.0;
                let v = (f64::from(j) + 0.5) as f32 / 32.0;
                let direction = square_to_sphere(u, v);
                assert!(
                    (direction.length() - 1.0).abs() < 1e-5,
                    "unit at ({u}, {v})"
                );
                let back = sphere_to_square(direction);
                assert!(
                    back.abs_diff_eq(Vec2::new(u, v), 1e-4),
                    "({u}, {v}) → {direction} → {back}"
                );
            }
        }
    }

    /// End-to-end resample: an octahedral fixture colored by the octant
    /// of each texel's direction, pushed through the resampler, probed at
    /// directions computed from cenote's documented equirect convention.
    /// Pins direction handling, the scale bake, and the orientation
    /// transform in one pass.
    #[test]
    fn resampling_maps_directions_scale_and_orientation() {
        const SIZE: u32 = 64;
        let dir = std::env::temp_dir().join(format!("cenote-pbrt-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let octant_color = |direction: Vec3| {
            Vec3::new(
                if direction.x > 0.0 { 1.0 } else { 0.25 },
                if direction.y > 0.0 { 1.0 } else { 0.25 },
                if direction.z > 0.0 { 1.0 } else { 0.25 },
            )
        };
        let mut texels = Vec::new();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let direction = square_to_sphere(
                    (x as f32 + 0.5) / SIZE as f32,
                    (y as f32 + 0.5) / SIZE as f32,
                );
                let color = octant_color(direction);
                texels.extend_from_slice(&[color.x, color.y, color.z, 1.0]);
            }
        }
        let source = dir.join("sky-octa.exr");
        cenote::output::write_exr(&source, SIZE, SIZE, &texels).expect("write fixture");

        // Identity orientation, scale 2: every octant color doubles.
        let out = dir.join("sky-equirect.exr");
        resample_octahedral(&source, Mat3::IDENTITY, 2.0, &out).expect("resamples");
        let (width, height, pixels) = cenote::output::read_exr(&out).expect("reads back");
        assert_eq!((width, height), (2 * SIZE, SIZE));
        let probe = |u: f32, v: f32| {
            let x = ((u * width as f32) as u32).min(width - 1);
            let y = ((v * height as f32) as u32).min(height - 1);
            let texel = ((y * width + x) * 4) as usize;
            Vec3::new(pixels[texel], pixels[texel + 1], pixels[texel + 2])
        };
        // Probes chosen well inside octants (no component near zero).
        for (u, v) in [(0.15, 0.3), (0.4, 0.6), (0.65, 0.25), (0.9, 0.75)] {
            let direction = equirect_direction(u, v);
            assert!(
                direction.abs().min_element() > 0.05,
                "weak probe ({u}, {v})"
            );
            let expected = octant_color(direction) * 2.0;
            assert!(
                probe(u, v).abs_diff_eq(expected, 1e-3),
                "at ({u}, {v}) looking {direction}: {} ≠ {expected}",
                probe(u, v)
            );
        }

        // A quarter turn about +Y: what the sky shows toward −Z must now
        // be what the source held toward the rotated axis.
        let rotated = dir.join("sky-rotated.exr");
        let quarter = Mat3::from_rotation_y(std::f32::consts::FRAC_PI_2);
        resample_octahedral(&source, quarter, 1.0, &rotated).expect("resamples");
        let (width, height, pixels) = cenote::output::read_exr(&rotated).expect("reads back");
        let probe = |u: f32, v: f32| {
            let x = ((u * width as f32) as u32).min(width - 1);
            let y = ((v * height as f32) as u32).min(height - 1);
            let texel = ((y * width + x) * 4) as usize;
            Vec3::new(pixels[texel], pixels[texel + 1], pixels[texel + 2])
        };
        for (u, v) in [(0.15, 0.3), (0.65, 0.25)] {
            let direction = equirect_direction(u, v);
            let expected = octant_color(quarter.inverse() * direction);
            assert!(
                probe(u, v).abs_diff_eq(expected, 1e-3),
                "rotated at ({u}, {v}): {} ≠ {expected}",
                probe(u, v)
            );
        }

        // A non-square image is refused with the converting tool named.
        let bad = dir.join("sky-equirect.exr");
        let error = resample_octahedral(&bad, Mat3::IDENTITY, 1.0, &dir.join("x.exr")).unwrap_err();
        assert!(error.to_string().contains("square"), "{error}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_constant_sky_writes_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("cenote-pbrt-sky-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let path = dir.join("constant.exr");
        write_constant([0.5, 1.0, 2.0], &path).expect("writes");
        let (width, height, pixels) = cenote::output::read_exr(&path).expect("reads");
        assert_eq!((width, height), (2, 1));
        assert_eq!(&pixels[..3], &[0.5, 1.0, 2.0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
