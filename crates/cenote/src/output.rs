//! EXR image input and output.
//!
//! Written files are linear `ACEScg` radiance, and their headers say so:
//! the chromaticities attribute carries the AP1 primaries, so a
//! color-managed viewer (`tev`, Nuke) converts to its display correctly
//! instead of assuming Rec. 709 and desaturating everything. Display
//! transforms are the tonemap kernel's job and never touch disk. Reading
//! exists for the golden-image tests and the demo environment, which
//! arrive as EXRs too.

use std::path::Path;

use crate::error::Result;

/// `ACEScg`'s color space on the CIE xy diagram — the AP1 primaries and
/// the ACES white point (≈ D60), from ACES spec S-2014-004.
const ACESCG_CHROMATICITIES: exr::meta::attribute::Chromaticities =
    exr::meta::attribute::Chromaticities {
        red: exr::math::Vec2(0.713, 0.293),
        green: exr::math::Vec2(0.165, 0.830),
        blue: exr::math::Vec2(0.128, 0.044),
        white: exr::math::Vec2(0.321_68, 0.337_67),
    };

/// Write row-major RGBA `f32` pixels (pixel (0, 0) top-left, the crate-wide
/// convention) to `path` as a linear EXR tagged with the `ACEScg`
/// chromaticities.
///
/// # Errors
///
/// [`crate::Error::Image`] if encoding or the underlying I/O fails.
///
/// # Panics
///
/// If `pixels` doesn't hold exactly `width × height` RGBA quads — a
/// programmer bug.
pub fn write_exr(path: &Path, width: u32, height: u32, pixels: &[f32]) -> Result<()> {
    use exr::prelude::{Image, SpecificChannels, Vec2, WritableImage};
    assert_eq!(
        pixels.len() as u64,
        u64::from(width) * u64::from(height) * 4,
        "pixel count doesn't match image dimensions"
    );
    let width = width as usize;
    let channels = SpecificChannels::rgba(|position: Vec2<usize>| {
        let idx = (position.y() * width + position.x()) * 4;
        (
            pixels[idx],
            pixels[idx + 1],
            pixels[idx + 2],
            pixels[idx + 3],
        )
    });
    let mut image = Image::from_channels((width, height as usize), channels);
    image.attributes.chromaticities = Some(ACESCG_CHROMATICITIES);
    image.write().to_file(path)?;
    Ok(())
}

/// Write a frame and its AOVs as one multi-layer EXR — the batch CLI's
/// output. Nuke-safe naming: beauty is the bare `R`/`G`/`B`/`A`, depth the
/// bare de-facto-standard `Z`, the denoiser guides `albedo.R/G/B` and
/// `normal.X/Y/Z` — no dots *within* a layer name. Color channels store
/// f16 (compositing's convention; the quantization is display-invisible),
/// `Z` stays f32 (depths span orders of magnitude, and +∞ marks the sky).
/// `beauty`/`albedo`/`normal` are row-major RGBA `f32` quads, `depth` one
/// `f32` per pixel — exactly [`crate::render::FilmAverages`]'s layout.
///
/// # Errors
///
/// [`crate::Error::Image`] if encoding or the underlying I/O fails.
///
/// # Panics
///
/// If any slice doesn't hold exactly `width × height` of its layout — a
/// programmer bug.
pub fn write_aov_exr(
    path: &Path,
    width: u32,
    height: u32,
    beauty: &[f32],
    albedo: &[f32],
    normal: &[f32],
    depth: &[f32],
) -> Result<()> {
    use exr::image::{Encoding, Image, Layer};
    use exr::prelude::{AnyChannel, AnyChannels, FlatSamples, SmallVec, WritableImage, f16};

    let texels = (u64::from(width) * u64::from(height)) as usize;
    for (label, quads) in [("beauty", beauty), ("albedo", albedo), ("normal", normal)] {
        assert_eq!(
            quads.len(),
            texels * 4,
            "{label} pixel count doesn't match image dimensions"
        );
    }
    assert_eq!(
        depth.len(),
        texels,
        "depth pixel count doesn't match image dimensions"
    );

    let f16_channel = |quads: &[f32], offset: usize| {
        FlatSamples::F16(
            quads
                .iter()
                .skip(offset)
                .step_by(4)
                .map(|value| f16::from_f32(*value))
                .collect(),
        )
    };
    let mut channels = SmallVec::new();
    for (name, offset) in [("R", 0), ("G", 1), ("B", 2), ("A", 3)] {
        channels.push(AnyChannel::new(name, f16_channel(beauty, offset)));
    }
    channels.push(AnyChannel::new("Z", FlatSamples::F32(depth.to_vec())));
    for (name, offset) in [("albedo.R", 0), ("albedo.G", 1), ("albedo.B", 2)] {
        channels.push(AnyChannel::new(name, f16_channel(albedo, offset)));
    }
    for (name, offset) in [("normal.X", 0), ("normal.Y", 1), ("normal.Z", 2)] {
        channels.push(AnyChannel::new(name, f16_channel(normal, offset)));
    }

    let layer = Layer::new(
        (width as usize, height as usize),
        exr::meta::header::LayerAttributes::default(),
        // Zip scanlines: what a production render would ship — the default
        // is RLE tiles, ~2× the size on real frames.
        Encoding::SMALL_LOSSLESS,
        AnyChannels::sort(channels),
    );
    let mut image = Image::from_layer(layer);
    image.attributes.chromaticities = Some(ACESCG_CHROMATICITIES);
    image.write().to_file(path)?;
    Ok(())
}

/// Read an EXR's first layer back as `(width, height, pixels)` in the same
/// row-major RGBA `f32` layout that [`write_exr`] takes — `f32` channels
/// round-trip losslessly.
///
/// # Errors
///
/// [`crate::Error::Io`] if the file can't be read, [`crate::Error::Image`]
/// if it can't be decoded.
pub fn read_exr(path: &Path) -> Result<(u32, u32, Vec<f32>)> {
    read_exr_bytes(&std::fs::read(path)?)
}

/// [`read_exr`] from an in-memory EXR — how the embedded demo environment
/// loads. A missing alpha channel reads as 1.
///
/// # Errors
///
/// [`crate::Error::Image`] if `bytes` don't decode as an EXR.
pub fn read_exr_bytes(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>)> {
    use exr::prelude::{ReadChannels, ReadLayers};
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |resolution, _channels| {
                (
                    resolution.width(),
                    vec![0.0_f32; resolution.width() * resolution.height() * 4],
                )
            },
            |(width, pixels): &mut (usize, Vec<f32>),
             position,
             (r, g, b, a): (f32, f32, f32, f32)| {
                let idx = (position.y() * *width + position.x()) * 4;
                pixels[idx..idx + 4].copy_from_slice(&[r, g, b, a]);
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(std::io::Cursor::new(bytes))?;
    let size = image.layer_data.size;
    let (_, pixels) = image.layer_data.channel_data.pixels;
    Ok((size.width() as u32, size.height() as u32, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f32` EXR channels are lossless, so write → read must reproduce the
    /// pixels exactly — the property the golden comparison stands on.
    #[test]
    fn exr_round_trips_exactly() {
        let pixels: Vec<f32> = (0..3 * 2 * 4).map(|i| i as f32 * 0.125).collect();
        let path =
            std::env::temp_dir().join(format!("cenote-roundtrip-{}.exr", std::process::id()));
        write_exr(&path, 3, 2, &pixels).expect("write");
        let read = read_exr(&path);
        let _ = std::fs::remove_file(&path);

        let (width, height, read_back) = read.expect("read");
        assert_eq!((width, height), (3, 2));
        assert_eq!(read_back, pixels);
    }

    /// The multi-layer EXR carries exactly the channels the plan names, in
    /// the types it names: beauty as bare `R/G/B/A` and the guides as
    /// `albedo.*`/`normal.*` in f16, depth as the bare f32 `Z` — where +∞
    /// (a sky pixel) must survive the trip exactly. And the header still
    /// declares `ACEScg`, like every file this module writes.
    #[test]
    fn aov_exr_layers_carry_the_named_channels() {
        use exr::prelude::{ReadChannels, ReadLayers, f16};

        let beauty: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
        let albedo = vec![0.5_f32; 8];
        let normal = vec![0.25_f32; 8];
        let depth = vec![1.5_f32, f32::INFINITY];
        let path = std::env::temp_dir().join(format!("cenote-aov-{}.exr", std::process::id()));
        write_aov_exr(&path, 2, 1, &beauty, &albedo, &normal, &depth).expect("write");

        let image = exr::prelude::read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .first_valid_layer()
            .all_attributes()
            .from_file(&path);
        let _ = std::fs::remove_file(&path);
        let image = image.expect("read");
        assert_eq!(
            image.attributes.chromaticities,
            Some(ACESCG_CHROMATICITIES),
            "the multi-layer file must declare ACEScg too"
        );

        let channels = &image.layer_data.channel_data.list;
        let names: Vec<String> = channels
            .iter()
            .map(|channel| channel.name.to_string())
            .collect();
        // Alphabetical — the EXR spec's required channel order.
        assert_eq!(
            names,
            [
                "A", "B", "G", "R", "Z", "albedo.B", "albedo.G", "albedo.R", "normal.X",
                "normal.Y", "normal.Z"
            ]
        );
        let samples = |name: &str| {
            &channels
                .iter()
                .find(|channel| channel.name.eq(name))
                .expect("channel present")
                .sample_data
        };
        let f16s = |name: &str| match samples(name) {
            exr::prelude::FlatSamples::F16(values) => values.clone(),
            other => panic!("{name} should store f16, holds {other:?}"),
        };
        assert_eq!(f16s("R"), [f16::from_f32(0.0), f16::from_f32(1.0)]);
        assert_eq!(f16s("albedo.G"), [f16::from_f32(0.5); 2]);
        assert_eq!(f16s("normal.Z"), [f16::from_f32(0.25); 2]);
        match samples("Z") {
            exr::prelude::FlatSamples::F32(values) => {
                assert_eq!(
                    values[0].to_bits(),
                    1.5_f32.to_bits(),
                    "f32 depth round-trips exactly"
                );
                assert!(
                    values[1].is_infinite() && values[1].is_sign_positive(),
                    "+inf depth must survive: {}",
                    values[1]
                );
            }
            other => panic!("Z should store f32, holds {other:?}"),
        }
    }

    /// Every written file declares what its numbers mean: the header must
    /// carry the `ACEScg` chromaticities, or a color-managed viewer falls
    /// back to assuming Rec. 709 (the EXR spec's default).
    #[test]
    fn written_headers_carry_acescg_chromaticities() {
        let path = std::env::temp_dir().join(format!("cenote-chroma-{}.exr", std::process::id()));
        write_exr(&path, 1, 1, &[0.0; 4]).expect("write");
        let meta = exr::meta::MetaData::read_from_file(&path, false);
        let _ = std::fs::remove_file(&path);

        let chromaticities = meta.expect("read meta").headers[0]
            .shared_attributes
            .chromaticities
            .expect("chromaticities attribute present");
        assert_eq!(chromaticities, ACESCG_CHROMATICITIES);
    }
}
