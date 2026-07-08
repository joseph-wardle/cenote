//! EXR image input and output.
//!
//! M0 files are linear RGBA with no color transform: the renderer is
//! writing *data* — geometric normals mapped to color — not color. `ACEScg`
//! and real radiance arrive with M1. View results in `tev`, which
//! auto-refreshes on file change. Reading exists for the golden-image
//! tests, which compare fresh renders against checked-in EXRs.

use std::path::Path;

use crate::error::Result;

/// Write row-major RGBA `f32` pixels (pixel (0, 0) top-left, the crate-wide
/// convention) to `path` as a linear EXR.
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
    assert_eq!(
        pixels.len() as u64,
        u64::from(width) * u64::from(height) * 4,
        "pixel count doesn't match image dimensions"
    );
    let width = width as usize;
    exr::prelude::write_rgba_file(path, width, height as usize, |x, y| {
        let idx = (y * width + x) * 4;
        (
            pixels[idx],
            pixels[idx + 1],
            pixels[idx + 2],
            pixels[idx + 3],
        )
    })?;
    Ok(())
}

/// Read an EXR's first layer back as `(width, height, pixels)` in the same
/// row-major RGBA `f32` layout that [`write_exr`] takes — `f32` channels
/// round-trip losslessly.
///
/// # Errors
///
/// [`crate::Error::Image`] if the file can't be opened or decoded.
pub fn read_exr(path: &Path) -> Result<(u32, u32, Vec<f32>)> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            (
                resolution.width(),
                vec![0.0_f32; resolution.width() * resolution.height() * 4],
            )
        },
        |(width, pixels): &mut (usize, Vec<f32>), position, (r, g, b, a): (f32, f32, f32, f32)| {
            let idx = (position.y() * *width + position.x()) * 4;
            pixels[idx..idx + 4].copy_from_slice(&[r, g, b, a]);
        },
    )?;
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
}
