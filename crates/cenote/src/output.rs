//! EXR image output.
//!
//! M0 files are linear RGBA with no color transform (D-015): the renderer is
//! writing *data* — a UV gradient now, geometric normals from m0-plan step 7
//! — not color. `ACEScg` and real radiance arrive with M1. View results in
//! `tev`, which auto-refreshes on file change (D-002).

use std::path::Path;

use crate::error::Result;

/// Write row-major RGBA `f32` pixels (pixel (0, 0) top-left, the crate-wide
/// convention) to `path` as a linear EXR.
///
/// # Errors
///
/// [`crate::Error::ImageWrite`] if encoding or the underlying I/O fails.
///
/// # Panics
///
/// If `pixels` doesn't hold exactly `width × height` RGBA quads — a
/// programmer bug (D-010).
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
