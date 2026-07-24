//! `cenote-flip`: mean-FLIP two EXRs and pass or fail against a threshold —
//! the pixel oracle the Hydra harness shells out to. The in-process golden
//! tests (`crates/cenote/tests/golden.rs`) FLIP a render against a checked-in
//! EXR without ever leaving Rust; this is the same comparison for images that
//! arrive as files, so the end-to-end usdrecord path (USD → the delegate →
//! the server → an EXR on disk) gets the same perceptual pin. FLIP absorbs the
//! path tracer's residual Monte-Carlo noise a byte compare would trip on, and
//! the saturated primary in `golden-stage.usda` makes a dropped server-side
//! Rec.709 conversion — the one drift the wire corpus cannot see — land far
//! above the threshold.
//!
//! ```sh
//! cenote-flip golden.exr actual.exr            # exit 0 iff mean FLIP <= 0.01
//! cenote-flip golden.exr actual.exr --threshold 0.02 --heatmap flip.exr
//! ```
//!
//! On failure it prints the mean and, with `--heatmap`, writes the FLIP error
//! map through the magma LUT for `tev`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context as _, bail};
use clap::Parser;

use cenote::output::{read_exr, write_exr};

/// The house mean-FLIP failure threshold, shared with the in-process goldens
/// (`crates/cenote/tests/golden.rs`): identical images score 0, floating-point
/// reordering and settled path-tracer noise stay well under it, and any visible
/// regression — a wrong shade, a dropped colour conversion — lands far above.
const DEFAULT_THRESHOLD: f32 = 0.01;

#[derive(Parser)]
#[command(version, about = "Mean-FLIP two EXRs and pass/fail against a threshold")]
struct Args {
    /// The reference EXR — the checked-in golden.
    golden: PathBuf,

    /// The EXR to test against it — the fresh render.
    actual: PathBuf,

    /// Mean-FLIP failure threshold. Exit non-zero once the mean exceeds it.
    #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f32,

    /// Write the FLIP error map (magma LUT) here when the comparison fails.
    #[arg(long)]
    heatmap: Option<PathBuf>,
}

fn main() -> anyhow::Result<ExitCode> {
    let args = Args::parse();

    let (gw, gh, golden) = read_exr(&args.golden)
        .with_context(|| format!("reading golden {}", args.golden.display()))?;
    let (aw, ah, actual) = read_exr(&args.actual)
        .with_context(|| format!("reading actual {}", args.actual.display()))?;
    if (gw, gh) != (aw, ah) {
        bail!(
            "dimensions differ: golden {gw}×{gh}, actual {aw}×{ah} — \
             regenerate the golden if the render resolution changed"
        );
    }

    let error_map = nv_flip::flip(
        flip_image(gw, gh, &golden),
        flip_image(aw, ah, &actual),
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );
    let mean = nv_flip::FlipPool::from_image(&error_map).mean();
    if mean <= args.threshold {
        println!("FLIP OK — mean {mean:.6} <= {:.6}", args.threshold);
        return Ok(ExitCode::SUCCESS);
    }

    eprintln!(
        "FLIP FAIL — mean {mean:.6} > {:.6}\n  golden: {}\n  actual: {}",
        args.threshold,
        args.golden.display(),
        args.actual.display(),
    );
    if let Some(path) = &args.heatmap {
        write_heatmap(path, &error_map)
            .with_context(|| format!("writing heatmap {}", path.display()))?;
        eprintln!(
            "  heatmap: {} (magma LUT; black = identical, bright = different)",
            path.display()
        );
    }
    Ok(ExitCode::FAILURE)
}

/// Quantize linear RGBA `f32` to the 8-bit RGB FLIP consumes: clamp to [0, 1],
/// round, drop alpha — the same reduction the in-process goldens use, so the
/// two comparisons share a blind spot (drift confined above white passes here)
/// and a threshold. Compared as if displayed without exposure or tonemap.
fn flip_image(width: u32, height: u32, pixels: &[f32]) -> nv_flip::FlipImageRgb8 {
    let rgb: Vec<u8> = pixels
        .chunks_exact(4)
        .flat_map(|rgba| {
            rgba[..3]
                .iter()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    nv_flip::FlipImageRgb8::with_data(width, height, &rgb)
}

/// Write the FLIP error map through the magma LUT as an EXR — a diagnostic
/// image for `tev`, not colour-managed data. Black is identical, bright is
/// perceptually different.
fn write_heatmap(path: &Path, error_map: &nv_flip::FlipImageFloat) -> anyhow::Result<()> {
    let pixels: Vec<f32> = error_map
        .apply_color_lut(&nv_flip::magma_lut())
        .to_vec()
        .chunks_exact(3)
        .flat_map(|rgb| {
            [
                f32::from(rgb[0]) / 255.0,
                f32::from(rgb[1]) / 255.0,
                f32::from(rgb[2]) / 255.0,
                1.0,
            ]
        })
        .collect();
    write_exr(path, error_map.width(), error_map.height(), &pixels)?;
    Ok(())
}
