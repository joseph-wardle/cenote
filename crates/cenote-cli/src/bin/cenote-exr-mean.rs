//! `cenote-exr-mean`: per-channel mean of an EXR's unclamped linear RGB.
//! Sibling of `cenote-relmse`: that one asks "did the estimator get
//! worse", this one asks "did energy go missing" — the question a bounce
//! cap poses.
//!
//! ```sh
//! cenote-exr-mean render.exr                    # mean R G B + luminance
//! cenote-exr-mean reference.exr render.exr      # both, plus the deficit
//! ```
//!
//! With two images the last line reports `1 − mean(render)/mean(reference)`
//! per channel and for luminance — positive means the render is darker.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, bail};
use clap::Parser;

use cenote::output::read_exr;

#[derive(Parser)]
#[command(version, about = "Per-channel mean of an EXR, and the deficit against a reference")]
struct Args {
    /// The image to average — or, when two are given, the reference the
    /// second is measured against.
    first: PathBuf,

    /// The render whose deficit against the first image is wanted.
    second: Option<PathBuf>,
}

/// Rec. 709 luminance weights, matching the renderer's own `color.slang`.
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Mean R, G, B over an RGBA-interleaved image. Alpha is skipped — it is 1
/// by construction. Errors on a non-finite sample, which would otherwise
/// silently poison the mean.
fn means(pixels: &[f32]) -> anyhow::Result<[f64; 3]> {
    let mut sums = [0.0_f64; 3];
    for rgba in pixels.chunks_exact(4) {
        for (sum, &sample) in sums.iter_mut().zip(&rgba[..3]) {
            if !sample.is_finite() {
                bail!("non-finite pixel value {sample}");
            }
            *sum += f64::from(sample);
        }
    }
    let count = (pixels.len() / 4).max(1);
    #[expect(clippy::cast_precision_loss, reason = "pixel counts sit far below 2^52")]
    Ok(sums.map(|sum| sum / count as f64))
}

fn luminance(rgb: [f64; 3]) -> f64 {
    rgb.iter().zip(&LUMA).map(|(c, w)| c * w).sum()
}

fn load(path: &std::path::Path) -> anyhow::Result<(u32, u32, Vec<f32>)> {
    read_exr(path).with_context(|| format!("reading {}", path.display()))
}

fn report(label: &str, rgb: [f64; 3]) {
    println!(
        "{label}  R {:.6e}  G {:.6e}  B {:.6e}  luma {:.6e}",
        rgb[0],
        rgb[1],
        rgb[2],
        luminance(rgb)
    );
}

fn main() -> anyhow::Result<ExitCode> {
    let args = Args::parse();
    let (width, height, first) = load(&args.first)?;
    let first_means = means(&first)?;
    report("mean", first_means);

    let Some(second_path) = &args.second else {
        return Ok(ExitCode::SUCCESS);
    };
    let (second_width, second_height, second) = load(second_path)?;
    if (width, height) != (second_width, second_height) {
        bail!("size mismatch: {width}×{height} vs {second_width}×{second_height}");
    }
    let second_means = means(&second)?;
    report("mean", second_means);
    let deficit = |reference: f64, render: f64| {
        if reference > 0.0 {
            1.0 - render / reference
        } else {
            0.0
        }
    };
    let per_channel: Vec<String> = first_means
        .iter()
        .zip(&second_means)
        .map(|(&r, &c)| format!("{:+.4}%", deficit(r, c) * 100.0))
        .collect();
    println!(
        "deficit  R {}  G {}  B {}  luma {:+.4}%",
        per_channel[0],
        per_channel[1],
        per_channel[2],
        deficit(luminance(first_means), luminance(second_means)) * 100.0
    );
    Ok(ExitCode::SUCCESS)
}
