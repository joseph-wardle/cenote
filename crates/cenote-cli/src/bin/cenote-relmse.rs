//! `cenote-relmse`: relative mean-squared error of a render against a
//! high-sample reference — B7's image oracle, "equal error at equal
//! samples". `cenote-flip` answers "does it look different"; this answers
//! "did the estimator get worse", the standing question now that
//! bit-exactness is no longer the cross-version gate.
//!
//! ```sh
//! cenote-relmse reference.exr render.exr                  # prints relMSE
//! cenote-relmse reference.exr render.exr --baseline 3.1e-2 --tolerance 0.03
//! ```
//!
//! With `--baseline` it pass/fails: exit 0 iff
//! relMSE <= baseline × (1 + tolerance).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, bail};
use clap::Parser;

use cenote::output::read_exr;

#[derive(Parser)]
#[command(version, about = "Relative MSE of a render against a reference EXR")]
struct Args {
    /// The high-sample reference EXR the error is measured against.
    reference: PathBuf,

    /// The render to score.
    render: PathBuf,

    /// The baseline build's relMSE at the same sample count. When given,
    /// exit non-zero once this render exceeds it by more than --tolerance.
    #[arg(long)]
    baseline: Option<f64>,

    /// Allowed fractional excess over --baseline (0.03 = 3%).
    #[arg(long, default_value_t = 0.03, requires = "baseline")]
    tolerance: f64,
}

/// The denominator's regularizer, pbrt's choice: per-channel
/// (c − r)² / (r² + 0.01). Keeps dark pixels from dominating without an
/// arbitrary luminance cutoff.
const EPSILON: f64 = 0.01;

/// Mean of (c − r)² / (r² + [`EPSILON`]) over the RGB channels of two
/// RGBA-interleaved images. Alpha is skipped — it is 1 on both sides by
/// construction. Errors on the first non-finite sample, which would
/// otherwise silently poison the mean.
fn relmse(reference: &[f32], render: &[f32]) -> anyhow::Result<f64> {
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for (r, c) in reference.chunks_exact(4).zip(render.chunks_exact(4)) {
        for channel in 0..3 {
            let r = f64::from(r[channel]);
            let c = f64::from(c[channel]);
            if !r.is_finite() || !c.is_finite() {
                bail!("non-finite pixel value (reference {r}, render {c})");
            }
            sum += (c - r).powi(2) / (r.powi(2) + EPSILON);
            count += 1;
        }
    }
    Ok(sum / count as f64)
}

fn main() -> anyhow::Result<ExitCode> {
    let args = Args::parse();
    let (rw, rh, reference) = read_exr(&args.reference)
        .with_context(|| format!("reading reference {}", args.reference.display()))?;
    let (cw, ch, render) = read_exr(&args.render)
        .with_context(|| format!("reading render {}", args.render.display()))?;
    if (rw, rh) != (cw, ch) {
        bail!("dimensions differ: reference {rw}×{rh}, render {cw}×{ch}");
    }
    let relmse = relmse(&reference, &render)?;

    match args.baseline {
        None => {
            println!("{relmse:.6e}");
            Ok(ExitCode::SUCCESS)
        }
        Some(baseline) => {
            let bar = baseline * (1.0 + args.tolerance);
            if relmse <= bar {
                println!("relMSE OK — {relmse:.6e} <= {bar:.6e} (baseline {baseline:.6e})");
                Ok(ExitCode::SUCCESS)
            } else {
                eprintln!(
                    "relMSE FAIL — {relmse:.6e} > {bar:.6e} (baseline {baseline:.6e} + {:.0}%)\n  \
                     reference: {}\n  render: {}",
                    args.tolerance * 100.0,
                    args.reference.display(),
                    args.render.display(),
                );
                Ok(ExitCode::FAILURE)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::relmse;

    #[test]
    fn identical_images_score_zero() {
        let image = [0.25, 0.5, 1.0, 1.0, 3.0, 0.0, 0.125, 1.0];
        assert_eq!(relmse(&image, &image).unwrap(), 0.0);
    }

    #[test]
    fn matches_hand_computed_value() {
        // One pixel, one differing channel: (1.5 − 1)² / (1² + 0.01),
        // averaged over the three channels.
        let reference = [1.0, 0.0, 0.0, 1.0];
        let render = [1.5, 0.0, 0.0, 1.0];
        let expected = 0.25 / 1.01 / 3.0;
        assert!((relmse(&reference, &render).unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn alpha_is_ignored() {
        let reference = [1.0, 1.0, 1.0, 1.0];
        let render = [1.0, 1.0, 1.0, 0.0];
        assert_eq!(relmse(&reference, &render).unwrap(), 0.0);
    }

    #[test]
    fn non_finite_pixels_error() {
        let reference = [f32::NAN, 0.0, 0.0, 1.0];
        let render = [0.0, 0.0, 0.0, 1.0];
        assert!(relmse(&reference, &render).is_err());
    }
}
