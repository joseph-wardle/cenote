//! Convergence test: the quantitative half of M3's validation harness.
//!
//! The goldens next door pin that the estimators don't *change*; this pins
//! that they *converge*. On the many-light scene — 256 emitters over a
//! cluster of occluders (D-085) — both the brute-force path tracer and
//! `ReSTIR` must drive their error toward zero as samples accumulate, and at
//! a matched sample budget `ReSTIR`'s resampling must get there with less
//! error. That gap is the whole reason the method exists: a single
//! next-event draw among 256 lights is mostly wasted, while `ReSTIR`
//! resamples the good candidates and borrows its neighbours'.
//!
//! Ground truth is `ReSTIR` at a high budget, not brute force. On this scene
//! `ReSTIR`'s variance is so much lower that brute force would need thousands
//! of samples to match its noise floor — too expensive for a reference the
//! other renders are measured against. The Part-2 goldens already pin that
//! the two estimators agree in the mean (brute-force `many-lights` and
//! `restir-many-lights` land within ~1e-4 per channel), so this reference
//! privileges neither estimator's *answer* — only its noise floor. The two
//! load-bearing claims are then read off brute force's *clean* numbers: it
//! converges toward the reference (so it agrees with `ReSTIR`, unbiasedness
//! quantified), and at a matched budget it carries more error than `ReSTIR`.
//!
//! Deterministic, like the goldens: accumulation replays the exact sample
//! sequence, so every relMSE below reproduces run to run — the thresholds
//! are margins on fixed numbers, not statistics on noisy ones. Skips cleanly
//! without a capable GPU, so plain `cargo test` passes everywhere.

use cenote::render::{RenderMode, Renderer};
use cenote::scene::Scene;

mod common;
use common::{accumulate, test_context};

/// Convergence-test resolution. Smaller than the goldens' 256²: the metric
/// is a per-pixel average, and this many pixels already settles it, while
/// the reference render stays quick.
const SIZE: u32 = 128;

/// Samples for the reference image — `ReSTIR` accumulated well past the test
/// budgets, so its residual noise sits far below the errors being compared.
const REFERENCE_SPP: u32 = 256;

/// The low matched budget, where a single next-event draw among 256 lights
/// is starved and `ReSTIR`'s reuse pays off most.
const LOW_SPP: u32 = 8;

/// The high matched budget — 4× the low one, so a plain Monte Carlo estimator
/// should roughly quarter its mean-squared error getting here.
const HIGH_SPP: u32 = 32;

/// Both estimators drive the many-light scene's error toward zero, and at a
/// matched budget `ReSTIR` gets there with less of it — the variance
/// reduction the method exists for, measured rather than asserted by faith.
#[test]
fn restir_converges_faster_than_brute_force() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = Scene::many_lights(&gpu).expect("many-lights scene");

    // Ground truth: ReSTIR far past the test budgets, its noise floor well
    // below the errors below.
    let mut restir = Renderer::new(&gpu).expect("renderer");
    restir.set_render_mode(RenderMode::Restir);
    let reference = accumulate(&gpu, &restir, &scene, SIZE, REFERENCE_SPP);

    // Brute force measured cleanly against that reference — a different
    // estimator, so no shared samples tilt its numbers.
    let brute = Renderer::new(&gpu).expect("renderer");
    let brute_low = rel_mse(&accumulate(&gpu, &brute, &scene, SIZE, LOW_SPP), &reference);
    let brute_high = rel_mse(&accumulate(&gpu, &brute, &scene, SIZE, HIGH_SPP), &reference);

    let restir_low = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, LOW_SPP), &reference);
    let restir_high = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, HIGH_SPP), &reference);

    eprintln!("relMSE brute:  {LOW_SPP} spp {brute_low:.5}, {HIGH_SPP} spp {brute_high:.5}");
    eprintln!("relMSE ReSTIR: {LOW_SPP} spp {restir_low:.5}, {HIGH_SPP} spp {restir_high:.5}");

    // Both estimators converge: 4× the samples, distinctly less error.
    assert!(
        brute_high < brute_low,
        "brute force didn't converge: {LOW_SPP} spp {brute_low:.5} → {HIGH_SPP} spp {brute_high:.5}"
    );
    assert!(
        restir_high < restir_low,
        "ReSTIR didn't converge: {LOW_SPP} spp {restir_low:.5} → {HIGH_SPP} spp {restir_high:.5}"
    );

    // Brute force converges *toward the ReSTIR reference* — the two agree,
    // so its error against that image is already small at the high budget.
    // This is the unbiasedness thesis read off the path tracer's clean
    // numbers rather than off the reference privileging its own answer.
    assert!(
        brute_high < 0.05,
        "brute force should be closing on the ReSTIR image by {HIGH_SPP} spp: {brute_high:.5}"
    );

    // The headline: at a matched budget ReSTIR carries clearly less error,
    // by a real margin — not the sliver a near-tie could fake. ReSTIR owns
    // only the primary hit's direct lighting here (the indirect bounces are
    // the same path tracer in both), so the win lands around 2×; a 1.5×
    // floor asserts it without pinning the exact factor a driver's
    // floating-point reordering could nudge. Read off brute force's clean
    // numbers, so the win isn't an artifact of the reference sharing samples
    // with ReSTIR.
    assert!(
        restir_low * 1.5 < brute_low,
        "ReSTIR should carry clearly less error at {LOW_SPP} spp: \
         ReSTIR {restir_low:.5} vs brute {brute_low:.5}"
    );
    assert!(
        restir_high * 1.5 < brute_high,
        "ReSTIR should carry clearly less error at {HIGH_SPP} spp: \
         ReSTIR {restir_high:.5} vs brute {brute_high:.5}"
    );
}

/// Per-channel relative MSE against `reference`: the mean over every colour
/// channel of `(estimate − reference)² / (reference² + ε)`. Relative, so it
/// weights a 10% error in shadow the same as in highlight instead of letting
/// the bright emitters dominate; the ε floor keeps a near-black reference
/// pixel from exploding the ratio, and the truly-black sky (0 in both) drops
/// out at zero.
fn rel_mse(estimate: &[f32], reference: &[f32]) -> f32 {
    const EPS: f32 = 1e-3;
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for (est, refr) in estimate.chunks_exact(4).zip(reference.chunks_exact(4)) {
        for channel in 0..3 {
            let difference = est[channel] - refr[channel];
            sum += f64::from(difference * difference / (refr[channel] * refr[channel] + EPS));
            count += 1.0;
        }
    }
    (sum / count) as f32
}
