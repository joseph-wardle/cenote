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

use cenote::environment::Environment;
use cenote::gpu::Context;
use cenote::material::Material;
use cenote::render::{RenderMode, Renderer};
use cenote::scene::{Camera, Object, Scene, ground_plane, icosphere};
use glam::{Mat4, Vec3};

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
    // by a real margin — not the sliver a near-tie could fake. Since M6 step 2
    // (D-134) the reservoir owns the *whole* path integral at the primary hit —
    // direct light plus the reconnection-reused indirect — so the win compounds
    // the resampling of both. Step 5b (indirect samples crossing the frame
    // boundary) moved the measured factor: the first frames trade a little
    // sample independence for the warm-start — the correlation the decay ramp
    // exists to anneal away (D-094) — narrowing the 8-spp margin from ~1.6×
    // to ~1.5× (32 spp: ~1.7×). A 1.3× floor — the same one the step-3c
    // indirect-GI gate below uses — asserts the real win without pinning a
    // factor that now sits on the old assert's edge. Read off brute force's
    // clean numbers, so the win isn't an artifact of the reference sharing
    // samples with ReSTIR.
    assert!(
        restir_low * 1.3 < brute_low,
        "ReSTIR should carry clearly less error at {LOW_SPP} spp: \
         ReSTIR {restir_low:.5} vs brute {brute_low:.5}"
    );
    assert!(
        restir_high * 1.3 < brute_high,
        "ReSTIR should carry clearly less error at {HIGH_SPP} spp: \
         ReSTIR {restir_high:.5} vs brute {brute_high:.5}"
    );
}

/// The indirect-glossy GI scene: the step-3b glossy-primary set (wavefront.rs's
/// twin, built through the public API) with its lighting turned
/// *indirect-only* — the emitter faces the sub-floor metal panel and shows the
/// world its dark back (emission is one-sided), and the environment is black,
/// so every camera-visible surface is lit exclusively through the panel's
/// glossy reflection. That is the regime path reuse exists for — per pixel the
/// good path (surface → panel → emitter) is rare, so plain PT is starved while
/// the reservoir keeps whichever walk found it and reuse spreads it — and its
/// samples are exactly the hybrid-shift shapes of steps 3a-3c (the panel
/// vertex is below the pair floor, so the reuse rides seed replay). Shared by
/// the step-3c gate below and the step-5 decay-handoff report, which measures
/// the same regime over frames instead of over samples.
fn indirect_glossy_scene(gpu: &Context) -> Scene {
    let objects = [
        Object {
            mesh: ground_plane(3.0),
            transform: Mat4::from_translation(Vec3::new(0.0, 2.0, -1.5))
                * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
            material: Material::glossy(Vec3::new(0.9, 0.8, 0.6), 0.0, 0.12).with_metalness(1.0),
        },
        Object {
            mesh: ground_plane(4.0),
            transform: Mat4::IDENTITY,
            material: Material::matte(Vec3::splat(0.8), 0.5),
        },
        Object {
            mesh: icosphere(2),
            transform: Mat4::from_translation(Vec3::new(0.9, 1.0, 0.6)),
            material: Material::matte(Vec3::new(0.75, 0.4, 0.35), 0.5),
        },
        // The one-sided emitter: front face toward the panel, back to the
        // world, so nothing sees it but through the panel.
        Object {
            mesh: ground_plane(1.0),
            transform: Mat4::from_translation(Vec3::new(0.0, 2.5, 2.5))
                * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            material: Material::emitter(Vec3::splat(12.0)),
        },
    ];
    let camera = Camera {
        position: Vec3::new(0.0, 2.0, 4.5),
        look_at: Vec3::new(0.0, 1.8, -1.5),
        up: Vec3::Y,
        vfov_degrees: 45.0,
        lens: None,
    };
    Scene::new(gpu, &objects, camera, &Environment::constant(Vec3::ZERO))
        .expect("indirect glossy scene")
}

/// The M6 step-3c "reuse is alive" gate (D-139): `ReSTIR` PT must beat plain
/// path tracing at equal spp on a glossy *GI* scene — the one claim every
/// unbiasedness gate is blind to, since an estimator whose every shift
/// silently returned J = 0 would still converge to the right image, just no
/// faster than brute force. The scene ([`indirect_glossy_scene`]) is that
/// regime distilled. Direct-lit variants of it measurably do NOT clear this
/// bar: there PT's summed per-vertex estimator is already low-variance and
/// the one-survivor resampling costs more than 5-neighbour reuse recovers —
/// the many-light and hard-GI regimes are where the method pays, and this
/// gate pins the hard-GI one.
#[test]
fn restir_pt_beats_brute_force_on_glossy_indirect_gi() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = indirect_glossy_scene(&gpu);

    // Ground truth: ReSTIR far past the test budgets, exactly as the
    // many-light test above (the unbiasedness gates already pin that the two
    // estimators agree in the mean, so the reference privileges neither's
    // answer — only its noise floor).
    let mut restir = Renderer::new(&gpu).expect("renderer");
    restir.set_render_mode(RenderMode::Restir);
    let reference = accumulate(&gpu, &restir, &scene, SIZE, REFERENCE_SPP);

    let brute = Renderer::new(&gpu).expect("renderer");
    let brute_low = rel_mse(&accumulate(&gpu, &brute, &scene, SIZE, LOW_SPP), &reference);
    let brute_high = rel_mse(&accumulate(&gpu, &brute, &scene, SIZE, HIGH_SPP), &reference);
    let restir_low = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, LOW_SPP), &reference);
    let restir_high = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, HIGH_SPP), &reference);

    eprintln!("indirect relMSE brute:  {LOW_SPP} spp {brute_low:.5}, {HIGH_SPP} spp {brute_high:.5}");
    eprintln!("indirect relMSE ReSTIR: {LOW_SPP} spp {restir_low:.5}, {HIGH_SPP} spp {restir_high:.5}");

    // Both converge — the sanity floor under the headline claim.
    assert!(
        restir_high < restir_low,
        "ReSTIR PT didn't converge: {LOW_SPP} spp {restir_low:.5} → {HIGH_SPP} spp {restir_high:.5}"
    );
    assert!(
        brute_high < brute_low,
        "brute force didn't converge: {LOW_SPP} spp {brute_low:.5} → {HIGH_SPP} spp {brute_high:.5}"
    );

    // The gate: reuse must actually move error, not merely not lie. The
    // measured margin sits near 2×; a 1.3× floor asserts a real win without
    // pinning the exact factor a driver's floating-point reordering could
    // nudge.
    assert!(
        restir_low * 1.3 < brute_low,
        "ReSTIR PT should beat plain PT at {LOW_SPP} spp on indirect glossy GI: \
         ReSTIR {restir_low:.5} vs brute {brute_low:.5}"
    );
    assert!(
        restir_high * 1.3 < brute_high,
        "ReSTIR PT should beat plain PT at {HIGH_SPP} spp on indirect glossy GI: \
         ReSTIR {restir_high:.5} vs brute {brute_high:.5}"
    );
}

/// The step-5 decay-handoff curve (M6 §4c): relMSE of the accumulated image
/// after 1/2/4/8/16/32 frames on the indirect-glossy scene, the shipping
/// estimator (temporal on, default decay window) against temporal-off. What
/// it shows (§4c's T5 outcomes): the held-camera *cost* of history
/// correlation early — correlated frames average slower — annealing to
/// equality as the decay ramp hands the still over to spatial-only
/// accumulation (D-094); the warm-start's payoff is per-frame quality during
/// motion, which an accumulated average cannot see. Each point accumulates
/// into a fresh film, so frames-since-reset — what the ramp actually reads —
/// equals the frame count. A report, not a gate (§4c decision 4): run
/// explicitly when a checkpoint needs numbers:
///
/// ```text
/// cargo test -p cenote --release --test convergence -- \
///     --ignored --test-threads=1 --nocapture
/// ```
#[test]
#[ignore = "a measurement report, not a gate — run explicitly for checkpoint numbers"]
fn restir_decay_handoff_report() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = indirect_glossy_scene(&gpu);
    let mut restir = Renderer::new(&gpu).expect("renderer");
    restir.set_render_mode(RenderMode::Restir);
    let reference = accumulate(&gpu, &restir, &scene, SIZE, REFERENCE_SPP);
    for frames in [1u32, 2, 4, 8, 16, 32] {
        restir.set_temporal_reuse(true);
        let on = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, frames), &reference);
        restir.set_temporal_reuse(false);
        let off = rel_mse(&accumulate(&gpu, &restir, &scene, SIZE, frames), &reference);
        eprintln!("handoff at {frames:2} frames: temporal-on {on:.5}, temporal-off {off:.5}");
    }
}

/// The step-6-0 baseline (M6 §4d decision 1), captured before the estimator
/// moves — two measurements the later rungs are read against.
///
/// **The floor curve.** Does the converged-still configuration — spatial-only,
/// fresh RNG per frame (D-085) — actually accumulate at brute force's ~1/N
/// rate, or does within-frame reuse correlation leave a floor? D-127 originally
/// credited `ReSTCV` as the fix for a convergence plateau; the step-6 deep-read
/// found it is a constant-factor fix (D-142), which leaves the floor question
/// to a measurement: accumulated relMSE at 8…128 frames for both estimators on
/// the indirect-glossy scene. Read the floor off `ReSTIR`'s own ×N column — a
/// ~1/N estimator holds it flat, a floor makes it climb without end. The
/// brute column rides along for scale, but its ratio to `ReSTIR` is not the
/// lens: brute accumulation walks each pixel's Owen-scrambled Sobol sequence
/// in order, a quasi-Monte Carlo estimator that can dip mildly *below* 1/N,
/// while resampling forfeits that low-discrepancy structure — so the ratio
/// drifts brute-ward with N even with no floor anywhere (the measured 6-0
/// shape, recorded in M6 §4d).
///
/// Each estimator is measured against a deep reference built from the *other*
/// one. Accumulation replays the exact sample sequence, so an estimate shares
/// its frames with any deeper reference from the same estimator, and the
/// overlap deflates the measured tail — `E[(mean_N − mean_R)²]` falls to
/// `v·(1/N − 1/R)`, half the true error at N = R/2 — exactly where a floor would
/// show. Cross-referencing keeps the samples independent; the residual each
/// reference adds inflates both curves' last points by a symmetric few
/// percent, which the ratio mostly cancels.
///
/// **The chroma baseline.** Per-channel relMSE on many-lights (the chromatic
/// scene — 256 randomly-hued emitters), the shipping estimator under the
/// suite's own protocol (ReSTIR-at-256 reference, matched budgets), as the
/// before-side of step 6a's vector-shading delta. Brute force at the high
/// budget rides along as the mechanism check: its per-channel spread is what
/// the scene itself induces, so the gap between the spreads is the part the
/// scalar-luminance target owns. Like the decay-handoff report: a report, not
/// a gate — run explicitly when a checkpoint needs numbers:
///
/// ```text
/// cargo test -p cenote --release --test convergence -- \
///     --ignored --test-threads=1 --nocapture
/// ```
#[test]
#[ignore = "a measurement report, not a gate — run explicitly for checkpoint numbers"]
fn restir_floor_and_chroma_report() {
    // Floor-curve references deep enough that their residual sits well under
    // the smallest error measured against them: ReSTIR's 128-frame error is
    // ~v_r/128 and the brute reference's residual ~2·v_r/4096, a ~+6%
    // inflation on the last point; the ReSTIR reference at 1024 independent
    // frames does the same for brute's side.
    const FLOOR_FRAMES: [u32; 5] = [8, 16, 32, 64, 128];
    const RESTIR_REFERENCE_FRAMES: u32 = 1024;
    const BRUTE_REFERENCE_FRAMES: u32 = 4096;

    let Some(gpu) = test_context() else {
        return;
    };

    let scene = indirect_glossy_scene(&gpu);
    let mut restir = Renderer::new(&gpu).expect("renderer");
    restir.set_render_mode(RenderMode::Restir);
    restir.set_temporal_reuse(false);
    let brute = Renderer::new(&gpu).expect("renderer");

    let restir_reference = accumulate(&gpu, &restir, &scene, SIZE, RESTIR_REFERENCE_FRAMES);
    let brute_reference = accumulate(&gpu, &brute, &scene, SIZE, BRUTE_REFERENCE_FRAMES);
    for frames in FLOOR_FRAMES {
        let restir_err = rel_mse(
            &accumulate(&gpu, &restir, &scene, SIZE, frames),
            &brute_reference,
        );
        let brute_err = rel_mse(
            &accumulate(&gpu, &brute, &scene, SIZE, frames),
            &restir_reference,
        );
        let n = frames as f32;
        eprintln!(
            "floor at {frames:3} frames: ReSTIR {restir_err:.6} (×N {:.3}), \
             brute {brute_err:.6} (×N {:.3}), ratio {:.3}",
            restir_err * n,
            brute_err * n,
            restir_err / brute_err
        );
    }

    // The chroma baseline: the shipping estimator (defaults untouched), the
    // suite's reference protocol. The reference shares the estimate's frames,
    // deflating these numbers by a protocol-constant factor — fine for the
    // 6a delta, which re-runs the identical protocol on both sides.
    let scene = Scene::many_lights(&gpu).expect("many-lights scene");
    let mut restir = Renderer::new(&gpu).expect("renderer");
    restir.set_render_mode(RenderMode::Restir);
    let reference = accumulate(&gpu, &restir, &scene, SIZE, REFERENCE_SPP);
    for spp in [LOW_SPP, HIGH_SPP] {
        let [r, g, b] = rel_mse_channels(
            &accumulate(&gpu, &restir, &scene, SIZE, spp),
            &reference,
        );
        eprintln!("chroma ReSTIR {spp:2} spp per-channel relMSE: R {r:.5}, G {g:.5}, B {b:.5}");
    }
    let brute = Renderer::new(&gpu).expect("renderer");
    let [r, g, b] = rel_mse_channels(
        &accumulate(&gpu, &brute, &scene, SIZE, HIGH_SPP),
        &reference,
    );
    eprintln!("chroma brute  {HIGH_SPP:2} spp per-channel relMSE: R {r:.5}, G {g:.5}, B {b:.5}");
}

/// Relative MSE against `reference`: the mean over every colour channel of
/// `(estimate − reference)² / (reference² + ε)`. Relative, so it weights a
/// 10% error in shadow the same as in highlight instead of letting the
/// bright emitters dominate; the ε floor keeps a near-black reference pixel
/// from exploding the ratio, and the truly-black sky (0 in both) drops out
/// at zero.
fn rel_mse(estimate: &[f32], reference: &[f32]) -> f32 {
    let [r, g, b] = rel_mse_channels(estimate, reference);
    (r + g + b) / 3.0
}

/// [`rel_mse`] split per colour channel — the chroma lens (M6 §4d decision
/// 7): a scalar-luminance resampling target selects survivors by intensity
/// and leaves their *hue* to luck, so its error should sit unevenly across
/// R/G/B where a brute-force estimator's sits evenly. No new metric — the
/// same relative squared error, just not averaged across the one axis the
/// colour-noise fix acts on.
fn rel_mse_channels(estimate: &[f32], reference: &[f32]) -> [f32; 3] {
    const EPS: f32 = 1e-3;
    let mut sums = [0.0_f64; 3];
    let mut count = 0.0_f64;
    for (est, refr) in estimate.chunks_exact(4).zip(reference.chunks_exact(4)) {
        for channel in 0..3 {
            let difference = est[channel] - refr[channel];
            sums[channel] +=
                f64::from(difference * difference / (refr[channel] * refr[channel] + EPS));
        }
        count += 1.0;
    }
    sums.map(|sum| (sum / count) as f32)
}
