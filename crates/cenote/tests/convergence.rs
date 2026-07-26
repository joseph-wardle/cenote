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

/// The M6 step-3c "reuse is alive" gate (D-139): `ReSTIR` PT must beat plain
/// path tracing at equal spp on a glossy *GI* scene — the one claim every
/// unbiasedness gate is blind to, since an estimator whose every shift
/// silently returned J = 0 would still converge to the right image, just no
/// faster than brute force.
///
/// The scene is the step-3b glossy-primary set (wavefront.rs's twin, built
/// through the public API) with its lighting turned *indirect-only*: the
/// emitter faces the sub-floor metal panel and shows the world its dark back
/// (emission is one-sided), and the environment is black, so every camera-
/// visible surface is lit exclusively through the panel's glossy reflection.
/// That is the regime path reuse exists for — per pixel the good path
/// (surface → panel → emitter) is rare, so plain PT is starved while the
/// reservoir keeps whichever walk found it and spatial reuse spreads it —
/// and its samples are exactly the hybrid-shift shapes of steps 3a-3c (the
/// panel vertex is below the pair floor, so the reuse rides seed replay).
/// Direct-lit variants of this scene measurably do NOT clear this bar: there
/// PT's summed per-vertex estimator is already low-variance and the
/// one-survivor resampling costs more than 5-neighbour reuse recovers — the
/// many-light and hard-GI regimes are where the method pays, and this gate
/// pins the hard-GI one.
#[test]
fn restir_pt_beats_brute_force_on_glossy_indirect_gi() {
    let Some(gpu) = test_context() else {
        return;
    };
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
    let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ZERO))
        .expect("indirect glossy scene");

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
