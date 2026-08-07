//! Convergence test: the quantitative half of the validation harness.
//!
//! The goldens next door pin that the estimator doesn't *change*; this pins
//! that it *converges*. Error against a deep reference must fall as samples
//! accumulate, and fall at the rate a Monte Carlo average is supposed to:
//! four times the budget, roughly a quarter of the mean-squared error. A
//! bias — a dropped Jacobian, a mis-weighted MIS combination, a stale
//! throughput — shows up here as an error curve that flattens against a
//! floor instead of walking down toward zero, which no single-image golden
//! can see.
//!
//! Two scenes, for the two regimes that stress different parts of the
//! estimator: [`Scene::many_lights`], 256 emitters over a cluster of
//! occluders, where next-event estimation carries the image; and
//! [`indirect_glossy_scene`], lit only through a glossy panel's reflection,
//! where it cannot help and every path has to find the light by scattering.
//!
//! Deterministic, like the goldens: accumulation replays the exact sample
//! sequence, so every relMSE below reproduces run to run — the thresholds
//! are margins on fixed numbers, not statistics on noisy ones. Skips cleanly
//! without a capable GPU, so plain `cargo test` passes everywhere.

use cenote::environment::Environment;
use cenote::gpu::Context;
use cenote::material::Material;
use cenote::render::Renderer;
use cenote::scene::{Camera, Object, Scene, ground_plane, icosphere};
use glam::{Mat4, Vec3};

mod common;
use common::{accumulate, test_context};

/// Convergence-test resolution. Smaller than the goldens' 256²: the metric
/// is a per-pixel average, and this many pixels already settles it, while
/// the reference render stays quick.
const SIZE: u32 = 128;

/// Samples for the reference image — far past the test budgets, so its
/// residual noise sits well below the errors being compared. Accumulation
/// replays one sample sequence, so the estimates below share their frames
/// with this reference; at 64× the deepest budget the resulting deflation is
/// under 2%, which no threshold here is tight enough to notice.
const REFERENCE_SPP: u32 = 2048;

/// The low budget — few enough samples that the image is visibly noisy.
const LOW_SPP: u32 = 8;

/// The high budget — 4× the low one, so a plain Monte Carlo estimator should
/// roughly quarter its mean-squared error getting here.
const HIGH_SPP: u32 = 32;

/// The shared protocol, so the two scenes below cannot drift on what the
/// gate means: a deep reference, the estimator measured at both budgets, and
/// the two claims asserted in one place.
///
/// The 1/N band is deliberately wide on both sides. Below: accumulation
/// walks each pixel's Owen-scrambled Sobol sequence in order, a
/// quasi-Monte-Carlo estimator that can converge *faster* than 1/N, so an
/// upper bound near 4 would fail on a scene that happens to be well-behaved.
/// Above: a scene whose error is dominated by rare high-variance paths
/// (fireflies) settles more slowly than 1/N without anything being wrong.
/// What the band actually catches is a curve that has stopped moving — the
/// bias signature — which lands far below 2.
fn assert_converges(gpu: &Context, scene: &Scene, label: &str) {
    let renderer = Renderer::new(gpu).expect("renderer");
    let reference = accumulate(gpu, &renderer, scene, SIZE, REFERENCE_SPP);
    let low = rel_mse(&accumulate(gpu, &renderer, scene, SIZE, LOW_SPP), &reference);
    let high = rel_mse(&accumulate(gpu, &renderer, scene, SIZE, HIGH_SPP), &reference);

    eprintln!("{label} relMSE: {LOW_SPP} spp {low:.5}, {HIGH_SPP} spp {high:.5} ({:.2}×)", low / high);

    assert!(
        high < low,
        "{label}: error didn't fall with samples: \
         {LOW_SPP} spp {low:.5} → {HIGH_SPP} spp {high:.5}"
    );
    let ratio = low / high;
    assert!(
        (2.0..8.0).contains(&ratio),
        "{label}: 4× the samples should buy roughly 4× less error, got {ratio:.2}× \
         ({LOW_SPP} spp {low:.5} → {HIGH_SPP} spp {high:.5})"
    );
}

/// The many-light regime: a single next-event draw among 256 emitters is
/// mostly wasted, so the image is noisy at a low budget and the alias table's
/// power-proportional selection is what makes it converge at all. A
/// mis-normalized selection pdf would bias the mean and flatten the curve.
#[test]
fn the_path_tracer_converges_on_many_lights() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = Scene::many_lights(&gpu).expect("many-lights scene");
    assert_converges(&gpu, &scene, "many-lights");
}

/// The indirect-glossy scene: a glossy metal panel below the floor faces the
/// one emitter, and the emitter shows the world its dark back (emission is
/// one-sided) against a black environment — so every camera-visible surface
/// is lit *exclusively* through the panel's reflection. Next-event estimation
/// cannot reach the light at all here; the whole image is carried by BSDF
/// sampling finding the panel and then the emitter behind it, which is the
/// hardest transport the renderer is asked to converge.
fn indirect_glossy_scene(gpu: &Context) -> Scene {
    let objects = [
        Object {
            mesh: ground_plane(3.0),
            transform: Mat4::from_translation(Vec3::new(0.0, 2.0, -1.5))
                * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
            material: Material::glossy(Vec3::new(0.9, 0.8, 0.6), 0.0, 0.12).with_metalness(1.0),
            medium: None,
            interior_priority: 0,
        },
        Object {
            mesh: ground_plane(4.0),
            transform: Mat4::IDENTITY,
            material: Material::matte(Vec3::splat(0.8), 0.5),
            medium: None,
            interior_priority: 0,
        },
        Object {
            mesh: icosphere(2),
            transform: Mat4::from_translation(Vec3::new(0.9, 1.0, 0.6)),
            material: Material::matte(Vec3::new(0.75, 0.4, 0.35), 0.5),
            medium: None,
            interior_priority: 0,
        },
        // The one-sided emitter: front face toward the panel, back to the
        // world, so nothing sees it but through the panel.
        Object {
            mesh: ground_plane(1.0),
            transform: Mat4::from_translation(Vec3::new(0.0, 2.5, 2.5))
                * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            material: Material::emitter(Vec3::splat(12.0)),
            medium: None,
            interior_priority: 0,
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

#[test]
fn the_path_tracer_converges_on_glossy_indirect_gi() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = indirect_glossy_scene(&gpu);
    assert_converges(&gpu, &scene, "indirect-glossy");
}

/// Relative MSE against `reference`: the mean over every colour channel of
/// `(estimate − reference)² / (reference² + ε)`. Relative, so it weights a
/// 10% error in shadow the same as in highlight instead of letting the
/// bright emitters dominate; the ε floor keeps a near-black reference pixel
/// from exploding the ratio, and the truly-black sky (0 in both) drops out
/// at zero.
fn rel_mse(estimate: &[f32], reference: &[f32]) -> f32 {
    const EPS: f32 = 1e-3;
    let mut sum = 0.0_f64;
    let mut count = 0.0_f64;
    for (est, refr) in estimate.chunks_exact(4).zip(reference.chunks_exact(4)) {
        for channel in 0..3 {
            let difference = est[channel] - refr[channel];
            sum += f64::from(difference * difference / (refr[channel] * refr[channel] + EPS));
        }
        count += 3.0;
    }
    (sum / count) as f32
}
