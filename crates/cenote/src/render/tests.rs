//! Render-orchestration tests: one-shot vs progressive equivalence, the
//! accumulate/resolve determinism invariant (bitwise-deterministic renders,
//! GPU-vs-host average agreement), and physical validation of the estimator
//! — the furnace matrix, MIS agreement, and the AOV guides.

use glam::{Mat4, Vec3};

use super::*;
use crate::environment::Environment;
use crate::material::Material;
use crate::scene::{Camera, Object, ground_plane};

fn pixel(pixels: &[f32], width: u32, x: u32, y: u32) -> &[f32] {
    let idx = ((y * width + x) * 4) as usize;
    &pixels[idx..idx + 4]
}

fn download_f32(gpu: &Context, buffer: &Buffer) -> Vec<f32> {
    bytemuck::pod_collect_to_vec(&gpu.download_buffer(buffer).expect("download"))
}

/// Accumulate `samples` waves of `scene` into a fresh `size`×`size`
/// film and return the raw per-pixel RGBA sums.
fn accumulate_sum(
    gpu: &Context,
    renderer: &Renderer,
    scene: &Scene,
    size: u32,
    samples: u32,
) -> Vec<f32> {
    let mut film = Film::new(gpu, size, size).expect("film");
    for _ in 0..samples {
        renderer
            .accumulate(gpu, scene, &mut film)
            .expect("accumulate");
    }
    download_f32(gpu, &film.beauty.sum)
}

/// A furnace scene: one big plane of the given material, scaled by
/// `scale` and centered at `center`, under a half-intensity gray sky,
/// with the camera just above looking obliquely down (the basis
/// forbids straight down) so every camera ray lands on it — through
/// `lens` when one is given (the plane dwarfs any aperture, so lens
/// rays land on it all the same). A path hits the plane, scatters
/// upward, and escapes — so for a white material the expected pixel
/// value is exactly the sky radiance (the energy-preservation
/// property the EON and compensated-GGX lobes are built around), and
/// for a pure Lambert surface every individual sample equals
/// albedo × sky.
fn furnace_scene(
    gpu: &Context,
    material: Material,
    center: Vec3,
    scale: f32,
    lens: Option<crate::scene::Lens>,
) -> Scene {
    let object = Object {
        mesh: ground_plane(5.0),
        transform: Mat4::from_translation(center) * Mat4::from_scale(Vec3::splat(scale)),
        material,
        medium: None,
        interior_priority: 0,
    };
    let camera = Camera {
        position: center + Vec3::new(0.0, scale, 0.0),
        look_at: center + Vec3::new(0.0, 0.0, -scale),
        up: Vec3::Y,
        vfov_degrees: 40.0,
        lens,
    };
    Scene::new(
        gpu,
        &[object],
        camera,
        &Environment::constant(Vec3::splat(0.5)),
    )
    .expect("furnace scene")
}

/// Accumulate `samples` waves through a BSDF-only engine and return
/// the per-pixel RGBA sums. The exactness furnace tests below use this
/// mode deliberately: single-strategy Lambert estimates are pointwise
/// exact (every sample equals albedo × sky), while next-event + MIS
/// estimates the same integral with per-sample variance — unbiased,
/// but no longer a tight per-pixel assertion. Strategy agreement is
/// the MIS-agreement tests' job, over in `wavefront.rs`.
fn bsdf_only_sum(gpu: &Context, scene: &Scene, size: u32, samples: u32) -> Vec<f32> {
    bsdf_only_sum_deep(gpu, scene, size, samples, Wavefront::DEFAULT_MAX_BOUNCES)
}

/// [`bsdf_only_sum`] at an explicit bounce cap — for the estimators whose
/// exactness the cap itself would otherwise bound. A path inside a
/// scattering medium turns several times before it leaves, and the truncated
/// tail is energy, not noise.
fn bsdf_only_sum_deep(
    gpu: &Context,
    scene: &Scene,
    size: u32,
    samples: u32,
    bounces: u32,
) -> Vec<f32> {
    bsdf_only_trace(gpu, scene, size, samples, bounces).1
}

/// The same sums under an explicit light-sampling strategy — for the
/// agreement gates, where the point is that three different derivations of
/// one image land in the same place.
fn strategy_sum(
    gpu: &Context,
    scene: &Scene,
    size: u32,
    samples: u32,
    light_sampling: LightSampling,
) -> Vec<f32> {
    traced_sum(
        gpu,
        scene,
        size,
        samples,
        Wavefront::DEFAULT_MAX_BOUNCES,
        light_sampling,
    )
    .1
}

/// The engine under [`bsdf_only_sum_deep`], handing back the wavefront that
/// traced the sums as well: the probe gates read their histogram off it,
/// and a wavefront built per call is what makes the counts this scene's
/// alone — the bins accumulate from allocation until they are read.
fn bsdf_only_trace(
    gpu: &Context,
    scene: &Scene,
    size: u32,
    samples: u32,
    bounces: u32,
) -> (Wavefront, Vec<f32>) {
    traced_sum(gpu, scene, size, samples, bounces, LightSampling::BsdfOnly)
}

fn traced_sum(
    gpu: &Context,
    scene: &Scene,
    size: u32,
    samples: u32,
    bounces: u32,
    light_sampling: LightSampling,
) -> (Wavefront, Vec<f32>) {
    let wavefront = Wavefront::new(
        gpu,
        &Kernels::embedded(),
        Wavefront::DEFAULT_CAPACITY,
        bounces,
        light_sampling,
    )
    .expect("wavefront");
    let radiance = gpu
        .create_buffer(
            "test.radiance",
            u64::from(size) * u64::from(size) * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )
        .expect("radiance buffer");
    let mut sum = vec![0.0_f32; (size * size * 4) as usize];
    for sample in 0..samples {
        wavefront
            .trace(gpu, scene, &radiance, size, size, sample)
            .expect("trace");
        for (total, value) in sum.iter_mut().zip(download_f32(gpu, &radiance)) {
            *total += value;
        }
    }
    (wavefront, sum)
}

/// Probe the demo image's invariants. Every pixel finishes exactly
/// once per wave (alpha 1, finite, non-negative), nearly the whole
/// frame is lit under the daytime HDRI — at 1 spp a pixel goes black
/// only when Russian roulette kills its path with every next-event
/// connection occluded, which is rare — and the top-left pixel is open
/// sky bright enough to be daytime. (The exact-background probe lives
/// with the constant-sky scene in `wavefront.rs`; the demo's HDRI
/// background varies per direction.)
#[test]
fn demo_image_is_sky_lit() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let (width, height) = (128, 128);
    let pixels = renderer
        .render(&gpu, &scene, width, height)
        .expect("render");

    let mut lit = 0;
    for chunk in pixels.chunks_exact(4) {
        assert_eq!(chunk[3..], [1.0], "a pixel was skipped: {chunk:?}");
        assert!(
            chunk[..3].iter().all(|c| c.is_finite() && *c >= 0.0),
            "non-finite or negative radiance: {chunk:?}"
        );
        if chunk[..3].iter().sum::<f32>() > 0.0 {
            lit += 1;
        }
    }
    assert!(
        lit > (width * height * 9 / 10) as usize,
        "most of the frame should be lit, got {lit} pixels"
    );
    assert!(
        pixel(&pixels, width, 0, 0)[..3].iter().sum::<f32>() > 0.5,
        "the top-left pixel should be open daytime sky"
    );
}

/// Dimensions that aren't a multiple of the workgroup size exercise the
/// kernel's bounds guard: partial workgroups must still write every
/// in-bounds pixel (hit or miss, alpha is 1) without tripping validation
/// on the ragged edge.
#[test]
fn ragged_dimensions_cover_every_pixel() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let pixels = renderer.render(&gpu, &scene, 33, 17).expect("render");
    for chunk in pixels.chunks_exact(4) {
        assert_eq!(chunk[3..], [1.0]);
    }
}

/// The diffuse white furnace: an albedo-1 EON plane under a uniform
/// sky must reflect exactly the sky radiance —
/// energy lost or gained anywhere in the estimator (a dropped
/// multiple-scattering lobe, a wrong pdf, a biased roulette) shifts the
/// result. At roughness 0 the lobe is Lambert and, BSDF-only, *every
/// sample of every pixel* equals the sky exactly, so the bound is
/// tight; at roughness 1 the per-sample value is stochastic (and the
/// albedo fit itself is only good to ~4e-4), so the mean over the full
/// MIS renderer carries the assertion.
#[test]
fn diffuse_furnace_closes() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let sky = 0.5;

    let lambert = furnace_scene(&gpu, Material::matte(Vec3::ONE, 0.0), Vec3::ZERO, 1.0, None);
    let sum = bsdf_only_sum(&gpu, &lambert, 32, 4);
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - sky).abs() < 1e-3,
                "Lambert furnace leaked: {value} vs {sky}"
            );
        }
    }

    let rough = furnace_scene(&gpu, Material::matte(Vec3::ONE, 1.0), Vec3::ZERO, 1.0, None);
    let samples = 64;
    let sum = accumulate_sum(&gpu, &renderer, &rough, 32, samples);
    let mean =
        sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * samples as f32);
    assert!(
        (mean - sky).abs() < 0.005,
        "rough furnace leaked: mean {mean} vs {sky}"
    );
}

/// The spawn-point offsets hold at scene scale — the property the van
/// Antwerpen rigorous error bounds exist for. A half-albedo Lambert
/// furnace, with the plane pushed 10⁴ m from the origin and scaled
/// 1000×, where hit reconstruction error reaches millimeters: every
/// sample must still be albedo × sky exactly. A bounce ray that
/// self-intersects the plane it just left multiplies in another albedo
/// factor and fails the bound loudly. (An albedo-1 furnace can't see
/// this — spurious extra bounces cost it no energy — which is why this
/// one is gray. BSDF-only, for the same per-sample exactness as the
/// Lambert furnace above.)
#[test]
fn ray_offsets_hold_at_scene_scale() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = furnace_scene(
        &gpu,
        Material::matte(Vec3::splat(0.5), 0.0),
        Vec3::new(1e4, 0.0, 1e4),
        1e3,
        None,
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    let expected = 0.5 * 0.5; // albedo × sky
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - expected).abs() < 1e-3,
                "self-intersection at scale: {value} vs {expected}"
            );
        }
    }
}

/// The furnace through a thin lens: with the aperture wide open, every
/// sample of every pixel must still equal albedo × sky exactly — a lens
/// ray is just a different ray, carrying weight 1. Any accidental
/// weighting by the lens sample (a pdf factor, a cosine, a
/// normalization slip) scales the whole image and fails loudly. The
/// blur itself is invisible here by construction — a uniform plane
/// looks the same from everywhere on the disk — which is exactly what
/// isolates the energy question from the geometry one (the viewer-side
/// blur test lives in `wavefront.rs`).
#[test]
fn the_furnace_closes_through_a_thin_lens() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = furnace_scene(
        &gpu,
        Material::matte(Vec3::splat(0.5), 0.0),
        Vec3::ZERO,
        1.0,
        Some(crate::scene::Lens {
            aperture_radius: 0.05,
            focus_distance: 1.5,
        }),
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    let expected = 0.5 * 0.5; // albedo × sky
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - expected).abs() < 1e-3,
                "the lens carried weight: {value} vs {expected}"
            );
        }
    }
}

/// A subsurface furnace scene: a closed unit icosphere of the given
/// material under a half-gray sky, the camera just above its pole looking
/// obliquely down-forward so every camera ray lands on it. A sphere
/// because the plane the other furnaces stand on is an open shell a walk
/// would leak straight out of — and convex, so an exited path can never
/// re-enter and every path carries exactly one walk.
fn subsurface_furnace_scene(gpu: &Context, material: Material) -> Scene {
    let object = Object {
        mesh: crate::scene::icosphere(4),
        transform: Mat4::IDENTITY,
        material,
        medium: None,
        interior_priority: 0,
    };
    let camera = Camera {
        position: Vec3::new(0.0, 1.2, 0.0),
        look_at: Vec3::new(0.0, 0.5, -0.35),
        up: Vec3::Y,
        vfov_degrees: 40.0,
        lens: None,
    };
    Scene::new(
        gpu,
        &[object],
        camera,
        &Environment::constant(Vec3::splat(0.5)),
    )
    .expect("subsurface furnace scene")
}

/// The subsurface material the walk furnaces share: the lobe at full
/// weight over a black diffuse base with no specular interface, so the
/// walk is the only technique and the entry carries no Fresnel.
fn walk_material(color: Vec3, radius: f32) -> Material {
    Material {
        subsurface_weight: 1.0,
        subsurface_color: color,
        subsurface_radius: radius,
        subsurface_radius_scale: Vec3::ONE,
        ..Material::matte(Vec3::ZERO, 0.0)
    }
}

/// The subsurface white furnace, at both ends of the walk's regime: at
/// radius 0.25 the sphere is ~8 mean free paths across and every path
/// scatters (α = 1, so only geometry ends a walk); at 1e9 every walk
/// crosses in one unscattered flight — the `σ_t` → 0 limit, closing the
/// same furnace through the no-event path.
///
/// A subsurface color of 1 inverts to an albedo of exactly 1, achromatic
/// extinction makes every distance draw's weight exactly 1, and entry and
/// exit lobes sample their own densities — so BSDF-only, every sample is
/// the sky *exactly*, up to one sliver: a cosine draw that crosses the
/// geometric horizon of the icosphere's smooth normals dies by the
/// standard shading-normal trade (~1e-4 of walks), zeroing its whole
/// sample. Hence the per-sample bound is one-sided — no sample may ever
/// *mint* energy — and the mean carries the two-sided bound, the death
/// rate an order of magnitude under it.
#[test]
fn subsurface_furnace_closes() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.5;
    for radius in [0.25_f32, 1e9] {
        let scene = subsurface_furnace_scene(&gpu, walk_material(Vec3::ONE, radius));
        let samples = 4;
        let sum = bsdf_only_sum(&gpu, &scene, 32, samples);
        let mut mean = 0.0;
        for chunk in sum.chunks_exact(4) {
            for channel in &chunk[..3] {
                let value = channel / samples as f32;
                assert!(
                    value < sky + 1e-3,
                    "the walk minted energy at radius {radius}: {value} vs {sky}"
                );
            }
            mean += chunk[0] / samples as f32;
        }
        mean /= 32.0 * 32.0;
        assert!(
            (mean - sky).abs() < 1e-3,
            "subsurface furnace mean drifted at radius {radius}: {mean} vs {sky}"
        );
    }
}

/// The partial-blend oracle, which the weight-0 and weight-1 gates cannot
/// reach: strictly between them the diffuse lobe and the walk share one
/// slot, and every factor that splits them has to appear exactly once.
///
/// Over an achromatic matte base of albedo `A` under a white subsurface
/// color (α = 1, so the walk conserves exactly), the two techniques carry
/// the *same* weight: the diffuse draw's `(1−w)·A/π · cos` over its
/// mixture density `wDiffuse·(cos/π)/wTotal`, and the entry's `w` over its
/// selection probability `wSubsurface/wTotal`, both land on `wTotal =
/// (1−w)·A + w`. So every sample is exactly `wTotal · sky` whichever
/// technique it drew, and any misplaced blend factor moves it: dropping
/// (1−w) from the diffuse evaluation or w from the entry mints energy, and
/// swapping the two shifts the mean (A ≠ 1 and w ≠ ½ keep the orderings
/// apart). One-sided per sample and two-sided on the mean, as the white
/// furnace above, for the same shading-normal sliver.
#[test]
fn subsurface_blends_with_the_diffuse_lobe() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.5;
    let base = 0.2;
    for weight in [0.25_f32, 0.75] {
        for radius in [0.25_f32, 1e9] {
            let scene = subsurface_furnace_scene(
                &gpu,
                Material {
                    subsurface_weight: weight,
                    subsurface_color: Vec3::ONE,
                    subsurface_radius: radius,
                    subsurface_radius_scale: Vec3::ONE,
                    ..Material::matte(Vec3::splat(base), 0.0)
                },
            );
            let samples = 4;
            let expected = sky * ((1.0 - weight) * base + weight);
            let sum = bsdf_only_sum(&gpu, &scene, 32, samples);
            let mut mean = 0.0;
            for chunk in sum.chunks_exact(4) {
                for channel in &chunk[..3] {
                    let value = channel / samples as f32;
                    assert!(
                        value < expected + 1e-3,
                        "the blend minted energy at weight {weight}, radius {radius}: \
                         {value} vs {expected}"
                    );
                }
                mean += chunk[0] / samples as f32;
            }
            mean /= 32.0 * 32.0;
            assert!(
                (mean - expected).abs() < 1e-3,
                "blended mean drifted at weight {weight}, radius {radius}: {mean} vs {expected}"
            );
        }
    }
}

/// The slab-reflectance oracle: an optically deep sphere (~100 mean free
/// paths across — walks end by absorption, never at the far side) under
/// the uniform sky must read back, per channel, the *authored*
/// multiple-scatter albedo — the van de Hulst inversion's whole promise —
/// through the full MIS renderer, so the exit connection, its
/// power-heuristic weight, and the continuation's emission weight are all
/// in the loop. The reflectance sits a consistent +2–3% relative above
/// C·sky (the fit's Lambertian-boundary assumption plus the limb's
/// thin-chord translucency); the 0.03·sky bound carries that bias with
/// margin, not just noise.
#[test]
fn subsurface_walks_to_the_authored_albedo() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let sky = 0.5;
    let color = Vec3::new(0.8, 0.5, 0.2);
    let scene = subsurface_furnace_scene(&gpu, walk_material(color, 0.02));
    let samples = 64;
    let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
    for channel in 0..3 {
        let mean = sum.chunks_exact(4).map(|chunk| chunk[channel]).sum::<f32>()
            / (32.0 * 32.0 * samples as f32);
        let expected = color[channel] * sky;
        assert!(
            (mean - expected).abs() < 0.03 * sky,
            "channel {channel} reflected {mean} for an authored {expected}"
        );
    }
}

/// The walk's histogram, split the way the `.probes.ron` sidecar splits it:
/// exit lengths, then the two deaths that never reach an exit. The upper
/// bound is load-bearing — the rejection and roulette counters are
/// *appended above* both stages' halves, so the walk's own deaths are the
/// last two bins of its half, not of the array.
#[cfg(feature = "probes")]
fn walk_probes(gpu: &Context, wavefront: &Wavefront) -> (Vec<u32>, u32, u32) {
    let bins = wavefront.probes(gpu).expect("probes");
    let walk = &bins[crate::wavefront::PROBE_VOLUME_BINS..crate::wavefront::PROBE_ENTRY_REJECT_BIN];
    let (exits, deaths) = walk.split_at(walk.len() - 2);
    (exits.to_vec(), deaths[0], deaths[1])
}

/// The walk cap's contract, measured at the top of the band it claims to
/// be exact over. `WALK_CAP` bounds the interior loop unconditionally, so
/// what a production proof has to establish is not that the bound exists
/// but that nothing reaches it: at a single-scatter albedo of 0.96 — just
/// above the densest channel of Jensen's measured skin, 0.959 — no walk
/// may be killed by it.
///
/// The sphere is the harshest geometry the claim has to survive, ~100 mean
/// free paths across, so walks end by absorption and roulette rather than
/// by stumbling on the boundary — which is what lets a walk get long at
/// all. The authored color that inverts to α = 0.96 is 0.630271: the fit's
/// C² term cancels against the radicand's, leaving the inverse *linear* in
/// C, so the constant is exact rather than searched for.
/// `scene::subsurface_inverts_the_multiple_scatter_albedo` pins it beside
/// the fit it inverts.
///
/// The α = 0.995 arm is here so the zero above is evidence and not a
/// counter that never fires — a stress regime the same geometry *does*
/// drive into the cap. Should a later change shorten deep walks enough to
/// empty it (Dwivedi guiding would), that arm failing is the improvement
/// announcing itself: re-pin the stress albedo, don't delete the arm.
#[cfg(feature = "probes")]
#[test]
fn subsurface_cap_never_fires_at_production_albedo() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    // The colors that invert to α = 0.96 and α = 0.995, both pinned beside
    // the fit in `scene::subsurface_inverts_the_multiple_scatter_albedo`.
    for (color, alpha, exact) in [(0.630_271_f32, 0.96, true), (0.849_846, 0.995, false)] {
        let scene = subsurface_furnace_scene(&gpu, walk_material(Vec3::splat(color), 0.02));
        let (size, samples) = (64, 256);
        let (wavefront, _) =
            bsdf_only_trace(&gpu, &scene, size, samples, Wavefront::DEFAULT_MAX_BOUNCES);
        let (exits, leaks, kills) = walk_probes(&gpu, &wavefront);
        let walked: u64 = exits.iter().map(|&walks| u64::from(walks)).sum();
        let paths = u64::from(size) * u64::from(size) * u64::from(samples);
        assert_eq!(leaks, 0, "a closed convex shell cannot leak");
        // The sphere covers about two thirds of the frame and is convex,
        // so one walk per covered path: a floor, not a count, but it is
        // the sample size the zero is evidence over — a gate that walked
        // nothing would pass it silently.
        assert!(
            walked > paths / 2,
            "only {walked} of {paths} paths walked at α = {alpha} — too few for the \
             count to mean anything"
        );
        assert_eq!(
            kills == 0,
            exact,
            "the walk cap fired {kills} times over {walked} walks at α = {alpha}"
        );
    }
}

/// The leak counter's own oracle. A leak is the one walk death that is a
/// property of the *asset* rather than the material, so the head's
/// zero-leak gate is only worth anything if the counter fires when a shell
/// really is open — otherwise a mesh that never leaks and a probe that
/// never counts read identically.
///
/// A plane is that open shell: the entry drives the path under a surface
/// with no far side, so every walk runs out of geometry on its first leg,
/// exits never happen, and the frame is exactly black — the leaked energy
/// is a death, not a contribution.
#[cfg(feature = "probes")]
#[test]
fn subsurface_leaks_are_counted_on_an_open_shell() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = furnace_scene(
        &gpu,
        walk_material(Vec3::splat(0.8), 0.02),
        Vec3::ZERO,
        1.0,
        None,
    );
    let (size, samples) = (32, 8);
    let (wavefront, sum) =
        bsdf_only_trace(&gpu, &scene, size, samples, Wavefront::DEFAULT_MAX_BOUNCES);
    let (exits, leaks, kills) = walk_probes(&gpu, &wavefront);
    let paths = size * size * samples;
    assert_eq!(
        (leaks, kills, exits.iter().sum::<u32>()),
        (paths, 0, 0),
        "every walk under an open shell leaks, and none of them exits"
    );
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            assert_eq!(channel.to_bits(), 0, "leaked energy reached the film");
        }
    }
}

/// A white Lambert plane under a black sky, lit by exactly one delta
/// light — built through the production path (description → prep), the
/// only route delta lights exist on. The single light means selection
/// probability 1, and a delta connection has MIS weight 1, so the
/// estimator collapses to a closed form per sample.
fn delta_light_scene(gpu: &Context, light: crate::scene::description::Light) -> Scene {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, InstancePatch, LightPatch, MaterialPatch, MeshPatch, Op,
        SettingsPatch,
    };
    use crate::scene::description::{MeshSource, SceneDescription, Texturable};

    let mut description = SceneDescription::new();
    description
        .apply(&ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                // The furnace framing: just above the plane, looking
                // obliquely down, so every camera ray lands on it.
                Op::Camera(CameraPatch {
                    position: Some([0.0, 1.0, 0.0]),
                    look_at: Some([0.0, 0.0, -1.0]),
                    ..CameraPatch::new("main")
                }),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![
                            [-5.0, 0.0, -5.0],
                            [-5.0, 0.0, 5.0],
                            [5.0, 0.0, 5.0],
                            [5.0, 0.0, -5.0],
                        ],
                        normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
                        uvs: None,
                        triangles: vec![[0, 1, 2], [0, 2, 3]],
                    }),
                    ..MeshPatch::new("plane")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([1.0; 3])),
                    specular_weight: Some(0.0),
                    ..MaterialPatch::new("lambert")
                })),
                Op::Instance(InstancePatch {
                    mesh: Some("plane".into()),
                    material: Some("lambert".into()),
                    ..InstancePatch::new("floor")
                }),
                Op::Light(LightPatch {
                    light: Some(light),
                    ..LightPatch::new("the-light")
                }),
            ],
        })
        .expect("valid scene data");
    Scene::prep(gpu, &mut description).expect("prep")
}

/// The delta-light furnace: a distant light aimed straight down at the
/// white Lambert plane delivers cosθ = 1 everywhere, so every sample
/// of every pixel is exactly (albedo/π) · E — with E = π, exactly 1.
/// Anything off in the connection — the irradiance-vs-radiance
/// convention, a stray falloff, the selection probability, a shadow
/// ray that misses open sky — shifts every pixel and fails the bound.
#[test]
fn a_distant_light_is_analytically_exact() {
    use crate::scene::description::Light;

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = delta_light_scene(
        &gpu,
        Light::Distant {
            direction: [0.0, -1.0, 0.0],
            irradiance: [std::f32::consts::PI; 3],
        },
    );
    let renderer = Renderer::new(&gpu).expect("renderer");
    let pixels = renderer.render(&gpu, &scene, 16, 16).expect("render");
    for chunk in pixels.chunks_exact(4) {
        for channel in &chunk[..3] {
            assert!(
                (channel - 1.0).abs() < 2e-3,
                "distant light off the closed form: {channel} vs 1"
            );
        }
    }
}

/// The point-light sibling: hoisted 1000 m up with intensity π · 10⁶,
/// the plane's visible patch (a couple of meters) sees r² and cosθ
/// constant to ~10⁻⁵, so the inverse-square estimate
/// (albedo/π) · I / r² lands within rounding of 1 — pinning the
/// falloff and the bounded shadow-ray distance (an occluder test
/// against the light's own position would break here first).
#[test]
fn a_point_light_is_analytically_exact() {
    use crate::scene::description::Light;

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = delta_light_scene(
        &gpu,
        Light::Point {
            position: [0.0, 1000.0, 0.0],
            intensity: [std::f32::consts::PI * 1e6; 3],
        },
    );
    let renderer = Renderer::new(&gpu).expect("renderer");
    let pixels = renderer.render(&gpu, &scene, 16, 16).expect("render");
    for chunk in pixels.chunks_exact(4) {
        for channel in &chunk[..3] {
            assert!(
                (channel - 1.0).abs() < 5e-3,
                "point light off the closed form: {channel} vs 1"
            );
        }
    }
}

/// A mirroring instance transform must not swap an emitter's face:
/// emission is one-sided off the *object-space* winding front, carried
/// through the transform. `ground_plane` is x-symmetric, so composing a
/// `diag(-1,1,1)` mirror maps the panel's point set onto itself while
/// reversing its world winding (determinant −1) — the same world-space
/// quad, re-expressed. Both expressions must light the receiver
/// identically. The failure mode this pins is nearly black, not
/// half-bright: the baked light record's world winding flips, next-event
/// estimation refuses the emitting side, and MIS's power heuristic —
/// whose hit-side weight uses the side-agnostic |cos| — keeps deferring
/// to the strategy that never fires.
#[test]
fn a_mirrored_emitter_lights_the_same_side() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    // An emissive panel 2 m up, rotated so its winding front faces the
    // receiver plane below; the camera looks down at the receiver.
    let facing_down =
        Mat4::from_translation(Vec3::Y * 2.0) * Mat4::from_rotation_x(std::f32::consts::PI);
    let mirror = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    let mean = |emitter: Mat4| {
        let objects = [
            Object {
                mesh: ground_plane(4.0),
                transform: Mat4::IDENTITY,
                material: Material::matte(Vec3::splat(0.5), 0.0),
                medium: None,
                interior_priority: 0,
            },
            Object {
                mesh: ground_plane(1.0),
                transform: emitter,
                material: Material {
                    emission: Vec3::splat(5.0),
                    ..Material::matte(Vec3::ZERO, 0.0)
                },
                medium: None,
                interior_priority: 0,
            },
        ];
        let camera = Camera {
            position: Vec3::new(0.0, 1.0, 3.0),
            look_at: Vec3::ZERO,
            up: Vec3::Y,
            vfov_degrees: 45.0,
            lens: None,
        };
        let scene = Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ZERO))
            .expect("scene");
        let sum = accumulate_sum(&gpu, &renderer, &scene, 32, 32);
        sum.chunks_exact(4)
            .map(|chunk| chunk[..3].iter().sum::<f32>())
            .sum::<f32>()
            / (32.0 * 32.0 * 32.0)
    };

    let plain = mean(facing_down);
    let mirrored = mean(facing_down * mirror);
    assert!(plain > 0.05, "the receiver must be lit at all: {plain}");
    let ratio = mirrored / plain;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "the mirrored panel must light the same side: {mirrored} vs {plain}"
    );
}

/// The white-furnace matrix over the full `OpenPBR` closure, extended
/// lobe by lobe. A white material of any construction must return
/// exactly the sky's
/// radiance — single-scatter GGX *fails this by design* (it loses up
/// to half its energy at roughness 1), so each row pins its own
/// energy machinery: the multiple-scattering compensation and its
/// baked `E`/`E_avg` tables, the analytic average Fresnel that makes IOR
/// a free axis, the tabulated layering albedos (dielectric, coat —
/// where the darkening factor must vanish against a white base — and
/// the LTC fuzz), the thin-walled interference series, and the
/// stochastic-opacity split in the intersect stage. The tolerance is
/// the tables' bake residual plus sampling noise.
#[test]
fn openpbr_furnace_matrix() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let white = Vec3::ONE;
    let configs = [
        ("metal r=0.05", Material::metal(white, 0.05)),
        ("metal r=0.5", Material::metal(white, 0.5)),
        ("metal r=1.0", Material::metal(white, 1.0)),
        ("glossy-diffuse r=0.05", Material::glossy(white, 0.0, 0.05)),
        ("glossy-diffuse r=0.5", Material::glossy(white, 0.0, 0.5)),
        (
            "glossy-diffuse r=1.0, rough base",
            Material::glossy(white, 1.0, 1.0),
        ),
        (
            "half metal",
            Material::glossy(white, 0.0, 0.5).with_metalness(0.5),
        ),
        (
            "glossy ior=2.5",
            Material::glossy(white, 0.0, 0.5).with_ior(2.5),
        ),
        (
            "glossy ior=1.1",
            Material::glossy(white, 0.0, 0.8).with_ior(1.1),
        ),
        (
            "coat over diffuse",
            Material::glossy(white, 0.0, 0.5).with_coat(1.0, 0.3),
        ),
        (
            "coat over metal",
            Material::metal(white, 0.5).with_coat(1.0, 0.1),
        ),
        (
            "fuzz over diffuse",
            Material::matte(white, 0.0).with_fuzz(1.0, 0.5),
        ),
        (
            "the full stack",
            Material::glossy(white, 0.3, 0.4)
                .with_metalness(0.3)
                .with_coat(0.7, 0.2)
                .with_fuzz(0.5, 0.7),
        ),
        ("glass plane r=0.4", Material::glass(0.4, 1.5)),
        ("thin glass r=0.4", Material::glass(0.4, 1.5).thin_walled()),
        (
            "half opacity",
            Material::matte(white, 0.0).with_opacity(0.5),
        ),
    ];
    let (sky, samples) = (0.5, 64);
    for (label, material) in configs {
        let scene = furnace_scene(&gpu, material, Vec3::ZERO, 1.0, None);
        let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
        let mean =
            sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * samples as f32);
        assert!(
            (mean - sky).abs() / sky < 0.015,
            "{label}: furnace leaked, mean {mean} vs {sky}"
        );
    }
}

/// The solid-glass furnace: a closed rough-glass sphere under the
/// uniform sky, where every path really enters an interior —
/// refraction in, possibly total internal reflection, refraction out
/// at the inverted IOR — so the whole frame must still average
/// exactly the sky. This is the 3D glass energy tables' test (both
/// branches: the η < 1 one is every exit), the interior-medium path
/// state, and the epsilon-free below-surface spawn points, at a
/// deeper bounce cap so truncation noise stays under the bound.
#[test]
fn the_glass_furnace_closes() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::with_max_bounces(&gpu, 16).expect("renderer");
    let objects = [Object {
        mesh: crate::scene::icosphere(3),
        transform: Mat4::from_translation(Vec3::Y * 2.0),
        material: Material::glass(0.2, 1.5),
        medium: None,
        interior_priority: 0,
    }];
    let camera = Camera {
        position: Vec3::new(0.0, 2.0, 4.0),
        look_at: Vec3::new(0.0, 2.0, 0.0),
        up: Vec3::Y,
        vfov_degrees: 40.0,
        lens: None,
    };
    let sky = 0.5;
    let scene = Scene::new(
        &gpu,
        &objects,
        camera,
        &Environment::constant(Vec3::splat(sky)),
    )
    .expect("scene");
    let samples = 128;
    let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
    let mean =
        sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * samples as f32);
    assert!(
        (mean - sky).abs() / sky < 0.015,
        "glass furnace leaked: mean {mean} vs {sky}"
    );
}

/// Turn a closed mesh inside out: reversed winding, flipped normals. An
/// enclosure the camera stands inside, whose emission faces in — the
/// volumetric furnace's oven.
fn inside_out(mut mesh: crate::scene::Mesh) -> crate::scene::Mesh {
    for triangle in &mut mesh.triangles {
        triangle.swap(1, 2);
    }
    for normal in &mut mesh.normals {
        *normal = -*normal;
    }
    mesh
}

/// Beer–Lambert through a *global* medium, against the closed form: a
/// bright plane three meters ahead, seen through a purely absorbing
/// atmosphere with a different extinction in every channel, must read
/// exactly `L·exp(-σ_t·d)`.
///
/// A purely absorbing medium takes `sampleMedium`'s closed-form branch —
/// no distance is drawn and no density divided — so what this pins is
/// that branch alone: the exact exponential, per channel, end to end
/// through the global-medium routing. The mixture density has its own
/// oracle in `a_transparent_channel_reaches_the_environment`, which
/// scatters.
///
/// The field of view is degrees wide so every ray's path length is the
/// center ray's to within 1e-4 — the assertion is against one distance.
#[test]
fn a_global_medium_attenuates_by_beer_lambert() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let emission = 4.0;
    let distance = 3.0;
    let sigma_t = Vec3::new(0.1, 0.3, 0.6);
    let scene = Scene::new_in_medium(
        &gpu,
        &[Object {
            // Face-on to the camera, wide enough to fill the frame.
            mesh: ground_plane(5.0),
            transform: Mat4::from_translation(Vec3::NEG_Z * distance)
                * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
            material: Material::emitter(Vec3::splat(emission)),
            medium: None,
            interior_priority: 0,
        }],
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 2.0,
            lens: None,
        },
        &Environment::constant(Vec3::ZERO),
        Some(&crate::scene::Medium {
            absorption: sigma_t,
            scattering: Vec3::ZERO,
            anisotropy: 0.0,
            volume: None,
        }),
    )
    .expect("scene");

    // Distance sampling makes each sample all-or-nothing (it reaches the
    // plane or it does not), so this is a mean, not a per-sample identity.
    let (size, samples) = (16, 2048);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    let count = f64::from(samples * size * size);
    for channel in 0..3 {
        let mean = sum
            .chunks_exact(4)
            .map(|pixel| f64::from(pixel[channel]))
            .sum::<f64>()
            / count;
        let expected = f64::from(emission) * f64::from(-sigma_t[channel] * distance).exp();
        assert!(
            (mean - expected).abs() / expected < 0.02,
            "channel {channel}: {mean} vs Beer–Lambert's {expected}"
        );
    }
}

/// The volumetric furnace: a purely scattering medium inside a uniformly
/// emissive shell must read exactly the shell's radiance, however many
/// times a path turns on the way out. Radiative equilibrium — the volume
/// counterpart of the surface furnaces above, and the one test that holds
/// distance sampling, its density, and the phase function to a single
/// number together.
///
/// The extinction is neutral on purpose: with one σ in every channel the
/// mixture density collapses to the single-channel density, so `σ_s/σ_t`
/// is exactly 1 and *every sample of every pixel* equals the emission —
/// the tight-bound trick the Lambert furnace uses. BSDF-only, so no
/// next-event estimation blurs the per-sample identity.
#[test]
fn the_volumetric_furnace_closes() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let emission = 0.5;
    let scene = Scene::new_in_medium(
        &gpu,
        &[Object {
            mesh: inside_out(crate::scene::icosphere(3)),
            transform: Mat4::from_scale(Vec3::splat(2.0)),
            material: Material::emitter(Vec3::splat(emission)),
            medium: None,
            interior_priority: 0,
        }],
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 60.0,
            lens: None,
        },
        &Environment::constant(Vec3::ZERO),
        // Half a mean free path across the shell's radius: a path turns a
        // few times before it leaves, and the bounce cap truncates nothing
        // measurable.
        Some(&crate::scene::Medium {
            absorption: Vec3::ZERO,
            scattering: Vec3::splat(0.25),
            anisotropy: 0.3,
            volume: None,
        }),
    )
    .expect("scene");

    // 64 bounces, not the default 8: the shell is half a mean free path
    // deep, so a few percent of paths still turn after eight vertices, and
    // the cap would take that energy off the top as a flat 6% deficit.
    let (size, samples) = (32, 16);
    let sum = bsdf_only_sum_deep(&gpu, &scene, size, samples, 64);
    assert_all_paths_finished(&sum, samples);
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / samples as f32;
            assert!(
                (value - emission).abs() / emission < 0.005,
                "volumetric furnace leaked: {value} vs {emission}"
            );
        }
    }
}

/// A camera standing *inside* a bounded fog attenuates from its very first
/// segment: the resolve walk finds the membership, bounce 0 seeds it, and
/// intersect owes the volume stage a visit before anything is hit — so
/// what arrives is the sky times Beer–Lambert over exactly the axial
/// meters between the camera and the fog's far face, per channel.
///
/// The closed form is what makes this discriminating where a furnace
/// cannot be: an albedo-1 fog conserves energy whether or not the seed
/// works, but an *absorbing* one darkens iff bounce 0 really knows it is
/// inside — a broken seed, or a bounce-0 routing that never asks the
/// camera's set, reads the unattenuated sky and misses by e^σd.
#[test]
fn a_camera_inside_a_bounded_fog_attenuates_from_its_first_segment() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = Vec3::splat(0.75);
    let sigma = Vec3::new(0.15, 0.35, 0.6);
    // Half-extent 3 around the origin: three axial meters of fog between
    // the camera and the far face at z = −3.
    let scene = axial_scene(
        &gpu,
        &[Object {
            mesh: crate::scene::cube(3.0),
            transform: Mat4::IDENTITY,
            material: Material::matte(Vec3::ONE, 0.0),
            medium: Some(crate::scene::Medium {
                absorption: sigma,
                scattering: Vec3::ZERO,
                anisotropy: 0.0,
                volume: None,
            }),
            interior_priority: 0,
        }],
        sky.x,
    );
    let (size, samples) = (8, 64);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    assert_all_paths_finished(&sum, samples);
    let count = f64::from(samples * size * size);
    for channel in 0..3 {
        let mean = sum
            .chunks_exact(4)
            .map(|pixel| f64::from(pixel[channel]))
            .sum::<f64>()
            / count;
        let expected = f64::from(sky[channel]) * f64::from(-sigma[channel] * 3.0).exp();
        assert!(
            (mean - expected).abs() / expected < 0.005,
            "channel {channel}: {mean} vs Beer–Lambert's {expected} \
             (the unattenuated sky is {})",
            sky[channel]
        );
    }
}

/// Synthesize the grid gates' fixture: a constant-density `.nvdb`
/// (`cenote-vdb-prep --constant`), whose background *equals* the constant —
/// so trilinear interpolation reads exactly `value` everywhere inside the
/// dilated shell, and the homogeneous limit compares with no
/// interpolation-falloff residual. `None` skips the test, the way a
/// GPU-less machine skips GPU tests: the tool is an optional build.
fn constant_grid_fixture(name: &str, value: f32) -> Option<std::path::PathBuf> {
    constant_grid_fixture_at(name, value, 32, GRID_SLAB_VOXEL)
}

/// [`constant_grid_fixture`] at an explicit resolution and voxel size — the
/// overlap gates' second grid, which must land on a lattice that shares
/// neither spacing nor phase with the first.
fn constant_grid_fixture_at(
    name: &str,
    value: f32,
    res: u32,
    voxel: f32,
) -> Option<std::path::PathBuf> {
    let tool = crate::vdb::find_prep_tool()?;
    let path =
        std::env::temp_dir().join(format!("cenote-gate-{name}-{}.nvdb", std::process::id()));
    let output = std::process::Command::new(tool)
        .args(["--constant", &res.to_string(), &value.to_string(), &voxel.to_string()])
        .arg(&path)
        .output()
        .expect("run cenote-vdb-prep");
    assert!(
        output.status.success(),
        "--constant failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(path)
}

/// The fixture grid's voxel size, and the density it is filled with.
const GRID_SLAB_VOXEL: f32 = 0.05;
const GRID_SLAB_DENSITY: f32 = 0.75;
/// Its shell in the placement's local frame: active `[0, 31]³` voxels
/// dilated one voxel each way, so `[−0.05, 1.6]` m on every axis.
const GRID_SLAB_LO: f32 = -GRID_SLAB_VOXEL;
const GRID_SLAB_EXTENT: f32 = 33.0 * GRID_SLAB_VOXEL;
/// Placed so the world box is `[−0.825, 0.825]²` across and
/// `z ∈ [−2.575, −0.925]` deep — every ray of the 2° frustum crosses its
/// full 1.65 m.
const GRID_SLAB_TRANSLATE: [f32; 3] = [
    -0.825 - GRID_SLAB_LO,
    -0.825 - GRID_SLAB_LO,
    -2.575 - GRID_SLAB_LO,
];

/// A description-path scene: the slab `ops` describe, floating between a
/// narrow-fov camera at the origin and the constant white sky, nothing
/// else.
fn grid_slab_scene(gpu: &Context, ops: Vec<crate::scene::changeset::Op>) -> Scene {
    use crate::scene::changeset::{CameraPatch, ChangeSet, EnvironmentPatch, Op, SettingsPatch};

    let mut all = vec![
        Op::Settings(SettingsPatch::new("settings")),
        Op::Camera(CameraPatch {
            position: Some([0.0; 3]),
            look_at: Some([0.0, 0.0, -1.0]),
            vfov_degrees: Some(2.0),
            ..CameraPatch::new("camera")
        }),
        // Pathless: the constant white sky, the gate's light source.
        Op::Environment(EnvironmentPatch::new("sky")),
    ];
    all.extend(ops);
    let mut description = crate::scene::description::SceneDescription::new();
    description
        .apply(&ChangeSet { ops: all })
        .expect("the slab set is valid");
    Scene::prep(gpu, &mut description).expect("prep the slab scene")
}

/// The slab, as `mesh` bounding `medium` — the two arms of the homogeneous
/// limit differ in exactly these two and nothing else.
fn grid_slab_ops(
    mesh: crate::scene::description::MeshSource,
    medium: crate::scene::changeset::MediumPatch,
) -> Vec<crate::scene::changeset::Op> {
    grid_slab_ops_at(mesh, medium, GRID_SLAB_TRANSLATE, [0.0; 3])
}

/// [`grid_slab_ops`] placed where the caller says — the camera-inside gate
/// wants the same slab standing around the origin instead of in front of it,
/// and turned to face the other way.
fn grid_slab_ops_at(
    mesh: crate::scene::description::MeshSource,
    medium: crate::scene::changeset::MediumPatch,
    translate: [f32; 3],
    rotate_degrees: [f32; 3],
) -> Vec<crate::scene::changeset::Op> {
    use crate::scene::changeset::{InstancePatch, MaterialPatch, MeshPatch, Op};
    use crate::scene::description::Transform;
    vec![
        Op::Mesh(MeshPatch {
            source: Some(mesh),
            ..MeshPatch::new("shell")
        }),
        Op::Material(Box::new(MaterialPatch::new("inert"))),
        Op::Medium(medium),
        Op::Instance(InstancePatch {
            mesh: Some("shell".into()),
            material: Some("inert".into()),
            medium: Some(Some("cloud".into())),
            transforms: Some(vec![Transform::Trs {
                translate,
                rotate_degrees,
                scale: [1.0; 3],
            }]),
            ..InstancePatch::new("slab")
        }),
    ]
}

/// The fixture grid's shell as an ordinary authored mesh, in the same local
/// frame — an independent spelling of the box `MeshSource::MediumBounds`
/// generates, so the twin below compares transports and not shells.
fn grid_slab_box() -> crate::scene::description::MeshSource {
    inline_box([GRID_SLAB_LO; 3], GRID_SLAB_EXTENT)
}

/// A ground plane at y = 0, `half` metres either side of the origin.
fn ground_quad(half: f32) -> crate::scene::description::MeshSource {
    crate::scene::description::MeshSource::Inline {
        positions: vec![
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
        ],
        normals: None,
        uvs: None,
        triangles: vec![[0, 2, 1], [0, 3, 2]],
    }
}

/// An axis-aligned box from `lo`, `size` on every axis, wound outward.
fn inline_box(lo: [f32; 3], size: f32) -> crate::scene::description::MeshSource {
    crate::scene::description::MeshSource::Inline {
        positions: (0..8)
            .map(|corner: u32| {
                [0, 1, 2].map(|axis| lo[axis] + ((corner >> axis) & 1) as f32 * size)
            })
            .collect(),
        normals: None,
        uvs: None,
        triangles: vec![
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ],
    }
}

/// The grid medium: the fixture at `absorption`/`scattering`, which the
/// grid's density multiplies point by point.
fn grid_slab_medium(
    nvdb: &std::path::Path,
    absorption: [f32; 3],
    scattering: [f32; 3],
    anisotropy: f32,
) -> crate::scene::changeset::MediumPatch {
    use crate::scene::changeset::MediumPatch;
    use crate::scene::description::VolumeSource;
    MediumPatch {
        absorption: Some(absorption),
        scattering: Some(scattering),
        anisotropy: Some(anisotropy),
        volume: Some(Some(VolumeSource {
            path: nvdb.to_owned(),
            grid: "density".to_owned(),
            temperature_grid: None,
            temperature_scale: 1.0,
            temperature_offset: 0.0,
            emission: [1.0; 3],
        })),
        ..MediumPatch::new("cloud")
    }
}

/// Per-channel mean of an accumulated sum.
fn channel_means(sum: &[f32], size: u32, samples: u32) -> [f64; 3] {
    let count = f64::from(samples * size * size);
    std::array::from_fn(|channel| {
        sum.chunks_exact(4)
            .map(|pixel| f64::from(pixel[channel]))
            .sum::<f64>()
            / count
    })
}

/// Gate 2: the tracker against the closed form it must agree with. A purely
/// absorbing constant grid is ratio tracking exactly, and its mean
/// transmittance over the slab must be Beer–Lambert's `exp(−density · σ ·
/// L)` — σ gray, so the working-space conversion (white-point preserving)
/// leaves the authored value intact.
#[test]
fn a_constant_grid_slab_is_beer_lambert_exact() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = constant_grid_fixture("absorb", GRID_SLAB_DENSITY) else {
        return;
    };
    let sigma = 0.4;
    let scene = grid_slab_scene(
        &gpu,
        grid_slab_ops(
            crate::scene::description::MeshSource::MediumBounds,
            grid_slab_medium(&nvdb, [sigma; 3], [0.0; 3], 0.0),
        ),
    );
    let (size, samples) = (16, 2048);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    let _ = std::fs::remove_file(&nvdb);
    assert_all_paths_finished(&sum, samples);
    let means = channel_means(&sum, size, samples);
    let expected = f64::from(-GRID_SLAB_DENSITY * sigma * GRID_SLAB_EXTENT).exp();
    for (channel, mean) in means.iter().enumerate() {
        assert!(
            (mean - expected).abs() / expected < 0.01,
            "channel {channel}: {mean} vs Beer–Lambert's {expected}"
        );
    }
}

/// The ramp fixture's profile: `base + amplitude · sin(π(k + ½)/32)` at
/// integer index `k` along z, and `base` — the tree's background — outside.
const GRID_RAMP_BASE: f32 = 0.25;
const GRID_RAMP_AMPLITUDE: f32 = 1.0;

/// That profile at integer index `k`, and `base` — the tree's background —
/// outside the ramped block. Every ramp oracle below integrates the
/// piecewise-linear function through these samples.
fn ramp_profile(k: i32) -> f64 {
    if (0..32).contains(&k) {
        f64::from(GRID_RAMP_BASE)
            + f64::from(GRID_RAMP_AMPLITUDE)
                * (std::f64::consts::PI * (f64::from(k) + 0.5) / 32.0).sin()
    } else {
        f64::from(GRID_RAMP_BASE)
    }
}

/// The same 32³ shell as [`constant_grid_fixture`], carrying that profile.
fn ramp_grid_fixture(name: &str) -> Option<std::path::PathBuf> {
    let tool = crate::vdb::find_prep_tool()?;
    let path =
        std::env::temp_dir().join(format!("cenote-gate-{name}-{}.nvdb", std::process::id()));
    let output = std::process::Command::new(tool)
        .args([
            "--ramp",
            "32",
            &GRID_RAMP_BASE.to_string(),
            &GRID_RAMP_AMPLITUDE.to_string(),
            &GRID_SLAB_VOXEL.to_string(),
        ])
        .arg(&path)
        .output()
        .expect("run cenote-vdb-prep");
    assert!(
        output.status.success(),
        "--ramp failed (rebuild vdb-prep): {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(path)
}

/// Gate 3: the tracker where the majorant genuinely varies, which no
/// constant grid can reach. Purely absorbing, so this is still ratio
/// tracking and its mean transmittance must be `exp(−σ ∫ d dl)`; along the
/// view axis the interpolated field is the piecewise-linear function through
/// the fixture's samples, so that integral is the trapezoid sum below.
///
/// Nothing here mirrors the lattice — the expectation is a property of the
/// *field*. What it falsifies is a ceiling that underestimates the density
/// under it, whatever the cause (dilation, the two spellings of the lattice
/// transform disagreeing, a walk one cell off): the shader's density bound
/// then clamps, and the slab comes out too bright. The rays run down −z, so
/// the backward branch of the walk's opening is the one under test.
#[test]
fn a_ramped_grid_slab_integrates_to_its_trapezoid_sum() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = ramp_grid_fixture("ramp") else {
        return;
    };
    let sigma = 0.4;
    let scene = grid_slab_scene(
        &gpu,
        grid_slab_ops(
            crate::scene::description::MeshSource::MediumBounds,
            grid_slab_medium(&nvdb, [sigma; 3], [0.0; 3], 0.0),
        ),
    );
    let (size, samples) = (16, 2048);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    let _ = std::fs::remove_file(&nvdb);
    assert_all_paths_finished(&sum, samples);

    // The shell spans index z ∈ [−1, 32]: 33 unit segments, the outer two
    // running down to the background the tree reads beyond the ramp.
    let trapezoid: f64 = (-1..32)
        .map(|k| f64::midpoint(ramp_profile(k), ramp_profile(k + 1)))
        .sum();
    let expected =
        (-f64::from(sigma) * f64::from(GRID_SLAB_VOXEL) * trapezoid).exp();
    for (channel, mean) in channel_means(&sum, size, samples).iter().enumerate() {
        // Tight on purpose: the measured residual is 0.08 % (path
        // lengthening across the 2° frustum, plus the estimator's noise at
        // 2048 samples), while a lattice one cell out of step shows up as
        // 1.7 %. Determinism is bitwise, so this does not flake.
        assert!(
            (mean - expected).abs() / expected < 0.005,
            "channel {channel}: {mean} vs the profile's {expected}"
        );
    }
}

/// The emission fixture's constants: the same 32³ slab, carrying a second
/// grid beside its density that holds this value everywhere. It is *not*
/// the temperature — the scale and offset below map it to 2500 K, so both
/// knobs are load-bearing and neither can be dropped unnoticed.
const GRID_FIRE_FIELD: f32 = 1000.0;
const GRID_FIRE_SCALE: f32 = 1.5;
const GRID_FIRE_OFFSET: f32 = 1000.0;
const GRID_FIRE_KELVIN: f64 =
    GRID_FIRE_FIELD as f64 * GRID_FIRE_SCALE as f64 + GRID_FIRE_OFFSET as f64;

/// Gate 11: what an emissive medium radiates, against the integral it is
/// estimating. The slab absorbs and scatters at gray coefficients, emits at
/// a constant 2500 K, and stands in front of a black sky with one bounce
/// allowed — so nothing reaches the film but the source term, and its
/// integral along a ray closes:
///
///     ∫₀ᴸ e^(−σ_t t) σ_a L_e dt  =  L_e · (σ_a/σ_t) · (1 − e^(−σ_t L)).
///
/// The bounce cap is what makes it closed: a path that scatters would
/// otherwise turn back through the same fire and collect more of it. The
/// emission collected *before* that scatter still belongs here, and the
/// null-collision weights are what make its expectation the whole integral
/// rather than the part in front of wherever the walk stopped — which is
/// the estimator claim this gate exists to falsify.
///
/// Every factor is separately visible: `σ_a/σ_t` is under one if the
/// scattering share is dropped, the exponential moves if the interval is
/// clipped, and `L_e` is *colored* — the temperature reaches the film only
/// through the blackbody table, so a channel swap, a wrong Kelvin mapping,
/// or an emission read off the density grid rather than the temperature one
/// all land somewhere else entirely.
///
/// Run at two albedos, which are two transports rather than two numbers.
/// The scattering pair is what separates absorption from extinction: at
/// that share an estimator coupled to `σ_t` reads three times high. The purely
/// absorbing one degenerates the tracker to ratio tracking — `ps` is zero,
/// no collision ever takes the scatter arm — so it is the case where
/// nothing but the null weights carries the emission out.
#[test]
fn an_emissive_grid_slab_radiates_its_source_integral() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(tool) = crate::vdb::find_prep_tool() else {
        return;
    };
    let nvdb = std::env::temp_dir().join(format!("cenote-gate-fire-{}.nvdb", std::process::id()));
    let output = std::process::Command::new(tool)
        .args([
            "--fire",
            "32",
            &GRID_SLAB_DENSITY.to_string(),
            &GRID_FIRE_FIELD.to_string(),
            &GRID_SLAB_VOXEL.to_string(),
        ])
        .arg(&nvdb)
        .output()
        .expect("run cenote-vdb-prep");
    assert!(
        output.status.success(),
        "--fire failed (rebuild vdb-prep): {}",
        String::from_utf8_lossy(&output.stderr)
    );
    for (absorption, scattering) in [(0.25_f32, 0.55_f32), (0.8_f32, 0.0_f32)] {
        emissive_slab_matches_its_integral(&gpu, &nvdb, absorption, scattering);
    }
    let _ = std::fs::remove_file(&nvdb);
}

fn emissive_slab_matches_its_integral(
    gpu: &crate::gpu::Context,
    nvdb: &std::path::Path,
    absorption: f32,
    scattering: f32,
) {
    use crate::scene::changeset::{EnvironmentPatch, MediumPatch, Op};
    use crate::scene::description::{MeshSource, VolumeSource};
    let mut ops = grid_slab_ops(
        MeshSource::MediumBounds,
        MediumPatch {
            absorption: Some([absorption; 3]),
            scattering: Some([scattering; 3]),
            volume: Some(Some(VolumeSource {
                path: nvdb.to_owned(),
                grid: "density".to_owned(),
                temperature_grid: Some("temperature".to_owned()),
                temperature_scale: GRID_FIRE_SCALE,
                temperature_offset: GRID_FIRE_OFFSET,
                emission: [1.0; 3],
            })),
            ..MediumPatch::new("cloud")
        },
    );
    // A black sky, so the film carries the source term and nothing else.
    ops.push(Op::Environment(EnvironmentPatch {
        tint: Some([0.0; 3]),
        ..EnvironmentPatch::new("sky")
    }));
    let scene = grid_slab_scene(gpu, ops);
    let (size, samples) = (16, 4096);
    let sum = bsdf_only_sum_deep(gpu, &scene, size, samples, 1);
    assert_all_paths_finished(&sum, samples);

    let sigma_t = f64::from(GRID_SLAB_DENSITY * (absorption + scattering));
    let share = f64::from(absorption / (absorption + scattering));
    let depth = 1.0 - (-sigma_t * f64::from(GRID_SLAB_EXTENT)).exp();
    let emitted = crate::blackbody::radiance(GRID_FIRE_KELVIN);
    for (channel, mean) in channel_means(&sum, size, samples).iter().enumerate() {
        let expected = f64::from(emitted[channel]) * share * depth;
        assert!(
            (mean - expected).abs() / expected < 0.02,
            "σ_a {absorption} σ_s {scattering}, channel {channel}: \
             {mean} vs the source integral's {expected}"
        );
    }
}

/// Gate 1: the homogeneous limit. A constant grid (density d, coefficients
/// σ) and a homogeneous volume of the same box at d·σ are the same medium;
/// the tracker and the closed form must converge to the same image.
/// Scattering on, so the whole tracker runs — collisions, spectral weights,
/// the phase mix, and the grid vertices' NEE skip (unbiased: BSDF-only
/// rendering, so both scenes sample lights the same way). The comparison is
/// statistical — two estimators, one mean.
#[test]
fn a_constant_grid_matches_its_homogeneous_twin() {
    use crate::scene::changeset::MediumPatch;
    use crate::scene::description::MeshSource;
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = constant_grid_fixture("twin", GRID_SLAB_DENSITY) else {
        return;
    };
    let absorption = [0.05, 0.1, 0.2];
    let scattering = [0.3, 0.5, 0.7];
    let grid = grid_slab_scene(
        &gpu,
        grid_slab_ops(
            MeshSource::MediumBounds,
            grid_slab_medium(&nvdb, absorption, scattering, 0.4),
        ),
    );
    // The twin: the same world box as an authored mesh, the same medium at
    // density-scaled coefficients, no grid — B8's engine verbatim.
    let scaled = |sigma: [f32; 3]| sigma.map(|s| GRID_SLAB_DENSITY * s);
    let twin = grid_slab_scene(
        &gpu,
        grid_slab_ops(
            grid_slab_box(),
            MediumPatch {
                absorption: Some(scaled(absorption)),
                scattering: Some(scaled(scattering)),
                anisotropy: Some(0.4),
                ..MediumPatch::new("cloud")
            },
        ),
    );
    let (size, samples, bounces) = (16, 2048, 32);
    let grid_sum = bsdf_only_sum_deep(&gpu, &grid, size, samples, bounces);
    let twin_sum = bsdf_only_sum_deep(&gpu, &twin, size, samples, bounces);
    let _ = std::fs::remove_file(&nvdb);
    assert_all_paths_finished(&grid_sum, samples);
    assert_all_paths_finished(&twin_sum, samples);
    let grid_means = channel_means(&grid_sum, size, samples);
    let twin_means = channel_means(&twin_sum, size, samples);
    for channel in 0..3 {
        assert!(
            (grid_means[channel] - twin_means[channel]).abs() / twin_means[channel] < 0.02,
            "channel {channel}: tracker mean {} vs closed-form twin {}",
            grid_means[channel],
            twin_means[channel]
        );
    }
}

/// The overlap gates' second grid: a *different* resolution and a
/// *different* voxel size from the fixture slab's, placed so its lattice
/// shares neither spacing nor phase with it — its world origin sits
/// 0.275 m up the slab's z axis, five and a half of the slab's voxels, so no
/// cell face of one falls on a cell face of the other.
///
/// It stands strictly inside the slab along the view axis (world
/// z ∈ [−2.30, −0.98] against the slab's [−2.575, −0.925]) and spans the
/// whole 2° frustum across it. Every camera ray therefore crosses slab
/// alone, then both, then slab alone: three segments, and the middle one is
/// the case under test.
const OVERLAP_RES: u32 = 21;
const OVERLAP_VOXEL: f32 = 0.06;
const OVERLAP_DENSITY: f32 = 0.5;
/// Its shell, `[−voxel, res·voxel]` dilated the same one voxel each way.
const OVERLAP_EXTENT: f32 = 22.0 * OVERLAP_VOXEL;
const OVERLAP_TRANSLATE: [f32; 3] = [
    -0.66 + OVERLAP_VOXEL,
    -0.66 + OVERLAP_VOXEL,
    -2.30 + OVERLAP_VOXEL,
];

/// That grid as a medium named apart from the slab's, purely absorbing —
/// the overlap arms are transmittance gates, and absorption alone keeps
/// them closed forms rather than statistics.
fn overlap_medium(nvdb: &std::path::Path, absorption: [f32; 3]) -> crate::scene::changeset::MediumPatch {
    crate::scene::changeset::MediumPatch {
        name: "inner".into(),
        ..grid_slab_medium(nvdb, absorption, [0.0; 3], 0.0)
    }
}

/// Its placement, reusing the slab set's `inert` boundary material.
fn overlap_ops(medium: crate::scene::changeset::MediumPatch) -> Vec<crate::scene::changeset::Op> {
    use crate::scene::changeset::{InstancePatch, MeshPatch, Op};
    use crate::scene::description::{MeshSource, Transform};
    vec![
        Op::Mesh(MeshPatch {
            source: Some(MeshSource::MediumBounds),
            ..MeshPatch::new("inner-shell")
        }),
        Op::Medium(medium),
        Op::Instance(InstancePatch {
            mesh: Some("inner-shell".into()),
            material: Some("inert".into()),
            medium: Some(Some("inner".into())),
            transforms: Some(vec![Transform::Trs {
                translate: OVERLAP_TRANSLATE,
                rotate_degrees: [0.0; 3],
                scale: [1.0; 3],
            }]),
            ..InstancePatch::new("inner-slab")
        }),
    ]
}

/// Assert a slab scene's mean transmittance against an optical depth,
/// per channel. The sky is white and the media purely absorbing, so what
/// arrives is `exp(−τ)` and nothing else.
fn assert_transmittance(gpu: &Context, scene: &Scene, tau: f64, tolerance: f64) {
    let (size, samples) = (16, 2048);
    let sum = bsdf_only_sum(gpu, scene, size, samples);
    assert_all_paths_finished(&sum, samples);
    let expected = (-tau).exp();
    for (channel, mean) in channel_means(&sum, size, samples).iter().enumerate() {
        assert!(
            (mean - expected).abs() / expected < tolerance,
            "channel {channel}: {mean} vs the closed form's {expected}"
        );
    }
}

/// Gate 12, first arm: two heterogeneous grids over one region compose by
/// *adding* their optical depths, across lattices that agree about nothing.
///
/// Extinctions add, so the transmittance of the overlap is
/// `exp(−(σ_A d_A + σ_B d_B) L)` — the product of what each grid would do
/// alone. That is the whole claim, and it is what pins the majorant policy:
/// a bound for the pair is `m_A + m_B`, never `max(m_A, m_B)`. Two
/// different voxel sizes at two different phases (the inner grid's origin
/// stands five and a half of the outer's voxels up its z axis) mean no
/// shared cell exists to take a max over in the first place — the arm is
/// built so that any design assuming one lattice for both grids cannot even
/// be expressed against it.
///
/// Verified by mutation to see a grid dropped from the majorant: 22.0 %
/// against a 1 % bound and a clean margin of 0.12 %. What it does *not*
/// separate is `max` from `+` as such — with one leading grid and one
/// follower and no homogeneous medium beside them, `max(0, x)` and `0 + x`
/// are the same arithmetic. Its teeth are against a bound that is too
/// small, whatever made it so.
///
/// It is not a gate on speed. Today this scene is tracked with one grid on
/// the spatial lattice and the other contributing its whole-grid maximum
/// over the entire segment, which is sound and slow; 11-f replaces that
/// with a flight per grid and must land on this same number.
#[test]
fn overlapping_grids_add_their_optical_depths() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(outer) = constant_grid_fixture("overlap-outer", GRID_SLAB_DENSITY) else {
        return;
    };
    let Some(inner) =
        constant_grid_fixture_at("overlap-inner", OVERLAP_DENSITY, OVERLAP_RES, OVERLAP_VOXEL)
    else {
        return;
    };
    let (sigma_outer, sigma_inner) = (0.4, 0.3);
    let mut ops = grid_slab_ops(
        crate::scene::description::MeshSource::MediumBounds,
        grid_slab_medium(&outer, [sigma_outer; 3], [0.0; 3], 0.0),
    );
    ops.extend(overlap_ops(overlap_medium(&inner, [sigma_inner; 3])));
    let scene = grid_slab_scene(&gpu, ops);
    let tau = f64::from(GRID_SLAB_DENSITY * sigma_outer * GRID_SLAB_EXTENT)
        + f64::from(OVERLAP_DENSITY * sigma_inner * OVERLAP_EXTENT);
    assert_transmittance(&gpu, &scene, tau, 0.01);
    let _ = std::fs::remove_file(&outer);
    let _ = std::fs::remove_file(&inner);
}

/// Gate 12, second arm: the same sum where one side's majorant genuinely
/// varies along the ray and the other's does not.
///
/// The constant arm above can be passed by an estimator that never walks a
/// lattice at all — two whole-grid maxima are exact bounds when the density
/// is constant, so nothing there separates a spatial majorant from a global
/// one. Here the ramp's ceiling changes cell by cell while the inner grid's
/// does not, so the pair's bound changes at the *union* of two unrelated
/// sets of cell faces. What it falsifies beyond the arm above is a walk that
/// carries one grid's cell structure onto the other: the ramp's profile is
/// asymmetric about the slab's midpoint, so borrowing the wrong lattice does
/// not cancel.
///
/// The oracle is the trapezoid sum
/// [`a_ramped_grid_slab_integrates_to_its_trapezoid_sum`] derives, plus the
/// inner grid's `σ d L` — one term each, because optical depths add.
///
/// Verified by the same mutation, a grid dropped from the majorant: 10.7 %
/// against a 1 % bound and a clean margin of 0.23 %.
#[test]
fn a_ramp_overlapping_a_constant_grid_integrates_to_their_sum() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(ramp) = ramp_grid_fixture("overlap-ramp") else {
        return;
    };
    let Some(inner) =
        constant_grid_fixture_at("overlap-flat", OVERLAP_DENSITY, OVERLAP_RES, OVERLAP_VOXEL)
    else {
        return;
    };
    let (sigma_ramp, sigma_inner) = (0.4, 0.3);
    let mut ops = grid_slab_ops(
        crate::scene::description::MeshSource::MediumBounds,
        grid_slab_medium(&ramp, [sigma_ramp; 3], [0.0; 3], 0.0),
    );
    ops.extend(overlap_ops(overlap_medium(&inner, [sigma_inner; 3])));
    let scene = grid_slab_scene(&gpu, ops);
    let trapezoid: f64 = (-1..32)
        .map(|k| f64::midpoint(ramp_profile(k), ramp_profile(k + 1)))
        .sum();
    let tau = f64::from(sigma_ramp) * f64::from(GRID_SLAB_VOXEL) * trapezoid
        + f64::from(OVERLAP_DENSITY * sigma_inner * OVERLAP_EXTENT);
    assert_transmittance(&gpu, &scene, tau, 0.01);
    let _ = std::fs::remove_file(&ramp);
    let _ = std::fs::remove_file(&inner);
}

/// The camera-inside slab's placement: the ramp fixture standing *around*
/// the origin rather than in front of it, and **turned to face the camera**
/// — a half turn about y, so the grid's +z runs down the view direction and
/// `world_to_index` is not the identity rotation any other grid gate uses.
///
/// Local coordinates are index coordinates scaled by the voxel size, so the
/// camera lands at index (15.5, 15.5, 3.3): mid-voxel and mid-majorant-cell
/// on every axis, which is the whole point. It looks up the ramp toward its
/// peak, and every ray of the 2° frustum leaves through the far face at
/// index 32, having crossed 28.7 voxels of it.
///
/// Where along z is chosen, not arbitrary. The majorant lattice over this
/// fixture is five cells spanning the 33-voxel shell — 6.6 voxels each — and
/// their ceilings run 0.92, 1.25, 1.25, 1.25, 0.92, so only the two ends
/// step at all. The camera stands in the first cell, one step below the
/// rise: an opening that carries that cell's *lower* ceiling into the next
/// one clamps the density it then meets, and the slab comes out bright.
/// Standing anywhere in the middle three, the same mistake swaps one ceiling
/// for an identical one and is invisible — measured, not assumed.
const CAMERA_INSIDE_TRANSLATE: [f32; 3] = [
    0.825 + GRID_SLAB_LO,
    -0.825 - GRID_SLAB_LO,
    3.3 * GRID_SLAB_VOXEL,
];
const CAMERA_INSIDE_TURN: [f32; 3] = [0.0, 180.0, 0.0];
/// The camera's index along z: local `0.165` over a `0.05` voxel.
const CAMERA_INSIDE_AT: f64 = 3.3;

/// Gate 13: a camera standing inside a *grid*, not a homogeneous fog.
///
/// [`a_camera_inside_a_bounded_fog_attenuates_from_its_first_segment`]
/// already gates the seed — `resolve_camera` finds the membership and
/// bounce 0 adopts it — and nothing here re-tests that. What is new is
/// where the tracker starts. Every other grid gate opens its walk on the
/// shell it entered through, which is a lattice boundary exactly; here the
/// segment begins at an arbitrary interior point, and the DDA has to name
/// the cell containing a position it did not step to and then step its
/// boundary times from *there*. A walk that assumes its first crossing is a
/// whole cell away — true on entry, false from inside — carries the opening
/// cell's ceiling too far.
///
/// The grid is the ramp, not the constant slab, and that is load-bearing: a
/// constant grid gives every cell the same ceiling, so no lattice mistake it
/// could make is visible in the image at all.
///
/// The oracle is the trapezoid sum of
/// [`a_ramped_grid_slab_integrates_to_its_trapezoid_sum`] truncated where
/// the camera stands: the part of the opening voxel in front of it, then
/// whole unit segments from index 7 up to the far face at 32.
///
/// Verified by mutation to see an opening that steps a whole cell before its
/// first crossing (`tNext = ±1/perT`, which is what the arithmetic reduces
/// to when the walk begins on a boundary): 2.8 % against a 1 % bound, on a
/// clean margin of 0.11 %.
///
/// Two things about that number are worth keeping, because both were
/// measured rather than reasoned and both cost time to find. The mutation is
/// **invisible** from three of the five majorant cells, which share a
/// ceiling — placement is what makes this gate a gate. And it is invisible
/// at the σ = 0.4 the other slab gates use, where it moves the image 0.5 %:
/// the clamp only bites over the few voxels where the next cell's density
/// actually exceeds this one's ceiling, so the σ here is raised to 2.0 to
/// put that band above the noise. A ceiling error is not a large effect on
/// a smooth field, which is worth knowing before trusting any gate to catch
/// one.
#[test]
fn a_camera_inside_a_grid_tracks_from_where_it_stands() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = ramp_grid_fixture("camera-inside") else {
        return;
    };
    let sigma = 2.0;
    let scene = grid_slab_scene(
        &gpu,
        grid_slab_ops_at(
            crate::scene::description::MeshSource::MediumBounds,
            grid_slab_medium(&nvdb, [sigma; 3], [0.0; 3], 0.0),
            CAMERA_INSIDE_TRANSLATE,
            CAMERA_INSIDE_TURN,
        ),
    );
    let fraction = CAMERA_INSIDE_AT.fract();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the camera's index is a small literal"
    )]
    let opening = CAMERA_INSIDE_AT.floor() as i32;
    // ∫_f^1 (1−s)·p(opening) + s·p(opening+1) ds — the part of the voxel the
    // camera stands in that is still in front of it.
    let partial = ramp_profile(opening) * (0.5 - fraction + 0.5 * fraction * fraction)
        + ramp_profile(opening + 1) * 0.5 * (1.0 - fraction * fraction);
    let trapezoid: f64 = partial
        + (opening + 1..32)
            .map(|k| f64::midpoint(ramp_profile(k), ramp_profile(k + 1)))
            .sum::<f64>();
    let tau = f64::from(sigma) * f64::from(GRID_SLAB_VOXEL) * trapezoid;
    assert_transmittance(&gpu, &scene, tau, 0.01);
    let _ = std::fs::remove_file(&nvdb);
}

/// Gate 14: editing a medium's coefficients re-uploads no grid.
///
/// The pool keys residency on (canonical file, grid name) and never
/// evicts, so an edit that leaves the `VolumeSource` alone finds every grid
/// it needs already there — [`GridPool::resident_bytes`] is flat across the
/// rebuild, and flat means nothing streamed from disk. That is the property
/// worth pinning: grids are the largest thing a scene owns (the wdas cloud
/// is 1.6 GiB), and re-streaming one on a coefficient slider would make
/// heterogeneous media uneditable regardless of how fast the tracker runs.
///
/// The render after the edit is half the gate. A pool that kept its bytes
/// but dropped the *offsets* would also read flat, so the edited scene has
/// to come back at the new σ — Beer–Lambert at three times the absorption
/// it was prepped with, off the same resident payload.
///
/// Verified by mutation to see the residency lookup fall through: the pool
/// doubles, 885 620 bytes against 442 804.
#[test]
fn a_coefficient_edit_re_uploads_no_grid() {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, Op, SettingsPatch,
    };
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = constant_grid_fixture("edit", GRID_SLAB_DENSITY) else {
        return;
    };
    let (before, after) = (0.15, 0.45);
    let mut ops = vec![
        Op::Settings(SettingsPatch::new("settings")),
        Op::Camera(CameraPatch {
            position: Some([0.0; 3]),
            look_at: Some([0.0, 0.0, -1.0]),
            vfov_degrees: Some(2.0),
            ..CameraPatch::new("camera")
        }),
        Op::Environment(EnvironmentPatch::new("sky")),
    ];
    ops.extend(grid_slab_ops(
        crate::scene::description::MeshSource::MediumBounds,
        grid_slab_medium(&nvdb, [before; 3], [0.0; 3], 0.0),
    ));
    let mut description = crate::scene::description::SceneDescription::new();
    description
        .apply(&ChangeSet { ops })
        .expect("the slab set is valid");
    let mut scene = Scene::prep(&gpu, &mut description).expect("prep the slab scene");
    let resident = scene.grid_bytes();
    assert!(resident > 0, "the slab's grid should be resident");

    // The edit: absorption only. The `VolumeSource` is untouched, so the
    // medium is dirty and the grid is not.
    description
        .apply(&ChangeSet {
            ops: vec![Op::Medium(crate::scene::changeset::MediumPatch {
                absorption: Some([after; 3]),
                ..crate::scene::changeset::MediumPatch::new("cloud")
            })],
        })
        .expect("the edit is valid");
    let dirty = description.take_dirty();
    scene
        .update(&gpu, &description, &dirty)
        .expect("rebuild the edited scene");
    assert_eq!(
        scene.grid_bytes(),
        resident,
        "a coefficient edit grew the grid pool — the residency cache missed"
    );
    assert_transmittance(
        &gpu,
        &scene,
        f64::from(GRID_SLAB_DENSITY * after * GRID_SLAB_EXTENT),
        0.01,
    );
    let _ = std::fs::remove_file(&nvdb);
}

/// The 11-c gate: the three light-sampling strategies must agree on a scene
/// where density grids stand between the lights and everything else.
///
/// This is the only gate that can see a wrong shadow transmittance through a
/// grid. BSDF-only never queues a connection, so every other grid test in
/// this file measures the tracker alone; next-event-only measures the shadow
/// pass alone — its connections are ratio-tracked through the cloud, and
/// nothing else decides the image. Agreement between the two is therefore
/// agreement between the tracker and the estimator that must integrate the
/// same medium.
///
/// Verified by mutation to see: no grid transmittance at all (20.4 %),
/// grids named by crossings but not by the launching vertex's membership
/// (14.2 %), only the first grid of an overlapping pair (6.3 %), a grid's
/// extinction summed in closed form as well as tracked (6.7 %), and a
/// lattice span truncated to half its length (8.0 %) — against a clean
/// margin of 0.9 % and a 3 % bound.
///
/// One thing it provably cannot see: an unbiased-roulette mistake. A
/// maximally broken `SHADOW_ROULETTE` — survivors never scaled back up —
/// moves this scene's mean by 0.013 %, because a threshold of 0.05 bounds
/// how much of the image can depend on the connections it plays against.
#[test]
fn light_sampling_strategies_agree_across_a_grid_volume() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let Some(nvdb) = constant_grid_fixture("shadow", GRID_SLAB_DENSITY) else {
        return;
    };
    let scene = grid_shadow_scene(&gpu, &nvdb);
    let (size, samples) = (24, 192);
    let mean = |light_sampling| {
        let sum = strategy_sum(&gpu, &scene, size, samples, light_sampling);
        let means = channel_means(&sum, size, samples);
        (means[0] + means[1] + means[2]) / 3.0
    };
    let mis = mean(LightSampling::Mis);
    let bsdf = mean(LightSampling::BsdfOnly);
    let nee = mean(LightSampling::NeeOnly);
    let _ = std::fs::remove_file(&nvdb);
    assert!(mis > 0.01, "the scene should be lit, got mean {mis}");
    for (name, value) in [("BSDF-only", bsdf), ("NEE-only", nee)] {
        let deviation = (value - mis).abs() / mis;
        assert!(
            deviation < 0.03,
            "{name} disagrees with MIS: {value} vs {mis} ({deviation:.4} relative)"
        );
    }
}

/// That gate's scene: the constant-density fixture as a cloud hanging over a
/// matte floor, under a white sky, with a small emitter buried inside it — a
/// second, smaller placement of the same grid pushed into its side, and a
/// homogeneous haze standing around both.
///
/// Every way a connection can meet a medium is in this one frame. It starts
/// inside a grid (the scatter vertices, and the buried emitter's), ends
/// inside one (every connection aimed at that emitter), crosses one from
/// outside (the floor's), crosses *two* where the placements overlap — which
/// must be two independent tracks, one per placement — and crosses the haze
/// alongside all of it, whose closed-form extent must compose with the
/// tracked ones rather than double-count them.
fn grid_shadow_scene(gpu: &Context, nvdb: &std::path::Path) -> Scene {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, MaterialPatch, MediumPatch,
        MeshPatch, Op, SettingsPatch,
    };
    use crate::scene::description::{MeshSource, Transform};

    // The fixture box is [−0.05, 1.6] on each axis; this lands its center at
    // (0, 1, 0), hanging over the floor with room to see under it.
    let center = [-0.775, 0.225, -0.775];
    let ops = vec![
        Op::Settings(SettingsPatch::new("settings")),
        Op::Camera(CameraPatch {
            position: Some([0.0, 1.6, 4.5]),
            look_at: Some([0.0, 0.9, 0.0]),
            vfov_degrees: Some(45.0),
            ..CameraPatch::new("camera")
        }),
        Op::Environment(EnvironmentPatch::new("sky")),
        Op::Mesh(MeshPatch {
            source: Some(ground_quad(4.0)),
            ..MeshPatch::new("floor")
        }),
        Op::Mesh(MeshPatch {
            source: Some(MeshSource::MediumBounds),
            ..MeshPatch::new("shell")
        }),
        // A light *inside* the cloud: its connections start and end within
        // the shell, crossing no face of it, so nothing but the launching
        // vertex's membership can name the grid they are tracked through.
        Op::Mesh(MeshPatch {
            source: Some(inline_box([-0.1, 0.9, -0.1], 0.2)),
            ..MeshPatch::new("ember")
        }),
        Op::Material(Box::new(MaterialPatch {
            emission_luminance: Some(30.0),
            ..MaterialPatch::new("glow")
        })),
        Op::Material(Box::new(MaterialPatch {
            base_color: Some(crate::scene::description::Texturable::Constant([0.6; 3])),
            ..MaterialPatch::new("matte")
        })),
        // Chromatic, so a channel dropped from the transmittance reads as a
        // tint rather than a level.
        Op::Medium(grid_slab_medium(
            nvdb,
            [0.30, 0.18, 0.09],
            [1.5, 2.4, 3.3],
            0.3,
        )),
        Op::Instance(InstancePatch {
            mesh: Some("floor".into()),
            material: Some("matte".into()),
            ..InstancePatch::new("ground")
        }),
        Op::Instance(InstancePatch {
            mesh: Some("ember".into()),
            material: Some("glow".into()),
            ..InstancePatch::new("light")
        }),
        Op::Instance(InstancePatch {
            mesh: Some("shell".into()),
            material: Some("matte".into()),
            medium: Some(Some("cloud".into())),
            transforms: Some(vec![Transform::Trs {
                translate: center,
                rotate_degrees: [0.0; 3],
                scale: [1.0; 3],
            }]),
            ..InstancePatch::new("cloud")
        }),
        // The same medium placed a second time, half-size and sunk into the
        // first around the emitter, so that every connection to it — and
        // every scatter near it — crosses two grids and must be tracked
        // twice. A different transform is a different record, so this is
        // also where a traversal that folded two placements into one, or
        // tracked only the first it found, shows up.
        Op::Instance(InstancePatch {
            mesh: Some("shell".into()),
            material: Some("matte".into()),
            medium: Some(Some("cloud".into())),
            transforms: Some(vec![Transform::Trs {
                translate: [-0.375, 0.625, -0.375],
                rotate_degrees: [0.0; 3],
                scale: [0.5; 3],
            }]),
            ..InstancePatch::new("wisp")
        }),
        // And a homogeneous volume around both, whose extent the same
        // traversal measures in closed form on the same connections.
        Op::Medium(MediumPatch {
            absorption: Some([0.04; 3]),
            scattering: Some([0.10; 3]),
            ..MediumPatch::new("haze")
        }),
        Op::Mesh(MeshPatch {
            source: Some(inline_box([-1.5, 0.05, -1.5], 3.0)),
            ..MeshPatch::new("hazebox")
        }),
        Op::Instance(InstancePatch {
            mesh: Some("hazebox".into()),
            material: Some("matte".into()),
            medium: Some(Some("haze".into())),
            ..InstancePatch::new("fog")
        }),
    ];
    let mut description = crate::scene::description::SceneDescription::new();
    description
        .apply(&ChangeSet { ops })
        .expect("the cloud set is valid");
    Scene::prep(gpu, &mut description).expect("prep the cloud scene")
}

/// Milk at albedo exactly 1: a white transmission color makes the whole
/// `σ_t` the gray shift, and a neutral scatter makes `σ_s` equal it bit
/// for bit — the furnace-exact scattering interior the tests below share.
fn albedo_one_milk() -> Material {
    Material {
        transmission_color: Vec3::ONE,
        transmission_depth: 1.0,
        transmission_scatter: Vec3::ONE,
        transmission_scatter_anisotropy: 0.3,
        ..Material::glass(0.1, 1.5)
    }
}

/// A sphere placed for the scattering-interior tests: translated to `at`,
/// then scaled.
fn sphere_at(
    mesh: crate::scene::Mesh,
    at: Vec3,
    scale: f32,
    material: Material,
    interior_priority: u32,
) -> Object {
    Object {
        mesh,
        transform: Mat4::from_translation(at) * Mat4::from_scale(Vec3::splat(scale)),
        material,
        medium: None,
        interior_priority,
    }
}

/// The scattering-interior tests' one viewpoint: a camera at the origin
/// looking down −Z, under a black sky, so every photon is the scene's own.
fn scene_in_the_dark(gpu: &Context, objects: &[Object]) -> Scene {
    Scene::new(
        gpu,
        objects,
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 60.0,
            lens: None,
        },
        &Environment::constant(Vec3::ZERO),
    )
    .expect("scene")
}

/// The closed-form family's one viewpoint: the origin looking down −Z
/// through a one-degree field under a constant sky, so every ray's path
/// length through whatever stands on the axis is the center ray's to
/// within 4e-5 — what lets these tests assert Beer–Lambert exactly rather
/// than on a loose mean.
fn axial_scene(gpu: &Context, objects: &[Object], sky: f32) -> Scene {
    Scene::new(
        gpu,
        objects,
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 1.0,
            lens: None,
        },
        &Environment::constant(Vec3::splat(sky)),
    )
    .expect("scene")
}

/// Channel-0 mean of a BSDF-only render — the reduction every axial
/// closed-form test asserts against. Asserts every path finished on the
/// way: a closed form met by a population with a few dropped paths is met
/// by accident.
fn axial_mean(gpu: &Context, scene: &Scene, size: u32, samples: u32) -> f64 {
    let sum = bsdf_only_sum(gpu, scene, size, samples);
    assert_all_paths_finished(&sum, samples);
    sum.chunks_exact(4).map(|p| f64::from(p[0])).sum::<f64>() / f64::from(samples * size * size)
}

/// Every path finished exactly once: each sample's one terminal add
/// carries alpha 1, so a pixel's alpha is exactly its sample count — the
/// mechanical check that no path was dropped or double-terminated.
fn assert_all_paths_finished(sum: &[f32], samples: u32) {
    for chunk in sum.chunks_exact(4) {
        assert!(
            (chunk[3] - samples as f32).abs() < 1e-3,
            "every path must finish exactly once, got alpha {}",
            chunk[3]
        );
    }
}

/// The furnace, inside milk: a *scattering refractive interior* at albedo
/// exactly 1 (see [`albedo_one_milk`]) in the emissive shell, with a
/// suppressed lower-priority boundary buried inside it. The march splits
/// the interior's segments at that boundary and restarts its distance
/// draws per leg, next-event estimation is
/// skipped at every vertex the milk dominates, and the weight-1 emission
/// `throughput.w = 0` promises is what the escaping paths carry — so if
/// any of that chain loses or double-counts a factor, the frame drifts
/// off the shell's radiance. MIS mode deliberately: BSDF-only would never
/// read the skip.
///
/// A furnace at albedo 1 cannot tell "scattering, correctly weighted"
/// from "accidentally inert", so the control renders the same scene with
/// the scatter unauthored (clear glass) on the same sample streams and
/// requires a different image: the volume stage demonstrably fired.
#[test]
fn a_scattering_interior_keeps_the_furnace_closed() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let emission = 0.5;
    let milk = albedo_one_milk();
    // A tinted absorber the milk outranks: if its interfaces shade rather
    // than cross — or the legs they split misweight — the furnace leaks.
    let lurker = Material {
        transmission_color: Vec3::new(0.2, 0.6, 0.9),
        transmission_depth: 0.2,
        ..Material::glass(0.05, 1.33)
    };
    let scene = |content: Material| {
        scene_in_the_dark(
            &gpu,
            &[
                sphere_at(
                    inside_out(crate::scene::icosphere(3)),
                    Vec3::ZERO,
                    2.0,
                    Material::emitter(Vec3::splat(emission)),
                    0,
                ),
                sphere_at(crate::scene::icosphere(3), Vec3::NEG_Z * 1.2, 0.55, content, 2),
                sphere_at(crate::scene::icosphere(2), Vec3::NEG_Z * 1.2, 0.3, lurker, 0),
            ],
        )
    };
    // 64 bounces as the volumetric furnace: the milk is about an optical
    // depth across, and truncated turns are energy, not noise.
    let renderer = Renderer::with_max_bounces(&gpu, 64).expect("renderer");
    let (size, samples) = (32, 128);
    let sum = accumulate_sum(&gpu, &renderer, &scene(milk), size, samples);
    for channel in 0..3 {
        let mean = sum
            .chunks_exact(4)
            .map(|chunk| chunk[channel])
            .sum::<f32>()
            / (size * size * samples) as f32;
        assert!(
            (mean - emission).abs() / emission < 0.015,
            "channel {channel}: milk furnace leaked, mean {mean} vs {emission}"
        );
    }
    assert_all_paths_finished(&sum, samples);
    let clear = accumulate_sum(
        &gpu,
        &renderer,
        &scene(Material {
            transmission_scatter: Vec3::ZERO,
            ..milk
        }),
        size,
        samples,
    );
    assert!(
        sum != clear,
        "the milk rendered exactly as clear glass — no medium event was ever sampled"
    );
}

/// An emitter submerged in a scattering interior is phase sampling's
/// alone: next-event estimation is skipped at every vertex the milk
/// dominates, and the weight-1 emission `throughput.w = 0` promises is
/// the whole estimator. The MIS image must therefore agree with the
/// BSDF-only image, which never had another strategy to begin with.
///
/// This is where the *volume* half of the skip is load-bearing rather
/// than hygiene, and this test fails without it: a connection from a
/// scattering vertex to a submerged emitter crosses no boundary, so it
/// is *visible* — but `trace_shadow` measures volumes, not interiors, so
/// it arrives without the milk's transmittance and reads too bright,
/// while the emission it competes with gets MIS-weighted down. The
/// furnace above cannot see that error: its paths always exit through
/// the wall, whose own skip rewrites w before any emission reads it.
///
/// The *surface* half has no oracle here — a submerged rock's NEE is
/// visible-but-unattenuated too, and the power heuristic leans so far
/// toward NEE for a small emitter that the emission down-weight nearly
/// cancels the excess at any density this scene can converge — so the
/// rock below extends the covered estimator combinations, and the
/// surface half stands on the invariant (one register test, both
/// kernels) plus the furnace's absorption gate rather than on a
/// discriminating bound of its own.
#[test]
fn a_submerged_emitter_is_phase_samplings_alone() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let objects = [
        sphere_at(crate::scene::icosphere(3), Vec3::NEG_Z * 1.2, 0.55, albedo_one_milk(), 0),
        sphere_at(
            crate::scene::icosphere(2),
            Vec3::NEG_Z * 1.2,
            0.15,
            Material::emitter(Vec3::splat(5.0)),
            0,
        ),
        // The rock: a submerged diffuse surface lit by the emitter beside
        // it — the surface-half sub-path the doc above names.
        sphere_at(
            crate::scene::icosphere(2),
            Vec3::new(0.25, 0.0, -1.2),
            0.12,
            Material::matte(Vec3::splat(0.8), 0.3),
            0,
        ),
    ];
    let scene = scene_in_the_dark(&gpu, &objects);
    let (size, samples) = (32, 256);
    let renderer = Renderer::with_max_bounces(&gpu, 64).expect("renderer");
    let mis = accumulate_sum(&gpu, &renderer, &scene, size, samples);
    let bsdf = bsdf_only_sum_deep(&gpu, &scene, size, samples, 64);
    let mean = |sum: &[f32]| {
        sum.chunks_exact(4).map(|chunk| f64::from(chunk[0])).sum::<f64>()
            / f64::from(size * size * samples)
    };
    let (mis, bsdf) = (mean(&mis), mean(&bsdf));
    assert!(
        (mis - bsdf).abs() / bsdf < 0.05,
        "the two strategies disagree on a submerged emitter: MIS {mis} vs BSDF-only {bsdf}"
    );
}

/// Suppression = deletion, at real parameters: a chromatic, absorbing
/// *and* scattering juice whose interior buries a lower-priority shell
/// must render as if the shell were deleted. Bit identity cannot say it
/// — every suppressed crossing splits a march segment, and the split
/// legs draw from different keys than the unsplit segment — so the claim
/// is statistical: relMSE against the deleted scene, bounded by twice
/// the deleted scene's own stream-to-stream noise floor, calibrated here
/// on the same pixel counts rather than pinned as a magic constant. What
/// albedo-1 cancellation hides above — the chromatic mixture pdf, the
/// `σ_s`/`σ_t` bookkeeping, the gray shift — is exactly what drifts this
/// bound if it drifts at all.
#[test]
fn a_suppressed_boundary_inside_juice_is_no_boundary_at_all() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    // Chromatic on both axes, but light must *survive* the core: an
    // opaque-dark juice would extinguish everything near the lurker and
    // the test would pass with suppression broken, having nothing to see.
    let juice = Material {
        transmission_color: Vec3::new(0.9, 0.8, 0.7),
        transmission_depth: 0.25,
        transmission_scatter: Vec3::new(0.5, 0.4, 0.3),
        transmission_scatter_anisotropy: 0.3,
        ..Material::glass(0.1, 1.5)
    };
    let lurker = Material {
        transmission_color: Vec3::new(0.1, 0.5, 0.9),
        transmission_depth: 0.1,
        ..Material::glass(0.05, 1.33)
    };
    let scene = |with_lurker: bool| {
        let mut objects = vec![
            sphere_at(
                inside_out(crate::scene::icosphere(3)),
                Vec3::ZERO,
                2.0,
                Material::emitter(Vec3::splat(0.5)),
                0,
            ),
            sphere_at(crate::scene::icosphere(3), Vec3::NEG_Z * 1.2, 0.55, juice, 2),
        ];
        if with_lurker {
            objects.push(sphere_at(
                crate::scene::icosphere(2),
                Vec3::NEG_Z * 1.2,
                0.4,
                lurker,
                0,
            ));
        }
        scene_in_the_dark(&gpu, &objects)
    };
    // Relative squared error, summed over pixels and channels — both
    // sides of the bound below use the same pixel count, so the sum
    // stands in for the mean.
    let relmse = |a: &[f32], b: &[f32]| {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(pa, pb)| pa[..3].iter().zip(&pb[..3]))
            .map(|(&va, &vb)| {
                let delta = f64::from(va - vb);
                delta * delta / (f64::from(vb) * f64::from(vb) + 1e-4)
            })
            .sum::<f64>()
    };
    let renderer = Renderer::with_max_bounces(&gpu, 64).expect("renderer");
    let (size, samples) = (32, 192);
    let scale = |sum: Vec<f32>| {
        sum.iter()
            .map(|value| value / samples as f32)
            .collect::<Vec<f32>>()
    };
    // Two independent streams of the deleted scene from one film — the
    // second block of samples is the difference of the running sums.
    let deleted = scene(false);
    let mut film = Film::new(&gpu, size, size).expect("film");
    for _ in 0..samples {
        renderer
            .accumulate(&gpu, &deleted, &mut film)
            .expect("accumulate");
    }
    let stream0 = download_f32(&gpu, &film.beauty.sum);
    for _ in 0..samples {
        renderer
            .accumulate(&gpu, &deleted, &mut film)
            .expect("accumulate");
    }
    let stream1: Vec<f32> = download_f32(&gpu, &film.beauty.sum)
        .iter()
        .zip(&stream0)
        .map(|(total, first)| total - first)
        .collect();
    let suppressed = accumulate_sum(&gpu, &renderer, &scene(true), size, samples);
    assert_all_paths_finished(&suppressed, samples);
    let (deleted0, deleted1, suppressed) = (scale(stream0), scale(stream1), scale(suppressed));
    let floor = relmse(&deleted0, &deleted1);
    let gap = relmse(&suppressed, &deleted1);
    assert!(floor > 0.0, "the two calibration streams cannot be identical");
    assert!(
        gap < 2.0 * floor,
        "suppressed vs deleted drifted past the noise floor: {gap} vs floor {floor}"
    );
}

/// Looking through boxes of fog along a nearly axial ray: what arrives is
/// the sky times Beer–Lambert over exactly the extent the boxes bound.
///
/// The one-box case pins the whole march — the boundary routes to this
/// stage, the crossing enters the medium, the far crossing leaves it, and
/// the sky beyond arrives through vacuum. The inside-out box must answer
/// identically: membership is crossing parity, not winding, so the
/// USD-pipeline mistake bounds exactly its own interior rather than
/// fogging the world outside it. The two-box case pins the rule that
/// makes overlap order-free: where they intersect the path is inside
/// both, and their extinctions *add*, so the answer is the same product
/// whether the boxes overlap, touch, or stand apart. The four disjoint
/// boxes pin the two things a chain of crossings can get wrong: a slot
/// freed on the way out has to be usable on the way into the next box,
/// and the eighth crossing is the last one `MARCH_BOUNDARY_CAP` models.
/// Five boxes are ten crossings — past the cap — and pin the overflow
/// from both sides at once: the march models exactly the first four
/// boxes, then runs *through* the fifth to the sky, so the answer is the
/// eight-meter product exactly. A larger cap would darken by the fifth
/// box; a march that stopped at it would read a wall that is not there.
///
/// A one-degree field keeps every ray within half a degree of the axis, so
/// the slant a wide frame would add to each crossing stays under 4e-5 of the
/// thickness — far below the tolerance, and the reason this can assert on a
/// closed form rather than on a mean.
#[test]
fn bounded_volumes_absorb_over_exactly_the_extent_they_bound() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = Vec3::splat(0.75);
    let sigma = Vec3::new(0.15, 0.35, 0.6);
    // Half-extent 1 at z = −5 and z = −6: two meters of fog each, one of
    // which they share.
    let fog = |z: f32, medium: crate::scene::Medium| Object {
        mesh: crate::scene::cube(1.0),
        transform: Mat4::from_translation(Vec3::Z * z),
        material: Material::matte(Vec3::ONE, 0.0),
        medium: Some(medium),
        interior_priority: 0,
    };
    let absorbing = |scale: f32| crate::scene::Medium {
        absorption: sigma * scale,
        scattering: Vec3::ZERO,
        anisotropy: 0.0,
        volume: None,
    };
    let row = |boxes: u8| -> Vec<Object> {
        (0..boxes)
            .map(|n| fog(3.0f32.mul_add(-f32::from(n), -5.0), absorbing(1.0)))
            .collect()
    };
    let (size, samples) = (8, 64);
    for (name, objects, depth) in [
        ("one box", vec![fog(-5.0, absorbing(1.0))], 2.0),
        (
            "one box, inside out",
            vec![Object {
                mesh: inside_out(crate::scene::cube(1.0)),
                ..fog(-5.0, absorbing(1.0))
            }],
            2.0,
        ),
        (
            "two overlapping boxes",
            vec![fog(-5.0, absorbing(1.0)), fog(-6.0, absorbing(2.0))],
            2.0f32.mul_add(2.0, 2.0),
        ),
        ("four boxes in a row", row(4), 8.0),
        // Ten crossings against a cap of eight: the fifth box is run
        // through untracked, so exactly the first four absorb.
        ("five boxes in a row", row(5), 8.0),
    ] {
        let scene = axial_scene(&gpu, &objects, sky.x);
        let sum = bsdf_only_sum(&gpu, &scene, size, samples);
        assert_all_paths_finished(&sum, samples);
        let count = f64::from(samples * size * size);
        for channel in 0..3 {
            let mean = sum
                .chunks_exact(4)
                .map(|pixel| f64::from(pixel[channel]))
                .sum::<f64>()
                / count;
            // The two-box depth already counts the shared meter twice; a
            // renderer that took one medium instead of their sum would land
            // a third of the way toward `expected`'s square root.
            let expected = f64::from(sky[channel]) * (-f64::from(sigma[channel] * depth)).exp();
            assert!(
                (mean - expected).abs() / expected < 0.005,
                "{name}, channel {channel}: {mean} vs Beer–Lambert's {expected}"
            );
        }
    }
}

/// A refractive interior *displaces* the medium around it — a path inside
/// glass is not also in the fog the glass stands in. A glass box sealed
/// inside a fog volume must read Beer–Lambert with the fog's σ over
/// exactly the fog meters *outside* the glass and the glass's σ over its
/// own meter: `combineMedia` resolving an interior additively — fog and
/// glass at once — would land a factor of `exp(σ_fog · glass extent)` darker,
/// and the surface stage double-applying fog where the volume stage
/// already owned the segment would too. Every medium here only absorbs,
/// so each side of the split is a closed form and the bound stays tight.
#[test]
fn an_interior_displaces_the_medium_it_stands_in() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let (sigma_fog, transmission) = (0.5, 0.85);
    let sigma_glass = -f64::from(transmission).ln(); // per meter: depth is 1
    // Fog z ∈ [−7, −5]; the glass z ∈ [−6.5, −5.5], sealed inside it: one
    // axial meter of glass, and one of fog split around it.
    let objects = [
        Object {
            mesh: crate::scene::cube(1.0),
            transform: Mat4::from_translation(Vec3::Z * -6.0),
            material: Material::matte(Vec3::ONE, 0.0),
            medium: Some(crate::scene::Medium {
                absorption: Vec3::splat(sigma_fog),
                scattering: Vec3::ZERO,
                anisotropy: 0.0,
                volume: None,
            }),
            interior_priority: 0,
        },
        Object {
            mesh: crate::scene::cube(0.5),
            transform: Mat4::from_translation(Vec3::Z * -6.0),
            material: Material {
                transmission_weight: 1.0,
                specular_weight: 0.0,
                transmission_color: Vec3::splat(transmission),
                transmission_depth: 1.0,
                ..Material::matte(Vec3::ONE, 0.0)
            },
            medium: None,
            interior_priority: 0,
        },
    ];
    let (size, samples) = (8, 64);
    let scene = axial_scene(&gpu, &objects, sky);
    let mean = axial_mean(&gpu, &scene, size, samples);
    let expected = f64::from(sky) * (-f64::from(sigma_fog) - sigma_glass).exp();
    assert!(
        (mean - expected).abs() / expected < 0.005,
        "glass in fog: {mean} vs Beer–Lambert's {expected} over one meter of each \
         (fog filling the glass too would read {})",
        f64::from(sky) * (-f64::from(sigma_fog) * 2.0 - sigma_glass).exp()
    );
}

/// One refractive interior inside another: the medium set has to hold both,
/// and leaving the inner one has to put the path back in the outer rather
/// than out of everything.
///
/// Two boxes of the same glass, so the answer does not depend on which of
/// the two the path is judged to be *in* while it is in both — every meter
/// between the outer faces absorbs the same. A set one entry deep cannot
/// hold both: it overwrites the outer on the way in, empties on the way out
/// of the inner, and leaves the far gap unabsorbed, landing three meters of
/// extinction short of four.
///
/// The glass is authored at `specular_weight` 0, which drives the closure's
/// relative IOR to 1: no refraction to bend the axial ray, no Fresnel to
/// take a share off each of the four crossings, and what reaches the camera
/// is Beer–Lambert alone.
#[test]
fn a_nested_interior_gives_the_path_back_to_the_one_around_it() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let transmission = 0.85;
    let sigma = -f64::from(transmission).ln(); // per meter: depth is 1
    let glass = Material {
        transmission_weight: 1.0,
        specular_weight: 0.0,
        transmission_color: Vec3::splat(transmission),
        transmission_depth: 1.0,
        ..Material::matte(Vec3::ONE, 0.0)
    };
    let box_of = |half: f32| Object {
        mesh: crate::scene::cube(half),
        transform: Mat4::from_translation(Vec3::Z * -6.0),
        material: glass,
        medium: None,
        interior_priority: 0,
    };
    let (size, samples) = (16, 256);
    let scene = axial_scene(&gpu, &[box_of(2.0), box_of(1.0)], sky);
    let mean = axial_mean(&gpu, &scene, size, samples);
    let expected = f64::from(sky) * (-sigma * 4.0).exp();
    assert!(
        (mean - expected).abs() / expected < 0.015,
        "nested glass: {mean} vs Beer–Lambert's {expected} over four meters \
         (three would be {})",
        f64::from(sky) * (-sigma * 3.0).exp()
    );
}

/// What the camera itself is inside of is resolved by tracing, not
/// authored: sample 0 dispatches one `resolve_camera` thread whose ray
/// walks from the camera position to infinity, toggling membership at
/// every boundary surface it crosses, and writes the resulting medium set
/// into the scene table — the word bounce 0 seeds every path from. Four
/// cameras against one constellation, with the exact entries — flags and
/// all — read back from the table: entries built anywhere but
/// `stackEntry` would rank as something they are not.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one constellation, five cameras — splitting it would rebuild the scene per case"
)]
fn the_camera_seed_names_exactly_what_the_camera_is_inside_of() {
    use crate::scene::{
        Medium, STACK_EMPTY, STACK_INTERIOR, STACK_PRIORITY_SHIFT, STACK_SCATTERING,
    };
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let glass = Material {
        transmission_weight: 1.0,
        transmission_color: Vec3::splat(0.9),
        transmission_depth: 1.0,
        ..Material::glass(0.0, 1.5)
    };
    // The inner solid scatters and outranks the outer, so its entry
    // carries every flag the word can: interior, priority, scattering.
    let milk = Material {
        transmission_scatter: Vec3::splat(0.5),
        ..glass
    };
    let fog = Medium {
        absorption: Vec3::splat(0.1),
        scattering: Vec3::splat(0.4),
        anisotropy: 0.0,
        volume: None,
    };
    // Instance 0: outer glass box, z in [-8, -4]. Instance 1: the milk
    // inside it, z in [-7, -5], priority 1. Instance 2: a fog cube far
    // off at x = 10, bounding a volume rather than closing an interior.
    // Instance 3: an opaque prop sitting exactly on the resolve ray of the
    // "inside the outer glass only" camera — walls hide nothing, because
    // membership is geometric, not visibility. Instance 4: a thin-walled
    // glass box far off at y = 20 — thin walls bound no interior, so a
    // camera inside one is inside nothing.
    let objects = [
        Object {
            mesh: crate::scene::cube(2.0),
            transform: Mat4::from_translation(Vec3::Z * -6.0),
            material: glass,
            medium: None,
            interior_priority: 0,
        },
        Object {
            mesh: crate::scene::cube(1.0),
            transform: Mat4::from_translation(Vec3::Z * -6.0),
            material: milk,
            medium: None,
            interior_priority: 1,
        },
        Object {
            mesh: crate::scene::cube(1.0),
            transform: Mat4::from_translation(Vec3::X * 10.0),
            material: Material::matte(Vec3::ONE, 0.0),
            medium: Some(fog),
            interior_priority: 0,
        },
        Object {
            mesh: crate::scene::cube(0.15),
            // (0, 0, -4.5) + 0.3 · the walk's fixed (1, 2, 3)/√14. If
            // `RESOLVE_DIRECTION` in resolve_camera.slang ever changes,
            // recompute this position — off the walk's ray the prop is
            // decorative and the opaque-surfaces case asserts nothing.
            transform: Mat4::from_translation(Vec3::new(0.0802, 0.1604, -4.2595)),
            material: Material::matte(Vec3::splat(0.5), 0.5),
            medium: None,
            interior_priority: 0,
        },
        Object {
            mesh: crate::scene::cube(1.0),
            transform: Mat4::from_translation(Vec3::Y * 20.0),
            material: glass.thin_walled(),
            medium: None,
            interior_priority: 0,
        },
    ];
    let outer = STACK_INTERIOR;
    let inner = 1 | STACK_INTERIOR | (1 << STACK_PRIORITY_SHIFT) | STACK_SCATTERING;
    let wavefront = Wavefront::new(
        &gpu,
        &Kernels::embedded(),
        Wavefront::DEFAULT_CAPACITY,
        2,
        LightSampling::Mis,
    )
    .expect("wavefront");
    let radiance = gpu
        .create_buffer(
            "test.radiance",
            8 * 8 * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )
        .expect("radiance buffer");
    for (name, position, mut expected) in [
        ("outside everything", Vec3::ZERO, vec![]),
        // The opaque prop stands square on this camera's resolve ray and
        // must change nothing: it is not a boundary, so it neither ends
        // the walk nor spends its cap.
        ("inside the outer glass only", Vec3::new(0.0, 0.0, -4.5), vec![outer]),
        ("inside both solids", Vec3::new(0.0, 0.0, -6.0), vec![outer, inner]),
        ("inside the fog", Vec3::new(10.0, 0.0, 0.0), vec![2]),
        ("inside a thin-walled box", Vec3::new(0.0, 20.0, 0.0), vec![]),
    ] {
        let camera = Camera {
            position,
            look_at: position + Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens: None,
        };
        let scene = Scene::new(
            &gpu,
            &objects,
            camera,
            &Environment::constant(Vec3::splat(0.5)),
        )
        .expect("scene");
        wavefront
            .trace(&gpu, &scene, &radiance, 8, 8, 0)
            .expect("trace");
        let table = gpu.download_buffer(scene.table()).expect("download");
        let seed: [u32; 4] = bytemuck::pod_read_unaligned(&table[table.len() - 16..]);
        let mut entries: Vec<u32> = seed.into_iter().filter(|&e| e != STACK_EMPTY).collect();
        entries.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            entries, expected,
            "{name}: seed {seed:08x?} carries the wrong membership"
        );
    }
}

/// The case the camera seed exists for: a camera *inside* tinted glass.
/// The first segment runs through the interior, and before the seed it was
/// attenuated by nothing — worse, the exit face then toggled the box *into*
/// the empty set, and the path left carrying glass it was no longer in.
/// With the seed, what reaches the camera is Beer–Lambert over the axial
/// two meters to the face, exactly.
///
/// `specular_weight` 0 drives the interface's relative IOR to 1, as in the
/// nested-interior test above: no refraction to bend the axial ray, no
/// Fresnel share — the sky through one tinted pane.
#[test]
fn a_camera_inside_tinted_glass_sees_beer_lambert_from_its_first_segment() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let transmission = 0.85;
    let sigma = -f64::from(transmission).ln(); // per meter: depth is 1
    let glass = Material {
        transmission_weight: 1.0,
        specular_weight: 0.0,
        transmission_color: Vec3::splat(transmission),
        transmission_depth: 1.0,
        ..Material::matte(Vec3::ONE, 0.0)
    };
    let scene = axial_scene(
        &gpu,
        &[Object {
            mesh: crate::scene::cube(2.0),
            transform: Mat4::IDENTITY,
            material: glass,
            medium: None,
            interior_priority: 0,
        }],
        sky,
    );
    let (size, samples) = (16, 256);
    let mean = axial_mean(&gpu, &scene, size, samples);
    let expected = f64::from(sky) * (-sigma * 2.0).exp();
    assert!(
        (mean - expected).abs() / expected < 0.015,
        "camera inside glass: {mean} vs Beer–Lambert's {expected} \
         (the unattenuated sky is {sky})"
    );
}

/// Two nested solids of the same glass are one solid. The interface between
/// them has a relative IOR of exactly 1 — there is no boundary there in any
/// physical sense — so putting a second box inside the first must change
/// nothing about what reaches the camera.
///
/// This is what a closure that always refracts against *vacuum* cannot do:
/// with a one-deep interior the exterior was always empty and 1.0 was right,
/// but once interiors nest, the inner box's two faces refract at 1.5 and
/// 1/1.5 against air that is not there, taking a Fresnel share off each.
/// Full specular weight, so the authored IOR reaches the interface
/// unremapped.
///
/// The glass has to *absorb* for the difference to be visible at all: under
/// a uniform sky a lossless dielectric is a furnace, and light turned back
/// by a spurious interface returns exactly what it would have delivered
/// anyway. With absorption the two are no longer interchangeable — a path
/// reflected at the inner face escapes through one meter of glass instead of
/// four, and arrives far *brighter* than the one that went through. That
/// puts the box inside 3.5% over the box alone before this was fixed, and
/// 0.06% after — the bar sits between them.
#[test]
fn nested_solids_of_one_glass_have_no_interface_between_them() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let glass = Material {
        transmission_color: Vec3::splat(0.6),
        transmission_depth: 1.0,
        ..Material::glass(0.0, 1.5)
    };
    let box_of = |half: f32| Object {
        mesh: crate::scene::cube(half),
        transform: Mat4::from_translation(Vec3::Z * -6.0),
        material: glass,
        medium: None,
        interior_priority: 0,
    };
    let (size, samples) = (16, 1024);
    let mean_of =
        |objects: &[Object]| axial_mean(&gpu, &axial_scene(&gpu, objects, sky), size, samples);
    let alone = mean_of(&[box_of(2.0)]);
    let nested = mean_of(&[box_of(2.0), box_of(1.0)]);
    assert!(
        (nested - alone).abs() / alone < 0.005,
        "a box inside a box of the same glass: {nested} vs {alone} for the box alone"
    );
}

/// Suppression is deletion. A glass sphere sealed inside a glass block of
/// higher priority is not there at all: both of its interfaces are false,
/// the path crosses them without spending a bounce or a surface event, and
/// the render must come out *bit-identical* to the block on its own.
///
/// Not a tolerance — an equality, and deliberately so. The march's legs
/// through the sphere sample the dominant interior, where `combineMedia`
/// returns vacuum, so each weight is exactly 1.0; the extra legs draw from
/// Sobol dimensions that are indexed rather than consumed, so spending
/// `VOLUME_DISTANCE` costs the other dimensions nothing; and radiance
/// writes are pixel-owned, so the extra queue hop cannot reorder anything.
/// If this ever needs a tolerance, something else changed.
///
/// The last case is the test's own falsification: give the two solids
/// *equal* priority and the sphere becomes a pair of true interfaces
/// again, which must not match. And the whole thing runs a second time at
/// a four-bounce cap, which is what pins "a false crossing costs no
/// bounce": four is exactly enough to enter the block, cross the sphere,
/// leave the block and reach the sky, so charging the crossing a bounce
/// strands the path at the cap with nothing. The equal-priority scene,
/// which really does spend two bounces in there, is stranded — that is
/// what the second assertion sees.
#[test]
fn an_interior_inside_a_higher_priority_one_is_not_there() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let glass = Material {
        transmission_color: Vec3::new(0.7, 0.8, 0.9),
        transmission_depth: 1.0,
        ..Material::glass(0.0, 1.5)
    };
    let block = |priority: u32| Object {
        mesh: crate::scene::cube(2.0),
        transform: Mat4::from_translation(Vec3::Z * -6.0),
        material: glass,
        medium: None,
        interior_priority: priority,
    };
    // Half the block's extent and concentric with it, so it is sealed
    // inside with room to spare on every face.
    let sphere = |priority: u32| Object {
        mesh: crate::scene::icosphere(3),
        transform: Mat4::from_translation(Vec3::Z * -6.0),
        material: glass,
        medium: None,
        interior_priority: priority,
    };
    let camera = Camera {
        position: Vec3::ZERO,
        look_at: Vec3::NEG_Z,
        up: Vec3::Y,
        vfov_degrees: 20.0,
        lens: None,
    };
    let sky_image = Environment::constant(Vec3::splat(0.75));
    let (size, samples) = (24, 64);
    for bounces in [Wavefront::DEFAULT_MAX_BOUNCES, 4] {
        let render = |objects: &[Object]| {
            let scene = Scene::new(&gpu, objects, camera, &sky_image).expect("scene");
            bsdf_only_sum_deep(&gpu, &scene, size, samples, bounces)
        };
        let alone = render(&[block(1)]);
        assert_eq!(
            render(&[block(1), sphere(0)]),
            alone,
            "a sealed lower-priority sphere must render as though it were not \
             there ({bounces} bounces)"
        );
        assert_ne!(
            render(&[block(0), sphere(0)]),
            alone,
            "equal priorities must leave the sphere's interfaces real — \
             matching the block alone means the equality above has no power \
             ({bounces} bounces)"
        );
    }
}

/// The drink: a glass wall the liquid interpenetrates, which is the whole
/// reason priority exists. Straight down the axis with the specular weight
/// at zero, so no interface reflects or bends and what arrives is
/// Beer–Lambert alone — over each solid's *dominant* extent, which is the
/// thing priority decides.
///
/// The glass outranks the water, so the overlap is glass: two meters of it
/// from the near face to the far one, and only the half meter of water
/// that sticks out past the glass counts as water. Without priority the
/// water's near face is a real interface, and everything past it absorbs at
/// whichever of the two `interiorOf` happened to rank higher — a different
/// answer, which the second assertion pins.
#[test]
fn the_higher_priority_solid_owns_the_overlap() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let (glass_t, water_t) = (0.55, 0.85);
    let solid = |transmission: f32| Material {
        transmission_weight: 1.0,
        specular_weight: 0.0,
        transmission_color: Vec3::splat(transmission),
        transmission_depth: 1.0,
        ..Material::matte(Vec3::ONE, 0.0)
    };
    // Glass spans z ∈ [−7, −5], water z ∈ [−7.5, −5.5]: they share a meter
    // and a half, and the water reaches half a meter past the glass.
    let solid_at = |z: f32, transmission: f32, priority: u32| Object {
        mesh: crate::scene::cube(1.0),
        transform: Mat4::from_translation(Vec3::Z * z),
        material: solid(transmission),
        medium: None,
        interior_priority: priority,
    };
    let (size, samples) = (16, 256);
    let mean_of = |glass_priority: u32| {
        let objects = [
            solid_at(-6.0, glass_t, glass_priority),
            solid_at(-6.5, water_t, 0),
        ];
        axial_mean(&gpu, &axial_scene(&gpu, &objects, sky), size, samples)
    };
    let sigma = |transmission: f32| -f64::from(transmission).ln(); // depth is 1
    let expected = f64::from(sky) * (-sigma(glass_t) * 2.0 - sigma(water_t) * 0.5).exp();
    let ranked = mean_of(1);
    assert!(
        (ranked - expected).abs() / expected < 0.01,
        "glass over water: {ranked} vs Beer–Lambert's {expected} over two meters \
         of glass and half a meter of water"
    );
    let unranked = mean_of(0);
    assert!(
        (unranked - expected).abs() / expected > 0.05,
        "without priority the overlap must not resolve to the same answer, \
         got {unranked} against {expected}"
    );
}

/// Priority past the depth the set holds. Five nested solids are one more
/// membership than there are slots, so the innermost crossing finds none —
/// and a false crossing with nowhere to go is *not* crossed, it is shaded,
/// which walks the scene back toward its no-priority behaviour rather than
/// leaving the set claiming the path is somewhere it is not. What that must
/// produce is not an exact answer but a sound one: finite, non-negative,
/// and one alpha per path.
#[test]
fn priority_nested_past_the_slots_stays_sound() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let glass = Material {
        transmission_color: Vec3::splat(0.8),
        transmission_depth: 1.0,
        ..Material::glass(0.0, 1.5)
    };
    // Concentric, outermost first, each outranking everything inside it —
    // so every interface but the outermost pair is false.
    let solids: Vec<Object> = (1..=5)
        .map(|n| Object {
            mesh: crate::scene::cube(n as f32 * 0.4),
            transform: Mat4::from_translation(Vec3::Z * -6.0),
            material: glass,
            medium: None,
            interior_priority: n,
        })
        .collect();
    let camera = Camera {
        position: Vec3::ZERO,
        look_at: Vec3::NEG_Z,
        up: Vec3::Y,
        vfov_degrees: 20.0,
        lens: None,
    };
    let sky_image = Environment::constant(Vec3::splat(0.75));
    let (size, samples) = (16, 64);
    let scene = Scene::new(&gpu, &solids, camera, &sky_image).expect("scene");
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    assert_all_paths_finished(&sum, samples);
    for chunk in sum.chunks_exact(4) {
        for value in chunk {
            assert!(
                value.is_finite() && *value >= 0.0,
                "five nested priorities must stay sound, got {value}"
            );
        }
    }
}

/// Nesting, to the depth the set holds and then past it. Four concentric
/// boxes are exactly the set's four slots as simultaneous memberships, and
/// their eight crossings are exactly the march's cap: the extinctions
/// still sum to the closed form with both limits saturated. Nine are past
/// both, and what that must produce is not an exact answer but a *sound*
/// one — finite, non-negative, and one alpha per path. No direction is
/// promised out here: a dropped entry loses its shell's fog, but a shell
/// whose exit outran the march is fog to the horizon (the far side of an
/// unresolved membership reads as the medium's outside — `crossMedium`'s
/// documented one-sided failure), so the camera can legitimately read
/// near-black. A NaN, a hang, or a path that never finishes would not be
/// legitimate.
#[test]
fn nested_volumes_sum_at_full_depth_and_stay_sound_past_it() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let sigma = 0.2;
    let shell = |half: f32| Object {
        mesh: crate::scene::cube(half),
        transform: Mat4::from_translation(Vec3::Z * -6.0),
        material: Material::matte(Vec3::ONE, 0.0),
        medium: Some(crate::scene::Medium {
            absorption: Vec3::splat(sigma),
            scattering: Vec3::ZERO,
            anisotropy: 0.0,
            volume: None,
        }),
        interior_priority: 0,
    };
    let (size, samples) = (8, 64);

    // Shells of half-extent 1..=n: the axial ray runs 2 + 4 + … + 2n
    // meters of fog, and every meter of it is inside one shell or more.
    let depth = |shells: u32| f64::from(shells * (shells + 1)); // Σ 2n
    for shells in [3u32, 4] {
        let nested: Vec<Object> = (1..=shells).map(|n| shell(n as f32)).collect();
        let scene = axial_scene(&gpu, &nested, sky);
        let mean = axial_mean(&gpu, &scene, size, samples);
        let expected = f64::from(sky) * (-f64::from(sigma) * depth(shells)).exp();
        assert!(
            (mean - expected).abs() / expected < 0.005,
            "{shells} nested shells: {mean} vs Beer–Lambert's {expected}"
        );
    }

    let deep: Vec<Object> = (1..=9).map(|n| shell(n as f32)).collect();
    let scene = axial_scene(&gpu, &deep, sky);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    assert_all_paths_finished(&sum, samples);
    for chunk in sum.chunks_exact(4) {
        for value in &chunk[..3] {
            assert!(
                value.is_finite() && *value >= 0.0 && *value <= sky * 1.005 * samples as f32,
                "nine nested shells must stay sound and never brighten, got {value}"
            );
        }
    }
}

/// A box of pure scattering standing in a uniform sky is invisible: every
/// path that enters leaves with the radiance it would have had anyway.
/// The volumetric furnace, bounded — it holds the march, the phase
/// function, and the chromatic distance sampling to energy conservation all
/// at once, and it is the test that would catch a boundary crossing that
/// entered a volume without leaving it.
///
/// Deep bounces, not the default eight: at unit optical thickness a path
/// turns several times before it finds its way out, and the truncated tail
/// is energy rather than noise.
#[test]
fn a_scattering_box_in_a_uniform_sky_disappears() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.6;
    let scene = Scene::new(
        &gpu,
        &[Object {
            mesh: crate::scene::cube(1.0),
            transform: Mat4::from_translation(Vec3::Z * -5.0),
            material: Material::matte(Vec3::ONE, 0.0),
            medium: Some(crate::scene::Medium {
                absorption: Vec3::ZERO,
                scattering: Vec3::splat(0.5),
                anisotropy: 0.4,
                volume: None,
            }),
            interior_priority: 0,
        }],
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 30.0,
            lens: None,
        },
        &Environment::constant(Vec3::splat(sky)),
    )
    .expect("scene");

    let (size, samples) = (16, 512);
    let sum = bsdf_only_sum_deep(&gpu, &scene, size, samples, 64);
    assert_all_paths_finished(&sum, samples);
    for (index, chunk) in sum.chunks_exact(4).enumerate() {
        for (channel, total) in chunk.iter().enumerate().take(3) {
            let mean = f64::from(*total) / f64::from(samples);
            assert!(
                (mean - f64::from(sky)).abs() / f64::from(sky) < 0.03,
                "pixel {index}, channel {channel}: {mean} vs the sky's {sky}"
            );
        }
    }
}

/// A channel that does not extinguish reaches the sky through an unbounded
/// medium, at full strength, while the channels that do reach nothing.
///
/// This is the one way a path can survive a global medium without hitting
/// anything, and so the only route that exercises the volume stage's *miss*
/// branch: a channel with `σ_t` = 0 draws an infinite distance, runs past
/// `tmax`, and comes back as pure transmittance. It is also the sharpest
/// test of the mixture density — the surviving channel's sample carries
/// weight 3 (one draw in three picks it), and a density taken from the
/// picked channel alone rather than the mixture would land it at 1.
#[test]
fn a_transparent_channel_reaches_the_environment() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let sky = 0.75;
    let scene = Scene::new_in_medium(
        &gpu,
        // A scene needs an object; this one sits behind the camera, so
        // every ray traced below sees only sky.
        &[Object {
            mesh: crate::scene::icosphere(1),
            transform: Mat4::from_translation(Vec3::Z * 20.0),
            material: Material::matte(Vec3::splat(0.5), 0.0),
            medium: None,
            interior_priority: 0,
        }],
        Camera {
            position: Vec3::ZERO,
            look_at: Vec3::NEG_Z,
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens: None,
        },
        &Environment::constant(Vec3::splat(sky)),
        Some(&crate::scene::Medium {
            // Red passes; green and blue turn back and never escape. It has
            // to be *scattering* that stops them: a medium that only
            // absorbs takes the closed-form branch, where no distance is
            // drawn and there is no mixture density to check.
            absorption: Vec3::ZERO,
            scattering: Vec3::new(0.0, 0.5, 0.5),
            anisotropy: 0.0,
            volume: None,
        }),
    )
    .expect("scene");

    let (size, samples) = (16, 512);
    let sum = bsdf_only_sum(&gpu, &scene, size, samples);
    let count = f64::from(samples * size * size);
    let mean = |channel: usize| {
        sum.chunks_exact(4)
            .map(|pixel| f64::from(pixel[channel]))
            .sum::<f64>()
            / count
    };
    let red = mean(0);
    assert!(
        (red - f64::from(sky)).abs() / f64::from(sky) < 0.02,
        "the transparent channel should read the sky exactly: {red} vs {sky}"
    );
    for channel in 1..3 {
        // Radiance sums non-negative terms, so this is exactly zero: an
        // extinguishing channel's every path turns back forever.
        assert!(
            mean(channel) <= 0.0,
            "an infinitely deep channel must arrive at exactly zero, got {}",
            mean(channel)
        );
    }
    assert_all_paths_finished(&sum, samples);
}

/// Beer–Lambert absorption, pinned per channel: a glass sphere whose
/// interior reaches (0.4, 1, 1) after one radius of travel absorbs
/// red only — the green channel must still close its furnace exactly
/// (absorption-free glass), while red must land well below it but
/// clearly above zero. A sign slip, a wrong distance, or absorption
/// applied to the wrong segment moves one channel and not the other.
#[test]
fn interior_absorption_is_spectral() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::with_max_bounces(&gpu, 16).expect("renderer");
    let mut material = Material::glass(0.2, 1.5);
    material.transmission_color = Vec3::new(0.4, 1.0, 1.0);
    material.transmission_depth = 1.0; // the sphere's radius
    let objects = [Object {
        mesh: crate::scene::icosphere(3),
        transform: Mat4::from_translation(Vec3::Y * 2.0),
        material,
        medium: None,
        interior_priority: 0,
    }];
    let camera = Camera {
        position: Vec3::new(0.0, 2.0, 4.0),
        look_at: Vec3::new(0.0, 2.0, 0.0),
        up: Vec3::Y,
        vfov_degrees: 40.0,
        lens: None,
    };
    let sky = 0.5;
    let scene = Scene::new(
        &gpu,
        &objects,
        camera,
        &Environment::constant(Vec3::splat(sky)),
    )
    .expect("scene");
    let samples = 128;
    let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
    let mean = |channel: usize| {
        sum.chunks_exact(4).map(|chunk| chunk[channel]).sum::<f32>()
            / (32.0 * 32.0 * samples as f32)
    };
    let (red, green) = (mean(0), mean(1));
    assert!(
        (green - sky).abs() / sky < 0.015,
        "the absorption-free channel leaked: {green} vs {sky}"
    );
    assert!(
        red < 0.9 * sky && red > 0.2 * sky,
        "red should be absorbed along interior chords: {red} vs sky {sky}"
    );
}

/// The transmission slots may not reach an opaque surface. At
/// `transmission_weight` 0 the closure skips the glass energy tables and
/// every consumer folds the interface tint away, so neither the interior
/// color nor its depth may move a single bit of the render. What the
/// closure leaves in the skipped scale is deliberately not asserted: at
/// weight 0 any finite stand-in is unobservable, which is exactly why
/// skipping the fetch is bit-identical. Only the folds are testable.
#[test]
fn opaque_shading_ignores_the_transmission_slots() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let render = |color: Vec3, depth: f32| {
        let mut material = Material::glossy(Vec3::new(0.8, 0.5, 0.3), 0.2, 0.35);
        material.transmission_color = color;
        material.transmission_depth = depth;
        let scene = furnace_scene(&gpu, material, Vec3::ZERO, 1.0, None);
        accumulate_sum(&gpu, &renderer, &scene, 32, 16)
    };
    // The tint takes the interior color only at depth 0, so vary both:
    // the depth-0 case is the one that moves `glassTint` off white.
    let tinted = Vec3::new(0.1, 0.7, 0.2);
    let reference = render(Vec3::ONE, 0.0);
    for (color, depth) in [(tinted, 0.0), (tinted, 2.0)] {
        assert_eq!(
            render(color, depth),
            reference,
            "opaque shading moved with transmission_color {color}, depth {depth}"
        );
    }
}

/// Stochastic opacity, per-sample exact: a half-opacity white Lambert
/// plane in the furnace. A camera ray either passes through (the
/// intersect stage's Bernoulli trial) and reads the sky directly, or
/// lands and bounces off albedo 1 — both worth exactly the sky, so
/// *every sample of every pixel* must equal it. Any weighting slipped
/// into the pass-through (or a miscounted alpha) fails loudly.
#[test]
fn stochastic_opacity_is_per_sample_exact() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = furnace_scene(
        &gpu,
        Material::matte(Vec3::ONE, 0.0).with_opacity(0.5),
        Vec3::ZERO,
        1.0,
        None,
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - 0.5).abs() < 1e-3,
                "opacity carried weight: {value} vs 0.5"
            );
        }
    }
}

/// The coat's physical darkening: a coat over a *gray* base traps
/// part of the base's exitance under internal reflection — the "wet
/// look", and it is strong: at coat IOR 1.6 the spec's internal
/// diffuse reflection coefficient is K ≈ 0.65, so a 0.5-albedo
/// Lambertian base darkens by Δ = (1−K)/(1−0.5·K) ≈ 0.52. Turning
/// `coat_darkening` from 0 to 1 must land the render in that
/// neighborhood (the coat's own reflection cushions the ratio above
/// Δ itself). The furnace matrix pins the white-base case, where
/// darkening must vanish; this pins that the factor engages with the
/// spec's magnitude.
#[test]
fn coat_darkening_darkens_a_gray_base() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mean_with = |darkening: f32| {
        let mut material = Material::matte(Vec3::splat(0.5), 0.0).with_coat(1.0, 0.1);
        material.coat_darkening = darkening;
        let scene = furnace_scene(&gpu, material, Vec3::ZERO, 1.0, None);
        let samples = 32;
        let sum = accumulate_sum(&gpu, &renderer, &scene, 16, samples);
        sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (16.0 * 16.0 * samples as f32)
    };
    let (off, on) = (mean_with(0.0), mean_with(1.0));
    let ratio = on / off;
    assert!(
        (0.45..0.75).contains(&ratio),
        "darkening should land near the spec's Δ ≈ 0.52 for this base: \
             {on} vs {off} (ratio {ratio})"
    );
}

/// A scratch directory for a texture test's generated fixtures.
fn fixture_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cenote-render-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("fixture dir");
    dir
}

/// The furnace, prepped from a description — the only route to
/// texture references. Same geometry and framing as [`furnace_scene`]
/// (one big plane, the camera just above looking obliquely down)
/// with the plane carrying a unit UV parameterization (u toward +x,
/// v toward +z), under a constant sky written as a 2×2 EXR into
/// `dir`. `material` is the plane's patch, named "surface";
/// `extra_ops` appends lights or overrides.
fn textured_furnace_scene(
    gpu: &Context,
    dir: &std::path::Path,
    material: crate::scene::changeset::MaterialPatch,
    sky: f32,
    extra_ops: Vec<crate::scene::changeset::Op>,
) -> Scene {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, MeshPatch, Op, SettingsPatch,
    };
    use crate::scene::description::{MeshSource, SceneDescription};

    // Named by value: the test-only environment cache keys by path,
    // so one path must never hold two different skies.
    let sky_path = dir.join(format!("sky-{sky}.exr"));
    crate::output::write_exr(&sky_path, 2, 2, &[sky; 16]).expect("sky EXR");
    let mut ops = vec![
        Op::Settings(SettingsPatch::new("main")),
        Op::Camera(CameraPatch {
            position: Some([0.0, 1.0, 0.0]),
            look_at: Some([0.0, 0.0, -1.0]),
            vfov_degrees: Some(40.0),
            ..CameraPatch::new("main")
        }),
        Op::Environment(EnvironmentPatch {
            path: Some(Some(sky_path)),
            ..EnvironmentPatch::new("sky")
        }),
        Op::Mesh(MeshPatch {
            source: Some(MeshSource::Inline {
                positions: vec![
                    [-5.0, 0.0, -5.0],
                    [-5.0, 0.0, 5.0],
                    [5.0, 0.0, 5.0],
                    [5.0, 0.0, -5.0],
                ],
                normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
                uvs: Some(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
                triangles: vec![[0, 1, 2], [0, 2, 3]],
            }),
            ..MeshPatch::new("plane")
        }),
        Op::Material(Box::new(material)),
        Op::Instance(InstancePatch {
            mesh: Some("plane".into()),
            material: Some("surface".into()),
            ..InstancePatch::new("surface")
        }),
    ];
    ops.extend(extra_ops);
    let mut description = SceneDescription::new();
    description.apply(&ChangeSet { ops }).expect("valid scene");
    Scene::prep(gpu, &mut description).expect("prep")
}

/// The furnace through the whole texture pipeline. A white
/// `base_color` *map* on a Lambert base must keep every sample at
/// exactly the sky: BC7 encodes flat white losslessly, the sampler's
/// sRGB decode maps 255 to exactly 1, and the in-shader IDT maps
/// white to white — so sampling, decode, and working-space conversion
/// collectively neither gain nor lose energy. The glossy variant
/// reads `specular_roughness` from a mid-gray BC4 map over a white
/// base and pins the mean: the energy-compensation machinery must
/// hold under sampled parameters exactly as under constants.
#[test]
fn the_textured_furnace_closes() {
    use crate::scene::changeset::MaterialPatch;
    use crate::scene::description::{Texturable, TextureRef};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("furnace");
    let white = dir.join("white.png");
    crate::texture::write_png(&white, 8, 8, &[255u8; 8 * 8 * 4]);
    let scene = textured_furnace_scene(
        &gpu,
        &dir,
        MaterialPatch {
            base_color: Some(Texturable::Texture(TextureRef {
                path: white,
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })),
            specular_weight: Some(0.0),
            ..MaterialPatch::new("surface")
        },
        0.5,
        vec![],
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - 0.5).abs() < 2e-3,
                "textured albedo leaked energy: {value} vs 0.5"
            );
        }
    }

    let gray = dir.join("gray.png");
    let texel = [128u8, 128, 128, 255];
    crate::texture::write_png(&gray, 8, 8, &texel.repeat(8 * 8));
    let scene = textured_furnace_scene(
        &gpu,
        &dir,
        MaterialPatch {
            base_color: Some(Texturable::Constant([1.0; 3])),
            specular_roughness: Some(Texturable::Texture(TextureRef {
                path: gray,
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })),
            ..MaterialPatch::new("surface")
        },
        0.5,
        vec![],
    );
    let renderer = Renderer::new(&gpu).expect("renderer");
    let samples = 64;
    let sum = accumulate_sum(&gpu, &renderer, &scene, 32, samples);
    let mean =
        sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * samples as f32);
    assert!(
        (mean - 0.5).abs() / 0.5 < 0.015,
        "mapped-roughness furnace leaked: mean {mean} vs 0.5"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The environment tint scales its lighting linearly, applied at
/// sampling: a half-gray tint over the unit sky must land every pixel at
/// half the untinted render. The draws are identical — the scene has no
/// other lights, so the environment's selection probability pins to 1
/// under either tint — leaving pure radiance scaling, so the tolerance
/// is float noise (via the tint's trip through `ACEScg`), not variance.
#[test]
fn the_environment_tint_scales_the_sky() {
    use crate::scene::changeset::{EnvironmentPatch, MaterialPatch, Op};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("env-tint");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let sum_with = |tint: Option<[f32; 3]>| {
        let ops = tint
            .map(|tint| {
                vec![Op::Environment(EnvironmentPatch {
                    tint: Some(tint),
                    ..EnvironmentPatch::new("sky")
                })]
            })
            .unwrap_or_default();
        let scene = textured_furnace_scene(
            &gpu,
            &dir,
            MaterialPatch {
                specular_weight: Some(0.0),
                ..MaterialPatch::new("surface")
            },
            1.0,
            ops,
        );
        accumulate_sum(&gpu, &renderer, &scene, 16, 4)
    };
    let full = sum_with(None);
    let half = sum_with(Some([0.5; 3]));
    for (halved, unit) in half.chunks_exact(4).zip(full.chunks_exact(4)) {
        for (h, f) in halved[..3].iter().zip(&unit[..3]) {
            assert!(
                (h - 0.5 * f).abs() <= 1e-3 * f.max(1.0),
                "a half-gray tint should halve the render: {h} vs {f}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Turning the environment turns the sky, in the authored direction: the
/// image's left and right columns wear different colors, the camera looks
/// down −Z (where the image's horizontal center faces), and a ±90° yaw
/// about +Y decides which column faces it — +90° carries the environment's
/// +X (the right column) to world −Z, −90° the left. An instanceless
/// scene, so the frame center reads the sky directly.
#[test]
fn the_environment_placement_turns_the_sky() {
    use crate::scene::changeset::{CameraPatch, ChangeSet, EnvironmentPatch, Op, SettingsPatch};
    use crate::scene::description::{SceneDescription, Transform};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("env-turn");
    let sky = dir.join("columns.exr");
    #[rustfmt::skip]
    crate::output::write_exr(&sky, 2, 2, &[
        4.0, 0.0, 0.0, 1.0,    0.0, 4.0, 0.0, 1.0,
        4.0, 0.0, 0.0, 1.0,    0.0, 4.0, 0.0, 1.0,
    ]).expect("column sky");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let center = |yaw: f32| {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch::new("main")),
                    Op::Environment(EnvironmentPatch {
                        path: Some(Some(sky.clone())),
                        transform: Some(Transform::Trs {
                            translate: [0.0; 3],
                            rotate_degrees: [0.0, yaw, 0.0],
                            scale: [1.0; 3],
                        }),
                        ..EnvironmentPatch::new("sky")
                    }),
                ],
            })
            .expect("valid scene");
        let scene = Scene::prep(&gpu, &mut description).expect("prep");
        let sum = accumulate_sum(&gpu, &renderer, &scene, 16, 1);
        let probe = pixel(&sum, 16, 8, 8);
        (probe[0], probe[1])
    };
    // Channel dominance, not purity: the columns author pure Rec.709
    // primaries, but the trip into ACEScg leaves Rec.709 green with a
    // substantial red component (and vice versa) — the two yaws still
    // land opposite orderings.
    let (red, green) = center(90.0);
    assert!(
        green > 2.0 * red.max(1e-3),
        "+90° yaw should face the camera at the green column: r {red} g {green}"
    );
    let (red, green) = center(-90.0);
    assert!(
        red > 2.0 * green.max(1e-3),
        "-90° yaw should face the camera at the red column: r {red} g {green}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// One emissive probe pins four properties at once: UV orientation (u
/// right, v down the image), texel addressing, the sampler's hardware
/// sRGB decode, and the in-shader IDT. A quad exactly filling the
/// frame (its half-extent over its distance matches the half-fov)
/// wears a 2×2 emission map — red green / blue white — so each
/// quadrant center lands on a texel center, and a camera hit on an
/// emitter reports its radiance exactly: every probe is an equality
/// against `acescg(srgb⁻¹(texel))`, within the sliver of bilinear mix
/// the camera jitter can reach.
#[test]
fn an_emission_map_pins_uv_orientation_and_the_idt() {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, InstancePatch, MaterialPatch, MeshPatch, Op, SettingsPatch,
    };
    use crate::scene::description::{MeshSource, SceneDescription, Texturable, TextureRef};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("probe");
    let map = dir.join("quadrants.png");
    #[rustfmt::skip]
        crate::texture::write_png(&map, 2, 2, &[
            255, 0, 0, 255,    0, 255, 0, 255,
            0, 0, 255, 255,    255, 255, 255, 255,
        ]);

    let mut description = SceneDescription::new();
    description
        .apply(&ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                Op::Camera(CameraPatch {
                    position: Some([0.0, 0.0, 2.0]),
                    look_at: Some([0.0; 3]),
                    // 2·atan(1/2): the ±1 quad at distance 2 exactly
                    // fills the frame.
                    vfov_degrees: Some(53.130_1),
                    ..CameraPatch::new("main")
                }),
                // No environment: black sky, so the map is the image.
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![
                            [-1.0, -1.0, 0.0],
                            [1.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [-1.0, 1.0, 0.0],
                        ],
                        normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
                        // v runs down the image: (0,0) at the
                        // upper-left corner the camera sees.
                        uvs: Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
                        triangles: vec![[0, 1, 2], [0, 2, 3]],
                    }),
                    ..MeshPatch::new("quad")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.0; 3])),
                    specular_weight: Some(0.0),
                    emission_luminance: Some(1.0),
                    emission_color: Some(Texturable::Texture(TextureRef {
                        path: map,
                        color_space: None,
                        channel: None,
                        scale: None,
                        uv: None,
                    })),
                    ..MaterialPatch::new("emit")
                })),
                Op::Instance(InstancePatch {
                    mesh: Some("quad".into()),
                    material: Some("emit".into()),
                    ..InstancePatch::new("emit")
                }),
            ],
        })
        .expect("valid scene");
    let scene = Scene::prep(&gpu, &mut description).expect("prep");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;
    let pixels = renderer.render(&gpu, &scene, size, size).expect("render");

    let expected = |srgb: Vec3| crate::color::acescg_from_rec709(srgb);
    for (x, y, texel, label) in [
        (16, 16, Vec3::new(1.0, 0.0, 0.0), "top-left red"),
        (48, 16, Vec3::new(0.0, 1.0, 0.0), "top-right green"),
        (16, 48, Vec3::new(0.0, 0.0, 1.0), "bottom-left blue"),
        (48, 48, Vec3::ONE, "bottom-right white"),
    ] {
        let probe = pixel(&pixels, size, x, y);
        let want = expected(texel);
        for (channel, (got, expect)) in probe[..3].iter().zip(want.to_array()).enumerate() {
            assert!(
                (got - expect).abs() < 0.06,
                "{label}, channel {channel}: {got} vs {expect}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The emission-map probe again, now through a sample-time remap: a
/// half-texel offset on both axes rotates the quadrant colors around
/// the image (the sampler wraps), and a value scale of 0.5 halves every
/// probe. Pins the storage-space convention — offset v pushes the
/// sampled window *down* the image — the wrap addressing tiling scales
/// depend on, and the multiplier's place on linear values before the
/// IDT.
#[test]
fn a_uv_remap_shifts_the_map_and_the_value_scale_multiplies() {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, InstancePatch, MaterialPatch, MeshPatch, Op, SettingsPatch,
    };
    use crate::scene::description::{
        MeshSource, SceneDescription, Texturable, TextureRef, UvTransform,
    };

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("remap");
    let map = dir.join("quadrants.png");
    #[rustfmt::skip]
        crate::texture::write_png(&map, 2, 2, &[
            255, 0, 0, 255,    0, 255, 0, 255,
            0, 0, 255, 255,    255, 255, 255, 255,
        ]);

    let mut description = SceneDescription::new();
    description
        .apply(&ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                Op::Camera(CameraPatch {
                    position: Some([0.0, 0.0, 2.0]),
                    look_at: Some([0.0; 3]),
                    vfov_degrees: Some(53.130_1),
                    ..CameraPatch::new("main")
                }),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![
                            [-1.0, -1.0, 0.0],
                            [1.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [-1.0, 1.0, 0.0],
                        ],
                        normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
                        uvs: Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
                        triangles: vec![[0, 1, 2], [0, 2, 3]],
                    }),
                    ..MeshPatch::new("quad")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.0; 3])),
                    specular_weight: Some(0.0),
                    emission_luminance: Some(1.0),
                    emission_color: Some(Texturable::Texture(TextureRef {
                        path: map,
                        color_space: None,
                        channel: None,
                        scale: Some(0.5),
                        uv: Some(UvTransform {
                            scale: [1.0; 2],
                            offset: [0.5, 0.5],
                        }),
                    })),
                    ..MaterialPatch::new("emit")
                })),
                Op::Instance(InstancePatch {
                    mesh: Some("quad".into()),
                    material: Some("emit".into()),
                    ..InstancePatch::new("emit")
                }),
            ],
        })
        .expect("valid scene");
    let scene = Scene::prep(&gpu, &mut description).expect("prep");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;
    let pixels = renderer.render(&gpu, &scene, size, size).expect("render");

    // Each screen quadrant's remapped UV lands on the texel center half
    // a period away on both axes: top-left reads the map's bottom-right,
    // and every probe wraps to the diagonally opposite quadrant.
    let expected = |srgb: Vec3| crate::color::acescg_from_rec709(srgb) * 0.5;
    for (x, y, texel, label) in [
        (16, 16, Vec3::ONE, "top-left → wrapped white"),
        (48, 16, Vec3::new(0.0, 0.0, 1.0), "top-right → wrapped blue"),
        (16, 48, Vec3::new(0.0, 1.0, 0.0), "bottom-left → wrapped green"),
        (48, 48, Vec3::new(1.0, 0.0, 0.0), "bottom-right → wrapped red"),
    ] {
        let probe = pixel(&pixels, size, x, y);
        let want = expected(texel);
        for (channel, (got, expect)) in probe[..3].iter().zip(want.to_array()).enumerate() {
            assert!(
                (got - expect).abs() < 0.06,
                "{label}, channel {channel}: {got} vs {expect}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Normal maps, both halves. Energy: a flat map (128, 128, 255) may
/// tilt shading by at most BC5's half-quantum, so the white Lambert
/// furnace's mean must stay at the sky. Direction: under a distant
/// light at 45°, a map tilted *toward* the light must render clearly
/// brighter than the same map tilted away — once along +u against a
/// light from +x (pinning the tangent's sign) and once along +v
/// against a light from +z (pinning the bitangent's).
#[test]
fn normal_maps_tilt_shading_and_keep_energy() {
    use crate::scene::changeset::{LightPatch, MaterialPatch, Op};
    use crate::scene::description::{Light, Texturable, TextureRef};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("normals");
    let flat = dir.join("flat.png");
    crate::texture::write_png(&flat, 8, 8, &[128u8, 128, 255, 255].repeat(8 * 8));
    let scene = textured_furnace_scene(
        &gpu,
        &dir,
        MaterialPatch {
            base_color: Some(Texturable::Constant([1.0; 3])),
            specular_weight: Some(0.0),
            geometry_normal: Some(Some(TextureRef {
                path: flat,
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })),
            ..MaterialPatch::new("surface")
        },
        0.5,
        vec![],
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    let mean = sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (32.0 * 32.0 * 4.0);
    assert!(
        (mean - 0.5).abs() < 1e-3,
        "a flat normal map moved the furnace: mean {mean} vs 0.5"
    );

    // ~30° tilts along ±u and ±v (192 ↔ +0.5, 64 ↔ −0.5), each pair
    // under a light from the axis the tilt faces.
    let tilted = |name: &str, texel: [u8; 4]| {
        let path = dir.join(name);
        crate::texture::write_png(&path, 8, 8, &texel.repeat(8 * 8));
        path
    };
    let mean_under = |map: std::path::PathBuf, travel: [f32; 3]| {
        let scene = textured_furnace_scene(
            &gpu,
            &dir,
            MaterialPatch {
                base_color: Some(Texturable::Constant([0.5; 3])),
                specular_weight: Some(0.0),
                geometry_normal: Some(Some(TextureRef {
                    path: map,
                    color_space: None,
                    channel: None,
                    scale: None,
                    uv: None,
                })),
                ..MaterialPatch::new("surface")
            },
            0.0, // black sky: the delta light is the only source
            vec![Op::Light(LightPatch {
                light: Some(Light::Distant {
                    direction: travel,
                    irradiance: [3.0; 3],
                }),
                ..LightPatch::new("sun")
            })],
        );
        let renderer = Renderer::new(&gpu).expect("renderer");
        let samples = 16;
        let sum = accumulate_sum(&gpu, &renderer, &scene, 16, samples);
        sum.chunks_exact(4).map(|chunk| chunk[0]).sum::<f32>() / (16.0 * 16.0 * samples as f32)
    };
    for (axis, toward, away, travel) in [
        (
            "u",
            [192u8, 128, 220, 255],
            [64u8, 128, 220, 255],
            [-1.0f32, -1.0, 0.0],
        ),
        (
            "v",
            [128u8, 192, 220, 255],
            [128u8, 64, 220, 255],
            [0.0f32, -1.0, -1.0],
        ),
    ] {
        let bright = mean_under(tilted(&format!("toward-{axis}.png"), toward), travel);
        let dark = mean_under(tilted(&format!("away-{axis}.png"), away), travel);
        assert!(
            bright > 2.0 * dark && dark > 0.0,
            "±{axis} tilt should swing the shading strongly: {bright} vs {dark}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Textured opacity, per-sample exact: a white Lambert plane whose
/// coverage is a 0/255 checker map in the furnace. Every camera ray
/// either passes through a hole (and reads the sky) or lands on
/// albedo 1 (and bounces to the sky) — both worth exactly the sky
/// whatever the map says, so *every sample of every pixel* must equal
/// it. This pins the per-crossing map lookup in the intersect stage's
/// Bernoulli trial: any weighting slipped into a textured
/// pass-through fails loudly.
#[test]
fn textured_opacity_is_per_sample_exact() {
    use crate::scene::changeset::MaterialPatch;
    use crate::scene::description::{Texturable, TextureRef};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("opacity");
    let checker = dir.join("holes.png");
    // 8×8, 4×4 quadrants: two opaque, two fully transparent.
    let rgba: Vec<u8> = (0..64)
        .flat_map(|index| {
            let (x, y) = (index % 8, index / 8);
            let solid = (x < 4) == (y < 4);
            [if solid { 255u8 } else { 0 }, 0, 0, 255]
        })
        .collect();
    crate::texture::write_png(&checker, 8, 8, &rgba);
    let scene = textured_furnace_scene(
        &gpu,
        &dir,
        MaterialPatch {
            base_color: Some(Texturable::Constant([1.0; 3])),
            specular_weight: Some(0.0),
            geometry_opacity: Some(Texturable::Texture(TextureRef {
                path: checker,
                color_space: None,
                channel: None,
                scale: None,
                uv: None,
            })),
            ..MaterialPatch::new("surface")
        },
        0.5,
        vec![],
    );
    let sum = bsdf_only_sum(&gpu, &scene, 32, 4);
    for chunk in sum.chunks_exact(4) {
        for channel in &chunk[..3] {
            let value = channel / 4.0;
            assert!(
                (value - 0.5).abs() < 1e-3,
                "textured opacity carried weight: {value} vs 0.5"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// First global illumination, made mechanical: sky light bounces off
/// a terracotta sphere onto a gray floor, so floor pixels beside the
/// sphere pick up a red cast that the far floor corner doesn't. Both
/// probes are the same neutral material — the difference is purely
/// bounced light. (A dedicated scene, not the demo: the probe
/// positions pin this geometry.)
#[test]
fn indirect_light_bleeds_color() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let objects = [
        Object {
            mesh: crate::scene::icosphere(2),
            transform: Mat4::from_translation(Vec3::Y),
            material: Material::matte(
                crate::color::acescg_from_rec709(Vec3::new(0.7, 0.22, 0.08)),
                0.6,
            ),
            medium: None,
            interior_priority: 0,
        },
        Object {
            mesh: ground_plane(5.0),
            transform: Mat4::IDENTITY,
            material: Material::matte(crate::color::acescg_from_rec709(Vec3::splat(0.65)), 0.1),
            medium: None,
            interior_priority: 0,
        },
    ];
    let camera = Camera {
        position: Vec3::new(0.0, 1.8, 5.0),
        look_at: Vec3::new(0.0, 1.0, 0.0),
        up: Vec3::Y,
        vfov_degrees: 40.0,
        lens: None,
    };
    let scene =
        Scene::new(&gpu, &objects, camera, &Environment::constant(Vec3::ONE)).expect("scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;
    let sum = accumulate_sum(&gpu, &renderer, &scene, size, 32);

    // Mean red/blue ratio over a 3×3 patch — single accumulated pixels
    // are still noisy at 32 samples.
    let redness = |x: u32, y: u32| {
        let (mut red, mut blue) = (0.0, 0.0);
        for dy in 0..3 {
            for dx in 0..3 {
                let probe = pixel(&sum, size, x + dx, y + dy);
                red += probe[0];
                blue += probe[2];
            }
        }
        red / blue
    };
    // The sphere (image center, radius ≈ 18 px at 64²) meets the floor
    // around y = 50; the corner patch sees almost none of it.
    let near = redness(30, 53);
    let far = redness(2, 60);
    assert!(
        near > far * 1.05,
        "no red bleed beside the sphere: near {near} vs far {far}"
    );
}

/// The AOVs' ground truth, pinned on a quad facing the camera dead-on.
/// A pure-diffuse surface records its guides at the first hit, so the
/// albedo AOV must equal the material's base color *exactly* (prep's
/// `ACEScg` conversion and all — the guide is what the beauty divides
/// by), the normal AOV the quad's +z. Depth is the sharp one: the
/// quad's plane is perpendicular to the camera forward, so every quad
/// pixel must read exactly the camera distance — hit *distance* grows
/// off-axis while the stashed forward cosine shrinks, and only their
/// product is constant, which pins the camera-plane-z convention
/// against a Euclidean-distance regression. Sky pixels: albedo white
/// (OIDN's background convention), no normal, depth +∞.
#[test]
fn aovs_record_albedo_normal_and_depth() {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, InstancePatch, MaterialPatch, MeshPatch, Op, SettingsPatch,
    };
    use crate::scene::description::{MeshSource, SceneDescription, Texturable};

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let mut description = SceneDescription::new();
    description
        .apply(&ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch::new("main")),
                Op::Camera(CameraPatch {
                    position: Some([0.0, 0.0, 2.0]),
                    look_at: Some([0.0; 3]),
                    // 2·atan(1/2): a ±1 quad at distance 2 would fill
                    // the frame; this ±0.6 one leaves sky at the edges.
                    vfov_degrees: Some(53.130_1),
                    ..CameraPatch::new("main")
                }),
                // No environment: black sky — the guides don't care.
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![
                            [-0.6, -0.6, 0.0],
                            [0.6, -0.6, 0.0],
                            [0.6, 0.6, 0.0],
                            [-0.6, 0.6, 0.0],
                        ],
                        normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
                        uvs: None,
                        triangles: vec![[0, 1, 2], [0, 2, 3]],
                    }),
                    ..MeshPatch::new("quad")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Constant([0.7, 0.2, 0.1])),
                    specular_weight: Some(0.0),
                    ..MaterialPatch::new("matte")
                })),
                Op::Instance(InstancePatch {
                    mesh: Some("quad".into()),
                    material: Some("matte".into()),
                    ..InstancePatch::new("quad")
                }),
            ],
        })
        .expect("valid scene");
    let scene = Scene::prep(&gpu, &mut description).expect("prep");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;
    let mut film = Film::new(&gpu, size, size).expect("film");
    for _ in 0..4 {
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
    }
    let averages = film.averages(&gpu).expect("averages");

    let expected_albedo = crate::color::acescg_from_rec709(Vec3::new(0.7, 0.2, 0.1)).to_array();
    // Quad pixels: the center, and one well off-axis (the quad spans
    // ±19 px around center 32) where distance ≠ perpendicular z.
    for (x, y) in [(32, 32), (44, 32), (32, 22)] {
        let index = ((y * size + x) * 4) as usize;
        for (channel, expected) in expected_albedo.iter().enumerate() {
            let got = averages.albedo[index + channel];
            assert!(
                (got - expected).abs() < 1e-5,
                "albedo at ({x},{y}) channel {channel}: {got} vs {expected}"
            );
        }
        let normal = &averages.normal[index..index + 3];
        assert!(
            normal[0].abs() < 1e-5 && normal[1].abs() < 1e-5 && (normal[2] - 1.0).abs() < 1e-5,
            "normal at ({x},{y}): {normal:?} vs +z"
        );
        let depth = averages.depth[(y * size + x) as usize];
        assert!(
            (depth - 2.0).abs() < 1e-3,
            "depth at ({x},{y}): {depth} vs the camera plane's 2.0"
        );
    }

    // A corner pixel sees only sky.
    let corner = ((2 * size + 2) * 4) as usize;
    for channel in 0..3 {
        let albedo = averages.albedo[corner + channel];
        assert!(
            (albedo - 1.0).abs() < 1e-5,
            "sky albedo should be white: {albedo}"
        );
        assert!(
            averages.normal[corner + channel].abs() < 1e-5,
            "sky has no normal"
        );
    }
    let sky_depth = averages.depth[(2 * size + 2) as usize];
    assert!(
        sky_depth.is_infinite() && sky_depth.is_sign_positive(),
        "sky depth should be +inf: {sky_depth}"
    );
}

/// The specular pass-through ramp, pinned by the normal guide. A
/// mirror-smooth metal floor scatters every path through
/// `LOBE_SPECULAR` at the roughness floor (0.035), so only
/// 0.035/0.15 of the guide records the floor's own normal — the rest
/// rides the reflection to the sky, whose albedo is white and whose
/// normal is nothing. The same floor at roughness 0.5 sits past the
/// ramp's end and records fully. Both are per-sample exact: the
/// record fraction is deterministic and the floor normal is exactly
/// +y, so the assertions are equalities, not statistics.
#[test]
fn specular_guides_pass_through_to_what_mirrors_show() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let normal_y = |roughness: f32| -> f32 {
        let scene = furnace_scene(
            &gpu,
            Material::metal(Vec3::ONE, roughness),
            Vec3::ZERO,
            1.0,
            None,
        );
        let size = 16;
        let mut film = Film::new(&gpu, size, size).expect("film");
        for _ in 0..2 {
            renderer
                .accumulate(&gpu, &scene, &mut film)
                .expect("accumulate");
        }
        let averages = film.averages(&gpu).expect("averages");
        let center = ((size / 2 * size + size / 2) * 4) as usize;
        // Sanity alongside: white metal or white sky, the albedo
        // guide reads ~1 either way, and the floor's depth is finite.
        assert!(
            (averages.albedo[center] - 1.0).abs() < 1e-3,
            "white-metal albedo guide should be ~1: {}",
            averages.albedo[center]
        );
        let depth = averages.depth[(size / 2 * size + size / 2) as usize];
        assert!(depth.is_finite() && depth > 0.0, "floor depth: {depth}");
        averages.normal[center + 1]
    };

    let mirror = normal_y(0.0);
    let expected = 0.035 / 0.15; // the roughness floor, into the ramp
    assert!(
        (mirror - expected).abs() < 1e-3,
        "a mirror should record only the ramp floor of its normal: \
             {mirror} vs {expected}"
    );
    let rough = normal_y(0.5);
    assert!(
        (rough - 1.0).abs() < 1e-3,
        "roughness past the ramp should record fully: {rough}"
    );
}

/// The hot-reload swap end to end, minus the file watch: recompile the
/// unmodified kernel set through the runtime `slangc` path, swap it in,
/// and require a pixel-identical frame — same source, same compiler,
/// same flags must mean the same image.
#[test]
fn reloaded_kernels_render_identically() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let mut renderer = Renderer::new(&gpu).expect("renderer");
    let before = renderer.render(&gpu, &scene, 64, 64).expect("render");

    let kernels = Kernels::recompile().expect("recompile");
    renderer.reload(&gpu, &kernels).expect("reload");
    let after = renderer.render(&gpu, &scene, 64, 64).expect("render");

    assert_eq!(before, after);
}

/// Two renders of the same scene must agree bit for bit — the
/// charter's replay guarantee, made mechanical. This is the check that
/// pins the wavefront's determinism rule: queue push order varies from
/// run to run, so any radiance write that isn't pixel-owned (or any
/// atomic accumulation) shows up here as flickering low bits.
#[test]
fn rendering_is_bitwise_deterministic() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let first = renderer.render(&gpu, &scene, 128, 128).expect("render");
    let second = renderer.render(&gpu, &scene, 128, 128).expect("render");
    assert_eq!(first, second);
}

/// The film adds each wave's sample — and consecutive samples genuinely
/// differ now that raygen jitters. Rebuild the expected sums from
/// individually traced samples 0..3: the CPU adds in the same order as
/// the three accumulation dispatches (one `f32` add per wave), so
/// agreement is bitwise.
#[test]
fn accumulation_adds_distinct_samples() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, 64, 64).expect("film");
    for _ in 0..3 {
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
    }
    assert_eq!(film.samples(), 3);

    let sample = |index: u32| -> Vec<f32> {
        let radiance = gpu
            .create_buffer(
                "test.sample",
                64 * 64 * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
            .expect("radiance buffer");
        renderer
            .wavefront
            .trace(&gpu, &scene, &radiance, 64, 64, index)
            .expect("trace");
        download_f32(&gpu, &radiance)
    };
    let (s0, s1, s2) = (sample(0), sample(1), sample(2));
    assert_ne!(s0, s1, "jitter must vary from sample to sample");

    let expected: Vec<f32> = s0
        .iter()
        .zip(&s1)
        .zip(&s2)
        .map(|((a, b), c)| a + b + c)
        .collect();
    assert_eq!(download_f32(&gpu, &film.beauty.sum), expected);

    // The batch readback is those sums divided by the count — the same
    // f32 division on both sides, so agreement is again bitwise.
    let average: Vec<f32> = expected.iter().map(|sum| sum / 3.0).collect();
    assert_eq!(film.beauty_average(&gpu).expect("average"), average);
}

/// After a reset, the next sample overwrites the stale sums — that *is*
/// the clear pass. And a reset restarts the sample sequence at index 0,
/// so the result must be bitwise identical to a fresh single frame.
#[test]
fn reset_restarts_the_accumulation() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, 64, 64).expect("film");
    for _ in 0..2 {
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
    }
    film.reset();
    renderer
        .accumulate(&gpu, &scene, &mut film)
        .expect("accumulate");
    assert_eq!(film.samples(), 1);

    let single = renderer.render(&gpu, &scene, 64, 64).expect("render");
    assert_eq!(download_f32(&gpu, &film.beauty.sum), single);
}

/// A rescaled film is the same renderer on fewer pixels, not a second
/// rendering path: shrunk to 32×32 it must be bit-identical to a film
/// created at 32×32, and taken back up to 64×64 the stale tail the reduced
/// render left behind must not survive into the full-size picture.
#[test]
fn a_rescaled_film_renders_what_a_smaller_one_does() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, 64, 64).expect("film");
    // Fill at full size first, so the tail past the reduced rectangle holds
    // real pixels: a leak is then a visible image, not zeros.
    renderer
        .accumulate(&gpu, &scene, &mut film)
        .expect("accumulate");

    film.rescale(32, 32);
    assert_eq!((film.width(), film.height(), film.samples()), (32, 32, 0));
    let mut native = Film::new(&gpu, 32, 32).expect("film");
    for _ in 0..2 {
        for target in [&mut film, &mut native] {
            renderer
                .accumulate(&gpu, &scene, target)
                .expect("accumulate");
        }
    }
    let rescaled = film.beauty_average(&gpu).expect("average");
    assert_eq!(rescaled.len(), 32 * 32 * 4, "the readback is the picture's");
    assert_eq!(rescaled, native.beauty_average(&gpu).expect("average"));

    // Back up: the first sample after a rescale overwrites every texel of
    // the larger rectangle, so this is a fresh single frame again.
    film.rescale(64, 64);
    renderer
        .accumulate(&gpu, &scene, &mut film)
        .expect("accumulate");
    let single = renderer.render(&gpu, &scene, 64, 64).expect("render");
    assert_eq!(download_f32(&gpu, &film.beauty.sum), single);
}

/// The GPU resolve must land the same average as the host
/// [`Film::average`] readback — same sums, same divisor. GPU division is
/// only correctly rounded to a couple of ULP (Vulkan's precision floor),
/// so the two agree to floating-point noise, not bit for bit; a real bug
/// (wrong divisor, transposed indices) misses by far more than that.
/// This is what lets the viewer and the CLI claim to show the same image.
#[test]
fn resolve_matches_host_average() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, 64, 64).expect("film");
    for _ in 0..3 {
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
    }
    let target = |name: &str, bytes: u64| {
        gpu.create_buffer(
            name,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )
        .expect("average buffer")
    };
    let beauty = target("test.average", 64 * 64 * 16);
    let albedo = target("test.average.albedo", 64 * 64 * 16);
    let normal = target("test.average.normal", 64 * 64 * 16);
    let depth = target("test.average.depth", 64 * 64 * 4);
    let targets = ResolveTargets {
        beauty: &beauty,
        albedo: &albedo,
        normal: &normal,
        depth: &depth,
    };
    renderer.resolve(&gpu, &film, &targets).expect("resolve");
    let host = film.averages(&gpu).expect("host averages");
    for (label, resolved, averaged) in [
        ("beauty", download_f32(&gpu, &beauty), host.beauty),
        ("albedo", download_f32(&gpu, &albedo), host.albedo),
        ("normal", download_f32(&gpu, &normal), host.normal),
        ("depth", download_f32(&gpu, &depth), host.depth),
    ] {
        for (gpu_value, host_value) in resolved.iter().zip(&averaged) {
            // Depth averages can be +inf on both sides — inf − inf is
            // NaN, so bit equality covers what the ULP bound can't.
            if gpu_value.to_bits() == host_value.to_bits() {
                continue;
            }
            assert!(
                (gpu_value - host_value).abs() <= 1e-5 * host_value.abs().max(1.0),
                "{label} resolve diverged from the host average: \
                     {gpu_value} vs {host_value}"
            );
        }
    }
}

/// The accumulation kernel's finite guard: a NaN or Inf in any channel
/// drops that pixel's whole contribution — on the overwrite path and
/// the additive path alike — while clean pixels land untouched.
#[test]
fn non_finite_contributions_are_dropped() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, 4, 1).expect("film");
    let poisoned: [f32; 16] = [
        f32::NAN,
        0.5,
        0.5,
        1.0, // NaN red
        0.5,
        f32::INFINITY,
        0.5,
        1.0, // Inf green
        0.5,
        0.5,
        0.5,
        f32::NEG_INFINITY, // -Inf alpha
        0.25,
        0.5,
        0.75,
        1.0, // clean
    ];
    // Swap in a hand-poisoned sample; the usual writer (the primary
    // kernel) can't produce one.
    film.beauty.sample = gpu
        .upload_buffer(
            "film.sample.poisoned",
            bytemuck::bytes_of(&poisoned),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .expect("upload");

    // Drive the accumulation kernel directly — the same pass the render
    // paths fold into the wave, here submitted alone against a poisoned
    // sample the primary kernel could never produce.
    let overwrite = renderer.accumulate_params(&film);
    gpu.submit_passes(&[renderer.accumulate_pass(&overwrite)])
        .expect("overwrite path");
    let expected_once = [
        0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        0.25, 0.5, 0.75, 1.0,
    ];
    assert_eq!(download_f32(&gpu, &film.beauty.sum), expected_once);

    // The second moment is taken over the *same* guarded samples: the three
    // poisoned pixels drop to 0, and the clean pixel carries luminance², so
    // mean(L²) − mean(L)² can never disagree with the beauty sum on which
    // samples counted.
    let luminance =
        |rgb: [f32; 3]| rgb[0] * 0.272_228_72 + rgb[1] * 0.674_081_74 + rgb[2] * 0.053_689_517;
    let clean_l2 = luminance([0.25, 0.5, 0.75]).powi(2);
    let moment2 = download_f32(&gpu, &film.moment2);
    assert_eq!(&moment2[..3], &[0.0, 0.0, 0.0], "poisoned pixels drop to 0");
    assert!(
        (moment2[3] - clean_l2).abs() < 1e-7,
        "clean pixel second moment: {} vs {clean_l2}",
        moment2[3]
    );

    film.samples = 1;
    let additive = renderer.accumulate_params(&film);
    gpu.submit_passes(&[renderer.accumulate_pass(&additive)])
        .expect("additive path");
    let doubled: Vec<f32> = expected_once.iter().map(|value| 2.0 * value).collect();
    assert_eq!(download_f32(&gpu, &film.beauty.sum), doubled);
    // The additive path doubles the second moment in lockstep with the sum.
    let moment2 = download_f32(&gpu, &film.moment2);
    assert_eq!(&moment2[..3], &[0.0, 0.0, 0.0]);
    assert!((moment2[3] - 2.0 * clean_l2).abs() < 1e-7);
}

/// The variance substrate's headline property: on a static
/// scene the estimator standard error `sqrt(Var / N)` falls as `1/sqrt(N)`,
/// because the per-sample luminance variance `Var` is a fixed property of the
/// pixel that the growing sample count only measures better. Rendering is
/// bitwise deterministic, so this is an exact comparison, not a statistical
/// one: quadrupling the samples must halve the mean standard error over the
/// noisy pixels.
#[test]
fn standard_error_falls_as_one_over_sqrt_samples() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;

    let standard_error = |samples: u32| -> Vec<f32> {
        let mut film = Film::new(&gpu, size, size).expect("film");
        for _ in 0..samples {
            renderer
                .accumulate(&gpu, &scene, &mut film)
                .expect("accumulate");
        }
        let se = film.standard_error(&gpu).expect("standard error");
        assert!(
            se.iter().all(|e| e.is_finite() && *e >= 0.0),
            "standard error must be finite and non-negative"
        );
        se
    };

    let coarse = standard_error(32);
    let fine = standard_error(128); // 4× the samples → half the error

    // Per-pixel error ratios over the pixels carrying real noise at the coarse
    // count (lit, stochastic surfaces; the smooth sky and black background have
    // essentially none). The *median* is the right statistic: it reads the
    // 1/sqrt(N) law off the well-behaved bulk while rejecting the heavy-tailed
    // firefly pixels (the sharp glossy front row) whose rare bright samples
    // make them converge slower than the law — the mean would be dragged down
    // by exactly those outliers.
    let mut ratios: Vec<f32> = coarse
        .iter()
        .zip(&fine)
        .filter(|(c, _)| **c > 1e-4)
        .map(|(c, f)| c / f)
        .collect();
    assert!(
        ratios.len() > 100,
        "expected a meaningful noisy region, got {} pixels",
        ratios.len()
    );
    ratios.sort_by(f32::total_cmp);
    let median = ratios[ratios.len() / 2];
    assert!(
        (median - 2.0).abs() < 0.2,
        "quadrupling samples should halve a typical pixel's standard error: median ratio {median}"
    );
}

/// The global auto-stop metric: the converged tally tracks real
/// per-pixel noise, not the sample count. It is exactly zero below the metric's
/// trust floor (`CONVERGENCE_MIN_SAMPLES`), reads a near-exact image as fully
/// converged, and reads a noisier one at the same sample count as less. And the
/// GPU tally matches, pixel for pixel, an independent host recount from the
/// variance substrate — the kernel computes exactly the documented relative
/// standard-error formula, not an approximation of it.
#[test]
fn converged_fraction_tracks_the_noise() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let renderer = Renderer::new(&gpu).expect("renderer");
    let size = 64;
    let total = (size * size) as f32;
    let luminance =
        |rgb: &[f32]| rgb[0] * 0.272_228_72 + rgb[1] * 0.674_081_74 + rgb[2] * 0.053_689_517;

    // Accumulate `samples` of `scene`, then return the GPU converged fraction —
    // after asserting it equals a host recount over the variance substrate that
    // mirrors the kernel exactly, gate and all: below the trust floor the kernel
    // counts nothing, so the mirror does too.
    let measure = |scene: &Scene, samples: u32| -> f32 {
        let mut film = Film::new(&gpu, size, size).expect("film");
        for _ in 0..samples {
            renderer
                .accumulate(&gpu, scene, &mut film)
                .expect("accumulate");
        }
        let fraction = film.converged_fraction(&gpu).expect("fraction");
        let host = if samples < Renderer::CONVERGENCE_MIN_SAMPLES {
            0
        } else {
            let se = film.standard_error(&gpu).expect("standard error");
            let beauty = download_f32(&gpu, &film.beauty.sum);
            let n = samples as f32;
            se.iter()
                .zip(beauty.chunks_exact(4))
                .filter(|(error, rgba)| {
                    let mean_l = luminance(rgba) / n;
                    **error / mean_l.max(Renderer::NOISE_FLOOR) < Renderer::NOISE_THRESHOLD
                })
                .count()
        };
        assert_eq!(
            (fraction * total).round() as usize,
            host,
            "the GPU tally must match the host recount from the variance substrate"
        );
        fraction
    };

    // Below the trust floor no pixel is counted, even at the last sample before
    // it — the variance estimate isn't trusted yet.
    let demo = Scene::demo(&gpu).expect("demo scene");
    // Genuinely zero — a single converged pixel would read 1/4096 ≈ 2.4e-4.
    assert!(
        measure(&demo, Renderer::CONVERGENCE_MIN_SAMPLES - 1) < f32::EPSILON,
        "no pixel counts below CONVERGENCE_MIN_SAMPLES"
    );

    // A delta-lit white Lambert plane is per-sample near-exact — negligible
    // variance — so once past the floor essentially every pixel reads converged.
    let exact = delta_light_scene(
        &gpu,
        crate::scene::description::Light::Distant {
            direction: [0.0, -1.0, 0.0],
            irradiance: [std::f32::consts::PI; 3],
        },
    );
    let converged = measure(&exact, Renderer::CONVERGENCE_MIN_SAMPLES);
    assert!(
        converged >= 0.99,
        "a near-exact image should read fully converged: {converged}"
    );

    // The demo, genuinely noisy at the same count, converges a strictly smaller
    // fraction — the metric reads per-pixel noise, not the sample clock.
    let noisy = measure(&demo, Renderer::CONVERGENCE_MIN_SAMPLES);
    assert!(
        noisy < converged,
        "the noisier image must converge fewer pixels: {noisy} vs {converged}"
    );
}

/// The runtime auto-stop threshold reaches the kernel: on one noisy
/// film, loosening [`Renderer::set_noise_threshold`] counts strictly more pixels
/// as converged than tightening it. Were the setter ignored, both would render
/// at the baked-in default and read identical — so the strict inequality proves the
/// renderer's live threshold, not the baked-in constant, is what the accumulate
/// kernel measures against (the substrate the CLI `--noise-threshold` and the
/// session's convergence-idle both ride).
#[test]
fn the_noise_threshold_reaches_the_kernel() {
    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let mut renderer = Renderer::new(&gpu).expect("renderer");
    let demo = Scene::demo(&gpu).expect("demo scene");
    let size = 64;

    // Accumulate a fixed, replayed sample sequence, so the only variable across
    // the two runs is the threshold the renderer carries into the kernel.
    let converged_at = |renderer: &Renderer| -> f32 {
        let mut film = Film::new(&gpu, size, size).expect("film");
        for _ in 0..Renderer::CONVERGENCE_MIN_SAMPLES {
            renderer
                .accumulate(&gpu, &demo, &mut film)
                .expect("accumulate");
        }
        film.converged_fraction(&gpu).expect("fraction")
    };

    renderer.set_noise_threshold(0.001);
    let tight = converged_at(&renderer);
    renderer.set_noise_threshold(0.5);
    let loose = converged_at(&renderer);
    assert!(
        loose > tight,
        "a looser threshold must converge strictly more pixels: {loose} vs {tight}"
    );
}

/// A closed box with a unit UV parameterization on every face, flat-shaded
/// so the shading and geometric normals agree and the walk's sidedness
/// guard refuses nothing. Closed because a walk needs an interior: the
/// plane the other textured tests use leaks every walk on its first leg.
fn uv_box(half: f32) -> crate::scene::description::MeshSource {
    let h = half;
    // Each face: four corners counter-clockwise seen from outside, and the
    // outward normal. u runs across the face, v down it, the same storage
    // convention `an_emission_map_pins_uv_orientation_and_the_idt` pins.
    let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
        ([[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], [0.0, 0.0, 1.0]),
        ([[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]], [0.0, 0.0, -1.0]),
        ([[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]], [1.0, 0.0, 0.0]),
        ([[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]], [-1.0, 0.0, 0.0]),
        ([[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]], [0.0, 1.0, 0.0]),
        ([[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]], [0.0, -1.0, 0.0]),
    ];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut triangles = Vec::new();
    for (corners, normal) in faces {
        let base = positions.len() as u32;
        positions.extend_from_slice(&corners);
        normals.extend_from_slice(&[normal; 4]);
        uvs.extend_from_slice(&[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        triangles.push([base, base + 1, base + 2]);
        triangles.push([base, base + 2, base + 3]);
    }
    crate::scene::description::MeshSource::Inline {
        positions,
        normals: Some(normals),
        uvs: Some(uvs),
        triangles,
    }
}

/// A subsurface albedo map is read at the vertex the walk *entered*
/// through. A box carries a map split down the middle — red on the left
/// half, blue on the right — and each half must render as the constant of
/// its own colour would.
///
/// The split is the test. A flat map against its constant would pass with
/// the UV taken from the exit hit, with u mirrored, or with a fixed texel
/// sampled — every way this can be wrong. Each assertion below also refuses
/// to be satisfied by the *other* half, so a mirrored u fails by the margin
/// it would otherwise have passed by.
///
/// The mean free path is 1/200th of the box and the probes sit a quarter of
/// a face from the seam, many diffusion lengths clear of the blend that
/// bilinear filtering and the transport both put there.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "three renders of one scene, and the point is that they differ in exactly \
              one patch field — splitting them apart would hide that"
)]
fn a_subsurface_map_is_read_where_the_walk_entered() {
    use crate::scene::changeset::{
        CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, MaterialPatch, MeshPatch, Op,
        SettingsPatch,
    };
    use crate::scene::description::{SceneDescription, TextureRef, Texturable};

    // Two colours far enough apart that the halves cannot be confused, as
    // 8-bit sRGB — the same bytes the constants below linearize by hand, so
    // the two routes describe one colour and the hardware decode is the
    // only thing between them.
    const LEFT: [u8; 3] = [230, 51, 51];
    const RIGHT: [u8; 3] = [51, 51, 230];
    const SIDE: u32 = 64;
    // Enough that the window means settle well inside the margin below —
    // the walk is the only technique here, so its variance is all there is.
    const SAMPLES: u32 = 256;

    let Some(gpu) = crate::gpu::test_context() else {
        return;
    };
    let dir = fixture_dir("sss-map");
    let sky = 0.5;
    let sky_path = dir.join("sky.exr");
    crate::output::write_exr(&sky_path, 2, 2, &[sky; 16]).expect("sky EXR");

    let srgb_to_linear = |byte: u8| {
        let value = f32::from(byte) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    let linear = |rgb: [u8; 3]| rgb.map(srgb_to_linear);

    let map = dir.join("split.png");
    let mut texels = Vec::with_capacity((SIDE * SIDE * 4) as usize);
    for _ in 0..SIDE {
        for x in 0..SIDE {
            let rgb = if x < SIDE / 2 { LEFT } else { RIGHT };
            texels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    crate::texture::write_png(&map, SIDE, SIDE, &texels);

    // The lobe alone: a black diffuse base and no specular interface, so
    // nothing but the walk reaches the film and the colour it carries is
    // the map's, undiluted.
    let walk_patch = |color: Option<Texturable<[f32; 3]>>| MaterialPatch {
        base_color: Some(Texturable::Constant([0.0; 3])),
        specular_weight: Some(0.0),
        subsurface_weight: Some(Texturable::Constant(1.0)),
        subsurface_color: color,
        subsurface_radius: Some(Texturable::Constant(0.005)),
        subsurface_radius_scale: Some(Texturable::Constant([1.0; 3])),
        ..MaterialPatch::new("skin")
    };
    let render = |color: Option<Texturable<[f32; 3]>>| -> Vec<f32> {
        let mut description = SceneDescription::new();
        description
            .apply(&ChangeSet {
                ops: vec![
                    Op::Settings(SettingsPatch::new("main")),
                    Op::Camera(CameraPatch {
                        position: Some([0.0, 0.0, 2.0]),
                        look_at: Some([0.0; 3]),
                        // 2·atan(0.5/2): the ±0.5 face fills the frame.
                        vfov_degrees: Some(28.072_49),
                        ..CameraPatch::new("main")
                    }),
                    Op::Environment(EnvironmentPatch {
                        path: Some(Some(sky_path.clone())),
                        ..EnvironmentPatch::new("sky")
                    }),
                    Op::Mesh(MeshPatch {
                        source: Some(uv_box(0.5)),
                        ..MeshPatch::new("box")
                    }),
                    Op::Material(Box::new(walk_patch(color))),
                    Op::Instance(InstancePatch {
                        mesh: Some("box".into()),
                        material: Some("skin".into()),
                        ..InstancePatch::new("box")
                    }),
                ],
            })
            .expect("valid scene");
        let scene = Scene::prep(&gpu, &mut description).expect("prep");
        let renderer = Renderer::new(&gpu).expect("renderer");
        accumulate_sum(&gpu, &renderer, &scene, 64, SAMPLES)
    };

    let textured = render(Some(Texturable::Texture(TextureRef {
        path: map,
        color_space: None,
        channel: None,
        scale: None,
        uv: None,
    })));
    let left = render(Some(Texturable::Constant(linear(LEFT))));
    let right = render(Some(Texturable::Constant(linear(RIGHT))));

    // A 16-pixel window centred a quarter of the way across each half.
    let window = |pixels: &[f32], x0: u32| -> Vec3 {
        let mut sum = Vec3::ZERO;
        for y in 24..40 {
            for x in x0..x0 + 16 {
                let probe = pixel(pixels, 64, x, y);
                sum += Vec3::new(probe[0], probe[1], probe[2]);
            }
        }
        sum / (256.0 * SAMPLES as f32)
    };
    for (side, x0, want, other) in [
        ("left", 8, &left, &right),
        ("right", 40, &right, &left),
    ] {
        let got = window(&textured, x0);
        let near = (got - window(want, x0)).length();
        let far = (got - window(other, x0)).length();
        // The margin is the test: `near` alone would be satisfied by any
        // map that happened to be dim, and `far` is what says the halves
        // were told apart rather than merely approximated.
        assert!(
            near < 0.02 && far > 8.0 * near,
            "{side} half: {near} from its own constant, {far} from the other's"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
