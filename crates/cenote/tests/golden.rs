//! Golden-image regression tests: render the demo scene and FLIP-compare
//! against the reference EXR checked in at `tests/golden/`.
//!
//! FLIP is a perceptual metric, so the threshold survives the legitimate
//! floating-point reordering a driver or compiler update can cause — where a
//! byte comparison would turn into noise. Tests skip cleanly without a
//! capable GPU, so plain `cargo test` works everywhere, including CI.
//!
//! A failure dumps the actual render and a FLIP heatmap into `target/tmp/`
//! for inspection in `tev`. When an image change is *intentional*,
//! regenerate the goldens — and eyeball them before committing — with:
//!
//! ```sh
//! UPDATE_GOLDENS=1 cargo test -p cenote --test golden
//! ```

use std::path::Path;

use cenote::gpu::Context;
use cenote::output::{read_exr, write_exr};
use cenote::render::{Film, Renderer};
use cenote::scene::Scene;

/// Golden resolution. Small on purpose: enough pixels to pin every feature
/// of the demo image, small enough to live in the repo forever.
const SIZE: u32 = 256;

/// Mean-FLIP failure threshold. Identical images score 0, and FP reordering
/// across driver/compiler updates moves noise and silhouette edges by a
/// pixel at most — far below this. Any visible regression (a wrong sphere
/// shade, a shifted silhouette) lands well above it.
const MAX_MEAN_FLIP: f32 = 0.01;

/// The demo image at 1 spp: the roughness × metalness grid of terracotta
/// spheres across the glossy floor, cross-lit by the Kloofendal HDRI's sun
/// and the warm quad — path traced with MIS, so the golden pins the whole
/// estimator (offsets, interpolated shading normals, `OpenPBR` lobes and
/// their energy-compensation fits across the roughness range, NEE weights,
/// environment tables, sampler) at a fixed seed.
#[test]
fn demo_scene_matches_golden() {
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let actual = renderer.render(&gpu, &scene, SIZE, SIZE).expect("render");
    compare_with_golden("demo", &actual);
}

/// The step-11 checkpoint — batch and viewer agree because they are the
/// same film. This golden is the demo accumulated to 64 spp and read back
/// as its linear average: exactly the image `cenote-cli --spp 64` writes,
/// and exactly the average the viewer is tonemapping 64 redraws after
/// opening. Either path drifting from the other (or both from this
/// reference) fails here.
#[test]
fn accumulated_demo_matches_golden() {
    const SPP: u32 = 64;
    let Some(gpu) = test_context() else {
        return;
    };
    let scene = Scene::demo(&gpu).expect("demo scene");
    let renderer = Renderer::new(&gpu).expect("renderer");
    let mut film = Film::new(&gpu, SIZE, SIZE).expect("film");
    for _ in 0..SPP {
        renderer
            .accumulate(&gpu, &scene, &mut film)
            .expect("accumulate");
    }
    let actual = film.average(&gpu).expect("average");
    compare_with_golden("demo-64spp", &actual);
}

/// GPU gate, mirroring the unit tests' `gpu::test_context`: `None` skips
/// the test with a note on stderr, so GPU-less machines pass cleanly.
fn test_context() -> Option<Context> {
    let _ = env_logger::builder().is_test(true).try_init();
    match Context::new() {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            eprintln!("skipping: no capable GPU here ({err})");
            None
        }
    }
}

/// Compare a fresh `SIZE`² render against `tests/golden/{name}.exr` — or,
/// under `UPDATE_GOLDENS=1`, (re)write that golden and pass.
fn compare_with_golden(name: &str, actual: &[f32]) {
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.exr"));

    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(golden_path.parent().expect("golden path has a parent"))
            .expect("create golden dir");
        write_exr(&golden_path, SIZE, SIZE, actual).expect("write golden");
        eprintln!(
            "updated {} — inspect it in tev before committing",
            golden_path.display()
        );
        return;
    }

    let (width, height, golden) = read_exr(&golden_path).unwrap_or_else(|err| {
        panic!(
            "can't read golden {}: {err}\n\
             if it doesn't exist yet: UPDATE_GOLDENS=1 cargo test -p cenote --test golden",
            golden_path.display()
        )
    });
    assert_eq!(
        (width, height),
        (SIZE, SIZE),
        "golden has stale dimensions — regenerate with UPDATE_GOLDENS=1"
    );

    let error_map = nv_flip::flip(
        flip_image(SIZE, SIZE, &golden),
        flip_image(SIZE, SIZE, actual),
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );
    let mean = nv_flip::FlipPool::from_image(&error_map).mean();
    if mean <= MAX_MEAN_FLIP {
        return;
    }

    // Dump what eyes need to diagnose the difference. The heatmap is the
    // FLIP error map through the magma LUT: black = identical, bright =
    // perceptually different.
    let dump_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(dump_dir).expect("create dump dir");
    let actual_path = dump_dir.join(format!("{name}.actual.exr"));
    write_exr(&actual_path, SIZE, SIZE, actual).expect("dump actual");
    let heatmap_path = dump_dir.join(format!("{name}.flip.exr"));
    write_heatmap(&heatmap_path, &error_map);
    panic!(
        "render differs from golden {} — mean FLIP {mean:.6} > {MAX_MEAN_FLIP}\n\
         actual render: {}\n\
         FLIP heatmap:  {}\n\
         if the change is intentional: UPDATE_GOLDENS=1 cargo test -p cenote --test golden",
        golden_path.display(),
        actual_path.display(),
        heatmap_path.display(),
    );
}

/// FLIP consumes 8-bit RGB: clamp to [0, 1], quantize, drop alpha (always 1
/// here). The renders are linear HDR, so this compares them as if displayed
/// without exposure or tonemap — highlights above 1 clip to white on both
/// sides — and the threshold is far coarser than one 8-bit step.
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

/// Write the FLIP error map through the magma LUT as an EXR, so `tev` shows
/// it right next to the renders. A diagnostic image, not color-managed data.
fn write_heatmap(path: &Path, error_map: &nv_flip::FlipImageFloat) {
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
    write_exr(path, error_map.width(), error_map.height(), &pixels).expect("dump heatmap");
}
