//! The corpus regression harness: import every vendored pbrt scene
//! (`tests/scenes/` at the repo root — see its README for provenance),
//! prove the result applies, render it, and FLIP-compare against the
//! checked-in golden. Import and apply run everywhere; the render half
//! skips cleanly without a capable GPU, so plain `cargo test` works in
//! any CI.
//!
//! FLIP is a perceptual metric, so the threshold survives the legitimate
//! floating-point reordering a driver or compiler update can cause. A
//! failure dumps the actual render and a FLIP heatmap into `target/tmp/`
//! for inspection in `tev`. When an image change is *intentional*,
//! regenerate — and eyeball — with:
//!
//! ```sh
//! UPDATE_GOLDENS=1 cargo test -p cenote-pbrt --test corpus
//! ```

use std::path::{Path, PathBuf};

use cenote::gpu::Context;
use cenote::output::{read_exr, write_exr};
use cenote::render::Renderer;
use cenote::scene::Scene;
use cenote::scene::description::SceneDescription;

/// One vendored scene: its directory under `tests/scenes/`, and the
/// golden's resolution — the scene's own aspect at a size small enough
/// to live in the repo forever.
const SCENES: &[(&str, u32, u32)] = &[
    ("cornell-box", 256, 256),
    ("veach-mis", 320, 180),
    ("teapot-full", 320, 180),
];

/// Samples per golden. Enough that the image reads clearly; noise is no
/// problem for the comparison because renders are deterministic — the
/// golden holds the *same* noise.
const SPP: u32 = 64;

/// Mean-FLIP failure threshold, matching the core crate's goldens.
const MAX_MEAN_FLIP: f32 = 0.01;

fn scene_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tests/scenes/{name}/scene-v4.pbrt"))
}

fn import(name: &str) -> (cenote::scene::changeset::ChangeSet, Vec<String>) {
    let generated = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-generated"));
    let imported = cenote_pbrt::import(&scene_path(name), &generated)
        .unwrap_or_else(|error| panic!("{name} fails to import: {error}"));
    (imported.set, imported.warnings)
}

/// The CPU half, run everywhere: every vendored scene imports and the
/// result applies — references resolve, files exist, singletons hold —
/// and the warnings are exactly the ones the corpus README documents.
#[test]
fn every_corpus_scene_imports_and_applies() {
    for &(name, ..) in SCENES {
        let (set, warnings) = import(name);
        let mut description = SceneDescription::new();
        description
            .apply(&set)
            .unwrap_or_else(|error| panic!("{name} fails to apply: {error}"));
        assert!(
            !description.instances().is_empty(),
            "{name} has no instances"
        );
        assert_eq!(description.cameras().len(), 1, "{name} camera");
        for warning in &warnings {
            assert!(
                warning.contains("two-sided")
                    || warning.contains("anisotropic")
                    || warning.contains("MakeNamedMedium")
                    || warning.contains("MediumInterface"),
                "{name} raised an undocumented warning: {warning}"
            );
        }
    }
}

/// The GPU half: import → prep → render → FLIP against the golden.
#[test]
fn corpus_renders_match_goldens() {
    let _ = env_logger::builder().is_test(true).try_init();
    let gpu = match Context::new() {
        Ok(gpu) => gpu,
        Err(err) => {
            eprintln!("skipping: no capable GPU here ({err})");
            return;
        }
    };
    for &(name, width, height) in SCENES {
        let (set, _) = import(name);
        let mut description = SceneDescription::new();
        description.apply(&set).expect("corpus scenes apply");
        let max_bounces = description
            .settings()
            .values()
            .next()
            .expect("one settings")
            .max_bounces;
        let scene = Scene::prep(&gpu, &mut description)
            .unwrap_or_else(|error| panic!("{name} fails to prep: {error}"));
        let renderer = Renderer::with_max_bounces(&gpu, max_bounces).expect("renderer");
        let mut film = cenote::render::Film::new(&gpu, width, height).expect("film");
        for _ in 0..SPP {
            renderer
                .accumulate(&gpu, &scene, &mut film)
                .expect("accumulate");
        }
        let actual = film.beauty_average(&gpu).expect("average");
        compare_with_golden(name, width, height, &actual);
    }
}

/// Compare a render against `tests/golden/{name}.exr` — or, under
/// `UPDATE_GOLDENS=1`, (re)write that golden and pass.
fn compare_with_golden(name: &str, width: u32, height: u32, actual: &[f32]) {
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.exr"));

    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(golden_path.parent().expect("golden path has a parent"))
            .expect("create golden dir");
        write_exr(&golden_path, width, height, actual).expect("write golden");
        eprintln!(
            "updated {} — inspect it in tev before committing",
            golden_path.display()
        );
        return;
    }

    let (golden_width, golden_height, golden) = read_exr(&golden_path).unwrap_or_else(|err| {
        panic!(
            "can't read golden {}: {err}\n\
             if it doesn't exist yet: UPDATE_GOLDENS=1 cargo test -p cenote-pbrt --test corpus",
            golden_path.display()
        )
    });
    assert_eq!(
        (golden_width, golden_height),
        (width, height),
        "golden has stale dimensions — regenerate with UPDATE_GOLDENS=1"
    );

    let error_map = nv_flip::flip(
        flip_image(width, height, &golden),
        flip_image(width, height, actual),
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );
    let mean = nv_flip::FlipPool::from_image(&error_map).mean();
    if mean <= MAX_MEAN_FLIP {
        return;
    }

    let dump_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(dump_dir).expect("create dump dir");
    let actual_path = dump_dir.join(format!("{name}.actual.exr"));
    write_exr(&actual_path, width, height, actual).expect("dump actual");
    let heatmap_path = dump_dir.join(format!("{name}.flip.exr"));
    write_heatmap(&heatmap_path, &error_map);
    panic!(
        "{name} differs from golden {} — mean FLIP {mean:.6} > {MAX_MEAN_FLIP}\n\
         actual render: {}\n\
         FLIP heatmap:  {}\n\
         if the change is intentional: UPDATE_GOLDENS=1 cargo test -p cenote-pbrt --test corpus",
        golden_path.display(),
        actual_path.display(),
        heatmap_path.display(),
    );
}

/// FLIP consumes 8-bit RGB: clamp to [0, 1], quantize, drop alpha. The
/// renders are linear HDR, so this compares them as if displayed without
/// exposure or tonemap — highlights above 1 clip to white on both sides.
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

/// The FLIP error map through the magma LUT as an EXR, for `tev`.
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
