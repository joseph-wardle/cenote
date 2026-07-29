//! Headless command line: `render` accumulates a scene (a `.ron` file, a
//! `.pbrt` file imported on the fly, or the built-in demo) to `--spp`
//! samples and writes the linear averages as one multi-layer EXR —
//! beauty, the denoiser's albedo and normal guides, and first-hit depth.
//! The film and the per-sample estimator are exactly the viewer's, so the
//! beauty layer is the image the viewer converges to. In builds with the
//! `denoise` feature, `--denoise` writes a second EXR of the OIDN-denoised
//! beauty beside it — the raw output is never replaced. `import` converts a
//! pbrt-v4 scene to a `.ron` scene file, printing every fidelity warning
//! the importer raises. `render --watch` stays alive and re-renders on
//! every shader edit: recompile via `slangc`, swap the pipeline on
//! success, keep the last good image on failure.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;
use clap::Parser;

#[derive(Parser)]
#[command(version, about = "Cenote: a GPU path tracer")]
enum Command {
    /// Render a scene to an EXR (the built-in demo when none is named).
    Render(RenderArgs),
    /// Convert a pbrt-v4 scene to a cenote .ron scene file.
    Import(ImportArgs),
}

// The flags are independent orthogonal switches, not a state — a state machine
// would model transitions that don't exist (mirrors the viewer's `Gui`).
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent CLI toggles, not a state to machine"
)]
#[derive(clap::Args)]
struct RenderArgs {
    /// Scene file, `.ron` or `.pbrt`. Omitted renders the built-in demo.
    scene: Option<PathBuf>,

    /// Samples per pixel. With --noise-threshold this is the hard cap; without
    /// it, the exact count rendered. Defaults to the scene's settings (demo: 64).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    spp: Option<u32>,

    /// Stop early once 98% of pixels reach this relative estimator standard
    /// error, instead of always running the full --spp (which stays the hard
    /// cap). A fraction in (0, 1]; 0.01 is a perceptually tight default.
    /// Omitted renders the whole sample budget.
    #[arg(long, value_parser = parse_noise_threshold)]
    noise_threshold: Option<f32>,

    /// Output width in pixels. Defaults to the scene's settings.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    width: Option<u32>,

    /// Output height in pixels. Defaults to the scene's settings.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    height: Option<u32>,

    /// Maximum path length in bounces. Defaults to the scene's settings.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=255))]
    depth: Option<u32>,

    /// Output EXR path.
    #[arg(long, default_value = "render.exr")]
    out: PathBuf,

    /// Also write an OIDN-denoised beauty as a second EXR beside --out
    /// (`shot.exr` → `shot.denoised.exr`). Needs a build with the
    /// `denoise` feature.
    #[arg(long)]
    denoise: bool,

    /// Render with ReSTIR-DI at the primary hit instead of the path tracer's
    /// per-bounce NEE. The two converge to the same image (the unbiasedness
    /// gate); this is the batch half of that comparison.
    #[arg(long)]
    restir: bool,

    /// Drop `ReSTIR`'s spatial-reuse pass, leaving single-frame initial RIS.
    /// Only meaningful with --restir; lets a batch isolate what the
    /// k-neighbour gather buys over the candidates pass alone.
    #[arg(long)]
    no_spatial: bool,

    /// Drop `ReSTIR`'s temporal reuse, leaving each frame's reservoirs
    /// independent. Only meaningful with --restir; the fresh-RNG spatial-only
    /// path the D-085 correctness anchor converges against.
    #[arg(long)]
    no_temporal: bool,

    /// Drop `ReSTIR`'s control-variate shading (`ReSTCV`), shading each
    /// pixel's surviving path alone — the zero-CV degenerate (D-130). Only
    /// meaningful with --restir and spatial reuse on; the layer the
    /// colour-noise fix is measured against.
    #[arg(long)]
    no_cv: bool,

    /// Re-render whenever a shader source is edited (hot reload).
    /// Compiles kernels from the source checkout; a broken edit prints
    /// the compiler's diagnostics and keeps the last good image.
    #[arg(long)]
    watch: bool,
}

#[derive(clap::Args)]
struct ImportArgs {
    /// The pbrt-v4 scene file.
    scene: PathBuf,

    /// Output .ron path. Derived assets (a resampled sky) are written
    /// beside it, and the scene's references are relativized against it.
    #[arg(long)]
    out: PathBuf,
}

/// Parse and validate `--noise-threshold`: a relative standard-error fraction
/// in (0, 1]. Above 1 every pixel is trivially "converged"; zero or negative is
/// nonsense — reject both at the flag rather than let them stop the render at
/// sample one (or never).
fn parse_noise_threshold(raw: &str) -> Result<f32, String> {
    let value: f32 = raw.parse().map_err(|_| format!("`{raw}` is not a number"))?;
    if value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err("must be in (0, 1] — a relative standard-error fraction".to_owned())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match Command::parse() {
        Command::Render(args) => render(&args),
        Command::Import(args) => import(&args),
    }
}

/// Load a scene file as a change-set, logging any import fidelity warnings.
fn load_scene(path: &Path) -> anyhow::Result<cenote::scene::changeset::ChangeSet> {
    let imported =
        cenote_pbrt::load(path).with_context(|| format!("loading scene {}", path.display()))?;
    for warning in &imported.warnings {
        log::warn!("{warning}");
    }
    Ok(imported.set)
}

fn render(args: &RenderArgs) -> anyhow::Result<()> {
    // Fail the flag before the render, not after it.
    #[cfg(not(feature = "denoise"))]
    if args.denoise {
        anyhow::bail!(
            "--denoise needs a build with the denoise feature: \
             cargo run -p cenote-cli --features denoise"
        );
    }
    let gpu = cenote::gpu::Context::new()?;
    // The scene and the settings that fill in unspecified flags: the
    // named file's, or the demo with its schema defaults (which match
    // the flags' historical defaults).
    let (scene, settings) = match &args.scene {
        Some(path) => {
            let set = load_scene(path)?;
            let mut description = cenote::scene::description::SceneDescription::new();
            description.apply(&set).context("scene rejected")?;
            let settings = description
                .settings()
                .values()
                .next()
                .cloned()
                .unwrap_or_default();
            let scene = cenote::scene::Scene::prep(&gpu, &mut description)
                .context("preparing the scene")?;
            (scene, settings)
        }
        None => (
            cenote::scene::Scene::demo(&gpu)?,
            cenote::scene::description::Settings::default(),
        ),
    };
    let width = args.width.unwrap_or(settings.resolution[0]);
    let height = args.height.unwrap_or(settings.resolution[1]);
    let spp = args.spp.unwrap_or(settings.spp);
    let depth = args.depth.unwrap_or(settings.max_bounces);

    let mut renderer = cenote::render::Renderer::with_max_bounces(&gpu, depth)?;
    renderer.set_render_mode(if args.restir {
        cenote::render::RenderMode::Restir
    } else {
        cenote::render::RenderMode::PathTracer
    });
    renderer.set_spatial_reuse(!args.no_spatial);
    renderer.set_temporal_reuse(!args.no_temporal);
    renderer.set_cv_shading(!args.no_cv);
    if let Some(threshold) = args.noise_threshold {
        renderer.set_noise_threshold(threshold);
    }
    let mut film = cenote::render::Film::new(&gpu, width, height)?;
    // One OIDN device for the process — built here, reused every reload,
    // rather than opened and dropped inside each frame.
    #[cfg(feature = "denoise")]
    let mut denoiser = if args.denoise {
        Some(cenote::denoise::Denoiser::new()?)
    } else {
        None
    };
    render_frame(
        &gpu,
        &scene,
        &renderer,
        &mut film,
        spp,
        args,
        #[cfg(feature = "denoise")]
        denoiser.as_mut(),
    )?;
    if !args.watch {
        return Ok(());
    }

    let watcher = cenote::shaders::ShaderWatcher::new()?;
    println!("watching for shader edits — Ctrl-C to stop");
    loop {
        watcher.wait()?;
        let start = Instant::now();
        // Compile and pipeline failures both leave the previous kernels —
        // and the previous image — in place; only render/write failures are
        // fatal.
        let reloaded = cenote::shaders::Kernels::recompile()
            .and_then(|kernels| renderer.reload(&gpu, &kernels));
        if let Err(e) = reloaded {
            eprintln!("{e}\nkeeping the previous kernels");
            continue;
        }
        // A reset replays the same sample sequence, so an unchanged kernel
        // reproduces the previous image bit for bit.
        film.reset();
        render_frame(
            &gpu,
            &scene,
            &renderer,
            &mut film,
            spp,
            args,
            #[cfg(feature = "denoise")]
            denoiser.as_mut(),
        )?;
        println!("reloaded in {} ms", start.elapsed().as_millis());
    }
}

/// Accumulate the film to `spp` samples and write its linear averages as
/// one multi-layer EXR (beauty + albedo/normal guides + depth) — the
/// batch half of the thesis: the same estimator the viewer shows
/// progressively, run to a fixed sample count and written to disk.
/// `--denoise` adds a second EXR of the OIDN-denoised beauty beside it;
/// the raw estimator output is never replaced.
fn render_frame(
    gpu: &cenote::gpu::Context,
    scene: &cenote::scene::Scene,
    renderer: &cenote::render::Renderer,
    film: &mut cenote::render::Film,
    spp: u32,
    args: &RenderArgs,
    #[cfg(feature = "denoise")] denoiser: Option<&mut cenote::denoise::Denoiser>,
) -> anyhow::Result<()> {
    for _ in 0..spp {
        renderer.accumulate(gpu, scene, film)?;
        // With --noise-threshold, stop as soon as enough of the image has
        // converged; --spp is the hard cap the loop bound already enforces.
        // Below CONVERGENCE_MIN_SAMPLES the metric is untrusted (D-094), so the
        // guard short-circuits the 4-byte readback there.
        if args.noise_threshold.is_some()
            && film.samples() >= cenote::render::Renderer::CONVERGENCE_MIN_SAMPLES
            && film.converged_fraction(gpu)? >= cenote::render::Renderer::CONVERGENCE_TARGET
        {
            break;
        }
    }
    let averages = film.averages(gpu)?;
    cenote::output::write_aov_exr(
        &args.out,
        film.width(),
        film.height(),
        &averages.beauty,
        &averages.albedo,
        &averages.normal,
        &averages.depth,
    )?;
    println!(
        "wrote {} ({}×{}, {} spp; layers: beauty, albedo, normal, Z)",
        args.out.display(),
        film.width(),
        film.height(),
        film.samples()
    );
    #[cfg(feature = "denoise")]
    if let Some(denoiser) = denoiser {
        let started = Instant::now();
        let filtered = denoiser.denoise(
            film.width(),
            film.height(),
            cenote::denoise::Quality::High,
            &averages.beauty,
            &averages.albedo,
            &averages.normal,
        )?;
        let out = args.out.with_extension("denoised.exr");
        cenote::output::write_exr(&out, film.width(), film.height(), &filtered)?;
        println!(
            "wrote {} (OIDN high quality, {} ms)",
            out.display(),
            started.elapsed().as_millis()
        );
    }
    Ok(())
}

fn import(args: &ImportArgs) -> anyhow::Result<()> {
    let out = std::path::absolute(&args.out)?;
    let out_dir = out.parent().context("--out has no parent directory")?;
    let imported = cenote_pbrt::import(&args.scene, out_dir)
        .with_context(|| format!("importing {}", args.scene.display()))?;
    for warning in &imported.warnings {
        eprintln!("warning: {warning}");
    }
    // Prove the scene applies — a dangling texture or PLY reference
    // surfaces here, at import, not at first render.
    let mut description = cenote::scene::description::SceneDescription::new();
    description
        .apply(&imported.set)
        .context("the imported scene does not apply")?;

    let mut set = imported.set;
    set.relativize_paths(out_dir);
    std::fs::write(&out, cenote::format::to_ron(&set)?)?;
    println!(
        "wrote {} ({} ops, {} warnings)",
        out.display(),
        set.ops.len(),
        imported.warnings.len()
    );
    Ok(())
}
