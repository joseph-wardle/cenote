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
//! success, keep the last good image on failure. `edit-latency` drives a
//! live session instead of a batch render, timing how long each kind of
//! scene edit takes to reach the screen.

mod latency;

use std::io::Write as _;
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
    /// Time how long each kind of scene edit takes to reach the screen.
    EditLatency(latency::LatencyArgs),
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

    /// Re-render whenever a shader source is edited (hot reload).
    /// Compiles kernels from the source checkout; a broken edit prints
    /// the compiler's diagnostics and keeps the last good image.
    #[arg(long)]
    watch: bool,

    /// Skip the end-of-render statistics — the console summary and the
    /// `.stats.ron` sidecar beside --out. About output, not overhead: the
    /// measuring is --no-gpu-timers' business.
    #[arg(long)]
    no_stats: bool,

    /// Record nothing on the GPU: no query pool, no timestamps, not one
    /// extra command in the submission. Every frame is bracketed by two
    /// stamps and only one in thirty-two is resolved kernel by kernel,
    /// precisely so timing stays under a percent — but this is the A/B that
    /// proves it: render a scene with and without, and compare.
    #[arg(long)]
    no_gpu_timers: bool,

    /// Append one line of per-frame statistics to this file, for plotting a
    /// settle curve. Opt-in: a converging render writes one line per
    /// sample, which is the point and also why it is not on by default.
    #[arg(long)]
    stats_trace: Option<PathBuf>,
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
    // First statement in the process, so time-to-first-ray covers the
    // shader compile, the scene load, and the acceleration-structure build
    // — all the things a person actually waits through.
    cenote::stats::mark_startup();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match Command::parse() {
        Command::Render(args) => render(&args),
        Command::Import(args) => import(&args),
        Command::EditLatency(args) => latency::run(&args),
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
    let (scene, settings, load) = match &args.scene {
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
            let (scene, load) = cenote::scene::Scene::prep_timed(&gpu, &mut description)
                .context("preparing the scene")?;
            (scene, settings, load)
        }
        None => (
            cenote::scene::Scene::demo(&gpu)?,
            cenote::scene::description::Settings::default(),
            cenote::stats::Phases::default(),
        ),
    };
    let width = args.width.unwrap_or(settings.resolution[0]);
    let height = args.height.unwrap_or(settings.resolution[1]);
    let spp = args.spp.unwrap_or(settings.spp);
    let depth = args.depth.unwrap_or(settings.max_bounces);

    let mut renderer = cenote::render::Renderer::with_max_bounces(&gpu, depth)?;
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
        None,
        load,
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
            Some(start),
            // A reload re-preps nothing: the scene is already resident, and
            // the recompile it waited on lands in the breakdown's remainder,
            // which is where an unnamed cost belongs.
            cenote::stats::Phases::default(),
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
///
/// `origin` is where the statistics' time-to-first-ray counts from: `None`
/// for the first render, meaning process start, and the moment the edit
/// landed for a `--watch` re-render. `load` says where the time up to that
/// point went, as far as the caller could see it.
#[expect(
    clippy::too_many_arguments,
    reason = "the batch render's whole configuration, already assembled by the caller; \
              a struct here would only rename the argument list"
)]
fn render_frame(
    gpu: &cenote::gpu::Context,
    scene: &cenote::scene::Scene,
    renderer: &cenote::render::Renderer,
    film: &mut cenote::render::Film,
    spp: u32,
    args: &RenderArgs,
    #[cfg(feature = "denoise")] denoiser: Option<&mut cenote::denoise::Denoiser>,
    origin: Option<Instant>,
    load: cenote::stats::Phases,
) -> anyhow::Result<()> {
    // A hot-reload re-render measures its own startup — what a person
    // waited through is the recompile, not the launch an hour ago.
    let mut recorder =
        origin.map_or_else(cenote::stats::Recorder::new, cenote::stats::Recorder::since);
    recorder.attribute_load(load);
    // `--no-gpu-timers` is the A/B: no pool, no timestamps, a submission
    // byte-identical to a build with no stats code in it at all.
    let mut timer = if args.no_gpu_timers {
        None
    } else {
        gpu.create_pass_timer(cenote::gpu::PassTimer::WAVE_CAPACITY)?
    };
    let mut trace = match &args.stats_trace {
        Some(path) => Some(std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("opening the stats trace {}", path.display()))?,
        )),
        None => None,
    };
    for _ in 0..spp {
        // The timer keeps its own cadence: every frame is bracketed, one in
        // `PassTimer::BREAKDOWN_INTERVAL` is resolved kernel by kernel. So
        // the frame times a benchmark reads are the renderer's, not the
        // instrument's.
        let started = Instant::now();
        let passes = renderer.accumulate_timed(gpu, scene, film, timer.as_mut())?;
        let frame = cenote::stats::Frame {
            cpu: started.elapsed(),
            passes,
            size: (film.width(), film.height()),
            samples: film.samples(),
            // An offline render has one resolution and renders every sample
            // at it.
            preview: false,
        };
        // Traced from this frame, not from the recorder's snapshot: a settle
        // curve wants *this* sample's wall-clock, and the snapshot
        // deliberately holds the last *instrumented* frame so its breakdown
        // and its CPU time stay a matched pair. Reading it here would repeat
        // one frame seven times in eight.
        if let Some(trace) = trace.as_mut() {
            writeln!(trace, "{}", frame.to_ron_line()?)?;
        }
        recorder.record(frame);
        // With --noise-threshold, stop as soon as enough of the image has
        // converged; --spp is the hard cap the loop bound already enforces.
        // Below CONVERGENCE_MIN_SAMPLES the metric is untrusted, so the guard
        // short-circuits the 4-byte readback there.
        if args.noise_threshold.is_some()
            && film.samples() >= cenote::render::Renderer::CONVERGENCE_MIN_SAMPLES
            && film.converged_fraction(gpu)? >= cenote::render::Renderer::CONVERGENCE_TARGET
        {
            break;
        }
    }
    // A buffered write that fails only on drop fails silently; a stats file
    // is worth knowing about.
    if let Some(trace) = trace.as_mut() {
        trace.flush().context("writing the stats trace")?;
    }
    // Memory is read once, at the end: this loop allocates nothing after
    // the first sample, so a sampled reading would be the same number many
    // times over.
    recorder.memory(gpu.memory());
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
    #[cfg(feature = "probes")]
    write_probes(gpu, renderer, film, &args.out)?;
    if !args.no_stats {
        let report = recorder.report(
            gpu.device_summary().to_owned(),
            args.scene
                .as_ref()
                .map_or_else(|| "demo".to_owned(), |path| path.display().to_string()),
        );
        print_report(&report);
        // The sidecar beside the image, so two runs diff as text and a change
        // of work is a readable change rather than a remembered impression.
        let sidecar = args.out.with_extension("stats.ron");
        std::fs::write(&sidecar, report.to_ron()?)
            .with_context(|| format!("writing {}", sidecar.display()))?;
        println!("wrote {}", sidecar.display());
    }
    Ok(())
}

/// Measurement builds only: the volume stage's scatter-event histogram as
/// a `.probes.ron` sidecar beside the image, plus a one-line console read.
/// The mean is per *camera path* — width × height × spp — so runs at
/// different sizes or sample counts still compare.
#[cfg(feature = "probes")]
fn write_probes(
    gpu: &cenote::gpu::Context,
    renderer: &cenote::render::Renderer,
    film: &cenote::render::Film,
    out: &Path,
) -> anyhow::Result<()> {
    #[derive(serde::Serialize)]
    struct ProbeReport {
        /// Volume-stage scatter events binned by the bounce they landed
        /// on, trailing zeros trimmed.
        events_by_bounce: Vec<u32>,
        total_events: u64,
        /// Subsurface walks binned by the scatter count they exited at
        /// (the last bin aggregates everything past it), trailing zeros
        /// trimmed.
        walk_exits_by_events: Vec<u32>,
        /// Walks killed at the walk cap — the gate that must read zero on
        /// production materials.
        walk_cap_kills: u32,
        /// Walks that ran out of geometry: the entry mesh is an open
        /// shell, and the energy left through the hole. A property of the
        /// asset, not the material — the gate a curated mesh must read
        /// zero on.
        walk_leak_deaths: u32,
        /// Subsurface draws the sidedness guard turned away before the
        /// walk began, and walks whose exit it turned away at the
        /// boundary. Both are silent deaths in the image, and both are a
        /// property of the *mesh* — how far its interpolated normals have
        /// drifted from the triangles beneath them — rather than of the
        /// material. Counted apart because they leave the same dark pixel
        /// and only the split says which end was at fault.
        ///
        /// Counts, not energy: the two coincide only where throughput is
        /// 1, which is the white-furnace diagnostic these were built for.
        /// Elsewhere a count bounds the loss without measuring it.
        walk_entry_rejects: u32,
        walk_exit_rejects: u32,
        /// Walks the interior roulette ended — the walk's largest death
        /// mode by far, and the one that makes every figure below
        /// conditional: a roulette death contributes no exit length, so
        /// the histogram and the totals drawn from it describe walks that
        /// reached the boundary, not walks that were started.
        walk_roulette_deaths: u32,
        /// Summed over [`Self::walk_exits_by_events`], and therefore over
        /// exits alone — see [`Self::walk_roulette_deaths`].
        total_walk_events: u64,
        /// Camera paths traced: width × height × spp.
        paths: u64,
        mean_events_per_path: f64,
        mean_walk_events_per_path: f64,
    }
    let bins = renderer.probes(gpu)?;
    let split = cenote::wavefront::PROBE_VOLUME_BINS;
    // The two rejection counters sit above both stages' halves rather
    // than inside either, so they come off first and the halves below
    // keep the bin-for-bin meaning they have always had.
    let walk_entry_rejects = bins[cenote::wavefront::PROBE_ENTRY_REJECT_BIN];
    let walk_exit_rejects = bins[cenote::wavefront::PROBE_EXIT_REJECT_BIN];
    let walk_roulette_deaths = bins[cenote::wavefront::PROBE_ROULETTE_BIN];
    let mut events = bins[..split].to_vec();
    let mut walk_exits = bins[split..cenote::wavefront::PROBE_ENTRY_REJECT_BIN].to_vec();
    // The two exitless deaths sit on top of the exit lengths, cap kills
    // last — popped in that order, so what remains is exits alone.
    let walk_cap_kills = walk_exits.pop().unwrap_or(0);
    let walk_leak_deaths = walk_exits.pop().unwrap_or(0);
    let total_events: u64 = events.iter().map(|&events| u64::from(events)).sum();
    let total_walk_events: u64 = walk_exits
        .iter()
        .enumerate()
        .map(|(length, &walks)| length as u64 * u64::from(walks))
        .sum();
    while events.last() == Some(&0) {
        events.pop();
    }
    while walk_exits.last() == Some(&0) {
        walk_exits.pop();
    }
    let paths =
        u64::from(film.width()) * u64::from(film.height()) * u64::from(film.samples());
    // Precision loss starts past 2^52 events — the u32 bins overflow long
    // before that matters.
    #[expect(clippy::cast_precision_loss, reason = "far below f64's integer range")]
    let report = ProbeReport {
        events_by_bounce: events,
        total_events,
        walk_exits_by_events: walk_exits,
        walk_cap_kills,
        walk_leak_deaths,
        walk_entry_rejects,
        walk_exit_rejects,
        walk_roulette_deaths,
        total_walk_events,
        paths,
        mean_events_per_path: total_events as f64 / paths as f64,
        mean_walk_events_per_path: total_walk_events as f64 / paths as f64,
    };
    let sidecar = out.with_extension("probes.ron");
    std::fs::write(
        &sidecar,
        ron::ser::to_string_pretty(&report, ron::ser::PrettyConfig::default())?,
    )
    .with_context(|| format!("writing {}", sidecar.display()))?;
    println!(
        "wrote {} ({} scatter events, {:.2} per path; {} walk events, {:.2} per path, \
         {} cap kills, {} leaks, {} entry rejects, {} exit rejects, \
         {} roulette deaths)",
        sidecar.display(),
        report.total_events,
        report.mean_events_per_path,
        report.total_walk_events,
        report.mean_walk_events_per_path,
        report.walk_cap_kills,
        report.walk_leak_deaths,
        report.walk_entry_rejects,
        report.walk_exit_rejects,
        report.walk_roulette_deaths
    );
    Ok(())
}

/// The end-of-render console block: the same [`Report`] the sidecar
/// carries, laid out for a person. One struct, two renderings — a number
/// on screen and the same number in the file can never drift.
fn print_report(report: &cenote::stats::Report) {
    let millis = |duration: std::time::Duration| duration.as_secs_f64() * 1000.0;
    println!("\n  {}", report.device);
    println!(
        "  {} — {}×{}, {} spp in {:.2} s sampling ({:.2} s from launch)",
        report.scene,
        report.size.0,
        report.size.1,
        report.samples,
        report.sampling.as_secs_f64(),
        report.wall.as_secs_f64(),
    );
    println!(
        "  frame  {:>8.2} ms mean   {:>8.2} median   {:>8.2} p95",
        millis(report.mean_frame),
        millis(report.smoothed.median),
        millis(report.smoothed.p95),
    );
    match report.bound {
        cenote::stats::Bound::Unknown => println!("  gpu           — (no timestamps)"),
        // Every sampled frame is bracketed, so this is the whole render's
        // device time against the whole render's sampling wall-clock — not
        // a subset scaled up. Dividing by `wall` instead would charge the
        // dispatches for the scene load.
        bound => println!(
            "  gpu    {:>8.2} ms ({:.0}% of sampling, {bound})",
            millis(report.gpu),
            100.0 * report.gpu.as_secs_f64() / report.sampling.as_secs_f64().max(f64::EPSILON),
        ),
    }
    if report.passes.has_breakdown() {
        // Sorted by cost, because the only reason to read this list is to
        // find out what to work on next.
        let mut passes: Vec<_> = report.passes.iter().collect();
        passes.sort_unstable_by_key(|pass| std::cmp::Reverse(pass.gpu));
        println!(
            "\n  kernel breakdown over {} of {} frames ({:.2} ms of the {:.2} ms above)",
            report.breakdown_frames,
            report.frames,
            millis(report.passes.total()),
            millis(report.gpu),
        );
        println!("  kernel                    total       per call   calls");
        for pass in passes {
            println!(
                "  {:<22}{:>8.2} ms  {:>8.3} ms  {:>6}",
                pass.label,
                millis(pass.gpu),
                millis(pass.gpu) / f64::from(pass.calls.max(1)),
                pass.calls,
            );
        }
    }
    let marks = &report.interactivity;
    let mark = |name: &str, value: Option<std::time::Duration>| match value {
        Some(value) => println!("  {name:<22}{:>8.1} ms", millis(value)),
        None => println!("  {name:<22}{:>8}", "—"),
    };
    println!();
    mark("first ray", marks.to_first_ray);
    mark("first pixel", marks.to_first_pixel);
    mark(
        &format!("{} spp", cenote::stats::READABLE_SAMPLES),
        marks.to_readable,
    );

    let memory = &report.peak_memory;
    let mib = |bytes: u64| bytes as f64 / f64::from(1u32 << 20);
    println!("\n  peak memory            {:>8.1} MiB", mib(memory.total()));
    for (name, value) in memory.buckets() {
        println!("    {name:<20}{:>8.1} MiB", mib(value));
    }
    if let Some(budget) = memory.budget {
        println!("    {:<20}{:>8.1} MiB", "device heap", mib(budget));
    }
    println!();
}

fn import(args: &ImportArgs) -> anyhow::Result<()> {
    let out = std::path::absolute(&args.out)?;
    let out_dir = out.parent().context("--out has no parent directory")?;
    // Derived assets (a resampled sky) are named after --out — they
    // belong to the RON that references them, not to the input file
    // (every Bitterli scene is a `scene-v4.pbrt`).
    let stem = out
        .file_stem()
        .context("--out has no file name")?
        .to_string_lossy();
    let imported = cenote_pbrt::import_as(&args.scene, out_dir, &stem)
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
