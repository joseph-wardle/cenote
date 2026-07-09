//! Headless batch renderer: bring up the GPU, accumulate `--spp` samples of
//! the demo scene, write the linear average as an EXR. The film and the
//! per-sample estimator are exactly the viewer's, so this writes the image
//! the viewer converges to. With `--watch`, stays alive and re-renders on
//! every shader edit: recompile via `slangc`, swap the pipeline on success,
//! keep the last good image on failure.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
#[command(
    version,
    about = "Cenote headless renderer: render one frame to an EXR"
)]
struct Args {
    /// Output width in pixels.
    #[arg(long, default_value_t = 1280, value_parser = clap::value_parser!(u32).range(1..))]
    width: u32,

    /// Output height in pixels.
    #[arg(long, default_value_t = 720, value_parser = clap::value_parser!(u32).range(1..))]
    height: u32,

    /// Samples per pixel to accumulate.
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..))]
    spp: u32,

    /// Maximum path length, in bounces.
    #[arg(
        long,
        default_value_t = cenote::wavefront::Wavefront::DEFAULT_MAX_BOUNCES,
        value_parser = clap::value_parser!(u32).range(1..),
    )]
    depth: u32,

    /// Output EXR path.
    #[arg(long, default_value = "render.exr")]
    out: PathBuf,

    /// Re-render whenever a shader source is edited (hot reload). Compiles
    /// kernels from the source checkout; a broken edit prints the compiler's
    /// diagnostics and keeps the last good image.
    #[arg(long)]
    watch: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let gpu = cenote::gpu::Context::new()?;
    let scene = cenote::scene::Scene::demo(&gpu)?;
    let mut renderer = cenote::render::Renderer::with_max_bounces(&gpu, args.depth)?;
    let mut film = cenote::render::Film::new(&gpu, args.width, args.height)?;
    render_frame(&gpu, &scene, &renderer, &mut film, &args)?;
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
        render_frame(&gpu, &scene, &renderer, &mut film, &args)?;
        println!("reloaded in {} ms", start.elapsed().as_millis());
    }
}

/// Accumulate the film to `--spp` samples and write its linear average —
/// the batch half of the thesis: the same estimator the viewer shows
/// progressively, run to a fixed sample count and written to disk.
fn render_frame(
    gpu: &cenote::gpu::Context,
    scene: &cenote::scene::Scene,
    renderer: &cenote::render::Renderer,
    film: &mut cenote::render::Film,
    args: &Args,
) -> anyhow::Result<()> {
    for _ in 0..args.spp {
        renderer.accumulate(gpu, scene, film)?;
    }
    let pixels = film.average(gpu)?;
    cenote::output::write_exr(&args.out, args.width, args.height, &pixels)?;
    println!(
        "wrote {} ({}×{}, {} spp)",
        args.out.display(),
        args.width,
        args.height,
        args.spp
    );
    Ok(())
}
