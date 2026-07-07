//! Headless batch renderer: bring up the GPU, render one frame, write an EXR
//! (decision D-002 — view it in `tev`, which auto-refreshes on file change).
//! The `--watch` hot-reload loop arrives in m0-plan step 8.

use std::path::PathBuf;

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

    /// Output EXR path.
    #[arg(long, default_value = "render.exr")]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let gpu = cenote::gpu::Context::new()?;
    let scene = cenote::scene::Scene::demo(&gpu)?;
    let pixels = cenote::render::render(&gpu, &scene, args.width, args.height)?;
    cenote::output::write_exr(&args.out, args.width, args.height, &pixels)?;

    println!(
        "wrote {} ({}×{})",
        args.out.display(),
        args.width,
        args.height
    );
    Ok(())
}
