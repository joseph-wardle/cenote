//! Headless batch renderer. Grows a real interface (`--width/--height/--out/--watch`)
//! when there is a renderer behind it (m0-plan §4, step 5).

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let gpu = cenote::gpu::Context::new()?;
    println!("cenote-cli {}", env!("CARGO_PKG_VERSION"));
    println!("device: {}", gpu.device_summary());
    Ok(())
}
