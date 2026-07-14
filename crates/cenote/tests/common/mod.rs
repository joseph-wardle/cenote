//! Shared harness for the GPU integration suites (`golden`, `convergence`): the
//! skip-if-no-GPU gate and the accumulate-into-a-film helper both use, so the
//! two can never drift on what "the image `cenote-cli --spp` writes" means.

use cenote::gpu::Context;
use cenote::render::{Film, Renderer};
use cenote::scene::Scene;

/// GPU gate: `None` skips the calling test with a note on stderr, so GPU-less
/// machines (and CI) pass cleanly.
pub fn test_context() -> Option<Context> {
    let _ = env_logger::builder().is_test(true).try_init();
    match Context::new() {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            eprintln!("skipping: no capable GPU here ({err})");
            None
        }
    }
}

/// Accumulate `spp` samples of `scene` through `renderer` into a fresh
/// `size`×`size` film and return its linear beauty average — exactly the image
/// `cenote-cli --spp spp` writes for that render mode.
pub fn accumulate(
    gpu: &Context,
    renderer: &Renderer,
    scene: &Scene,
    size: u32,
    spp: u32,
) -> Vec<f32> {
    let mut film = Film::new(gpu, size, size).expect("film");
    for _ in 0..spp {
        renderer
            .accumulate(gpu, scene, &mut film)
            .expect("accumulate");
    }
    film.beauty_average(gpu).expect("average")
}
