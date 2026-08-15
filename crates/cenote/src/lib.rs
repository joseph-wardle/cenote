//! Cenote — a GPU-first, interactive-progressive path tracer on Vulkan ray tracing.
//!
//! The defining thesis: the interactive preview and the converged final frame are
//! the *same estimator*. No biased preview modes — what you see at one second is an
//! honest prediction of the frame at one hour.
//!
//! # Module map
//!
//! | Module      | Role |
//! |-------------|------|
//! | `gpu`       | Unsafe-Vulkan quarantine: device context, buffers, submits, pipelines, acceleration structures, window presentation, the viewer's egui overlay pass. Code outside this module does not touch raw `vk` handles. |
//! | `shaders`   | Embedded SPIR-V registry, `slangc` runtime recompile, hot-reload watching |
//! | `scene`     | The whole scene subsystem. `scene::description` is the typed named-object schema and `scene::changeset` its only edit path; the private prep path joins that model to GPU residency, rebuilding only what an edit dirtied. The pipeline it feeds on lives beside it: `scene::source` (PLY meshes, `.hair` grooms, `NanoVDB` grids, texture prep with its DDS cache), `scene::curves` (every piece of curve mathematics), and the GPU-record halves — `scene::material` (`OpenPBR` records), `scene::lights` (the power-proportional alias table), `scene::environment` (equirect CDF tables), `scene::tables` + `scene::blackbody` (the closure's baked LUTs) |
//! | `format`    | The `.ron` scene-file boundary: versioned RON serialization of change-sets |
//! | `color`     | Authored `Rec.709` → `ACEScg` conversion at scene prep |
//! | `wavefront` | The engine core: `SoA` path state, GPU stage queues, indirect dispatch — one [`wavefront::Wavefront::trace`] is one sample per pixel, written pixel-owned so renders are bitwise deterministic |
//! | `render`    | Frame orchestration: [`render::Renderer`] accumulates samples into a [`render::Film`] and resolves the linear average; [`render::Session`] runs that loop on its own thread, publishes frames to peek at, and lands queued change-sets at wave boundaries; [`render::Tonemap`] is the consumer's downstream view transform |
//! | `output`    | Linear EXR write + read (read exists for the golden-image tests and the demo environment) |
//! | `denoise`   | OIDN denoising of the film's averages by host copy, guided by the albedo/normal AOVs — on the renderer's own GPU, found by its Vulkan UUID. [`render::Session`] runs it over the frames it publishes when the settings ask |
//! | `error`     | The crate-wide [`enum@Error`] |
//!
//! The GPU kernels live in `shaders/` (Slang, compiled to SPIR-V by `build.rs`,
//! registered in `shaders.rs`). Their stage chain — raygen → intersect →
//! (`shade_miss` | `shade_surface`) → `trace_shadow`, then the `accumulate`,
//! `resolve`, and `tonemap` film kernels — is mapped in [`wavefront`]'s module
//! doc, and each `.slang` file's header states its own job.
//!
//! # Conventions
//!
//! Right-handed, Y-up, camera looks −Z. Distances in meters. Host math uses `glam`;
//! shader code states the matching conventions in its own headers.
//!
//! Several structs are shared with the kernels byte-for-byte — the geometry
//! record, materials, lights, path state, push-constant blocks. Each Rust
//! definition names its Slang twin (and vice versa), and a layout drift
//! between the two surfaces as a failed golden-image test, never as silent
//! corruption.

pub mod color;
pub mod denoise;
pub mod error;
pub mod format;
pub mod gpu;
pub mod output;
pub mod render;
pub mod scene;
pub mod shaders;
pub mod stats;
pub mod wavefront;

pub use error::{Error, Result};
