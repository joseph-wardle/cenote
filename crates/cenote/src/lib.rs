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
//! | `scene`     | The scene model — `scene::description` is the typed named-object schema, `scene::changeset` its only edit path — and the prep path that joins it to GPU residency: [`scene::Scene::prep`] builds a description fresh, and its dirty-driven update rebuilds only what an edit touched |
//! | `format`    | The `.ron` scene-file boundary: versioned RON serialization of change-sets |
//! | `material`  | `OpenPBR` surface parameters — the host half of the material schema |
//! | `lights`    | The light list — emissive triangles and delta lights — and its power-proportional alias table, built at prep |
//! | `environment` | The equirect environment light: EXR load and the CDF sampling tables, built at prep |
//! | `color`     | Authored `Rec.709` → `ACEScg` conversion at scene prep |
//! | `tables`    | The closure's baked lookup tables — GGX energy data (regenerable from its own QMC baker) and the vendored LTC sheen fit — embedded, uploaded with the scene's resident buffers |
//! | `wavefront` | The engine core: `SoA` path state, GPU stage queues, indirect dispatch — one [`wavefront::Wavefront::trace`] is one sample per pixel, written pixel-owned so renders are bitwise deterministic |
//! | `render`    | Frame orchestration: one-shot linear frames for the CLI and tests, and the progressive path — [`render::Renderer`] accumulates samples into a [`render::Film`] and resolves the linear average. [`render::Session`] runs that loop on its own thread, publishes frames for a consumer to peek, and carries the edit channel: queued change-sets land at wave boundaries as stop → apply → minimal re-prep → restart. [`render::Tonemap`] is the consumer's downstream view transform (exposure + ACES), which the viewer owns and the CLI skips |
//! | `output`    | Linear EXR write + read (read exists for the golden-image tests and the demo environment) |
//! | `error`     | The crate-wide [`enum@Error`] |
//!
//! The GPU kernels themselves live in `shaders/` (Slang, compiled to SPIR-V
//! by `build.rs`; the embedded/recompiled `Kernels` set is registered in
//! `shaders.rs`). Their stage chain — raygen → intersect →
//! (`shade_miss` | `shade_surface`) → `trace_shadow`, then the `accumulate`,
//! `resolve`, and `tonemap` film kernels — is mapped in [`wavefront`]'s
//! module doc, and each `.slang` file's header states its own job.
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
pub mod environment;
pub mod error;
pub mod format;
pub mod gpu;
pub mod lights;
pub mod material;
pub mod output;
pub mod render;
pub mod scene;
pub mod shaders;
mod tables;
pub mod wavefront;

pub use error::{Error, Result};
