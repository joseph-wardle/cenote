//! Cenote — a GPU-first, interactive-progressive path tracer on Vulkan ray tracing.
//!
//! The defining thesis: the interactive preview and the converged final frame are
//! the *same estimator*. No biased preview modes — what you see at one second is an
//! honest prediction of the frame at one hour.
//!
//! # Module map
//!
//! | Module    | Role |
//! |-----------|------|
//! | `gpu`     | Unsafe-Vulkan quarantine: device context, buffers, submits, pipelines, acceleration structures, window presentation, the viewer's egui overlay pass. Code outside this module does not touch raw `vk` handles. |
//! | `shaders` | Embedded SPIR-V registry, `slangc` runtime recompile, hot-reload watching |
//! | `scene`   | Procedural test geometry and camera (real scene I/O arrives in M2) |
//! | `render`  | Frame orchestration: one-shot linear frames for the CLI and tests, and the progressive path — [`render::Renderer`] accumulates samples into a [`render::Film`] and tonemaps (ACES) for display |
//! | `output`  | Linear EXR write + read (read exists for the golden-image tests in `tests/golden.rs`) |
//! | `error`   | The crate-wide [`enum@Error`] |
//!
//! # Conventions
//!
//! Right-handed, Y-up, camera looks −Z. Distances in meters. Host math uses `glam`;
//! shader code states the matching conventions in its own headers.

pub mod error;
pub mod gpu;
pub mod output;
pub mod render;
pub mod scene;
pub mod shaders;

pub use error::{Error, Result};
