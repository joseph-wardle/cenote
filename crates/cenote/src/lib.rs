//! Cenote — a GPU-first, interactive-progressive path tracer on Vulkan ray tracing.
//!
//! The defining thesis: the interactive preview and the converged final frame are
//! the *same estimator*. No biased preview modes — what you see at one second is an
//! honest prediction of the frame at one hour. See `docs/charter.md` for the full
//! vision and `docs/decisions.md` for why things are the way they are.
//!
//! # Module map
//!
//! Modules land as they gain a caller (build order in `docs/m0-plan.md` §4):
//!
//! | Module    | Role | Status |
//! |-----------|------|--------|
//! | `gpu`     | Unsafe-Vulkan quarantine: device context, buffers, submits, pipelines, acceleration structures. Code outside this module does not touch raw `vk` handles. | context, buffers, submits, pipelines done; accel step 6 |
//! | `shaders` | Embedded SPIR-V registry, `slangc` runtime recompile, hot-reload watching | embedding done; reload lands step 8 |
//! | `scene`   | Procedural test geometry and camera (real scene I/O arrives in M2) | planned |
//! | `render`  | Frame orchestration: dispatch the primary kernel, read pixels back | fill kernel renders; ray query step 7 |
//! | `output`  | EXR writing | done |
//! | `error`   | The crate-wide [`enum@Error`] | done |
//!
//! # Conventions
//!
//! Right-handed, Y-up, camera looks −Z. Distances in meters. Host math uses `glam`;
//! shader code states the matching conventions in its own headers.

pub mod error;
pub mod gpu;
pub mod output;
pub mod render;
pub mod shaders;

pub use error::{Error, Result};
