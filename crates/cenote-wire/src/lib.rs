//! The render server's wire — everything that crosses the process
//! boundary between a scene-graph host (the Hydra delegate) and
//! `cenote-server`, spelled out as explicit types.
//!
//! Three pieces, one per module:
//!
//! - [`scene`] — a **full 1:1 mirror** of the renderer's `Op` and its
//!   seven patches. The wire's contract is exactly "a serialized
//!   `ChangeSet`": total (every edit the scene API can express) and
//!   Hydra-agnostic. The mirror is deliberately a second set of types
//!   rather than a re-export — this crate must never depend on the
//!   renderer — and the wire→`Op` translation in `cenote-server`
//!   exhaustively destructures every struct here, so a field added on
//!   either side is a compile error, not a silent drop.
//! - [`protocol`] — the request/response envelope and its framing: a
//!   u32-LE length prefix and a `MessagePack` payload encoded with
//!   `rmp-serde` **defaults** (positional struct arrays). Strict
//!   request/response: every client message gets exactly one reply and
//!   the server never speaks unprompted, so a C++ client needs no reader
//!   thread.
//! - [`fb`] — the shared-memory framebuffer layout: the header page, the
//!   double-buffered plane offsets, and the lock-free tear protocol. The
//!   pixels never touch the socket.
//!
//! The encoded bytes are the cross-language contract, pinned by the
//! golden corpus in `tests/` (Rust is the authority; the C++ encoder must
//! reproduce the goldens byte for byte). Changing anything serializable
//! here means regenerating the corpus in the same commit —
//! `UPDATE_GOLDENS=1 cargo test -p cenote-wire` — and mirroring the
//! change in `hydra/wire/`.

pub mod fb;
pub mod protocol;
pub mod scene;
