//! The render server's library half: the pieces of `cenote-server` that
//! are meaningful outside the binary — the shared-memory framebuffer
//! ([`shm`], whose read side the integration test and, later, the C++
//! delegate follow) and the wire→renderer translation ([`translate`],
//! whose exhaustive destructuring is the compile-time half of the D-100
//! drift guard). The binary itself (`main.rs`) is the process contract:
//! one stdout line, one client, strict request/response, EOF is shutdown.

pub mod shm;
pub mod translate;
