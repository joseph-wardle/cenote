//! The shared-memory framebuffer layout — the one deliberately
//! platform-specific piece of the transport (POSIX `shm_open` + `mmap`).
//! This module is layout only: the constants and arithmetic both sides map
//! by. The writer lives in `cenote-server`; the readers are the integration
//! test and the C++ delegate's `HdRenderBuffer`.
//!
//! One segment per size, named `/cenote-<pid>-<generation>` and described
//! in-band by `protocol::FbDesc`. Inside:
//!
//! - one 4 KiB **header page** — identification, dimensions, plane
//!   offsets, and the continuous status a strict request/response socket
//!   can't carry (`front_index`, `frame_counter`, `samples`, `converged`,
//!   `rejected_edits`, `epoch`);
//! - **two pixel buffers**, each a beauty plane (RGBA f32, row-major,
//!   linear `Rec.709` — converted server-side) followed by a depth
//!   plane (f32, camera-plane z at the first hit, +∞ where every sample
//!   missed).
//!
//! # The tear protocol
//!
//! No locks, no futexes — the reader can never block the render. The
//! writer strictly alternates buffers and publishes each frame once:
//!
//! 1. fill the back buffer (`1 - front_index`) completely;
//! 2. store `front_index` = that buffer;
//! 3. increment `frame_counter` (release order, so a reader that sees the
//!    new count sees the finished pixels), updating `samples`/`converged`
//!    alongside.
//!
//! The reader: load `frame_counter` (acquire), then `front_index`, copy
//! that buffer out, load `frame_counter` again (acquire). The copy is
//! valid **iff the counter advanced at most 1** across it — one advance
//! means the writer published into the *other* buffer while we copied;
//! two or more means it may have wrapped back into ours, so retry.
//! `rejected_edits` is a plain monotonic count (an idle client that sees
//! it move sends `Ping` to collect the messages); reading it needs no
//! protocol.

/// `b"CNFB"`, read as a little-endian u32 at offset 0 — the first check a
/// mapper makes, before trusting anything else on the page.
pub const MAGIC: u32 = u32::from_le_bytes(*b"CNFB");

/// Bumped on any change to this module's layout. A reader finding a
/// version it doesn't know must unmap and treat the server as
/// incompatible, exactly like a `Hello`/`Welcome` protocol mismatch.
pub const LAYOUT_VERSION: u32 = 2;

/// The header's page — plane data starts here. One page keeps every
/// atomic the two processes share away from pixel cachelines.
pub const HEADER_BYTES: u64 = 4096;

/// Byte offsets of the header fields, from the start of the segment.
/// Every field is little-endian; the u64s are 8-aligned. C++ mirrors
/// these names one for one.
pub mod header {
    /// [`super::MAGIC`], u32.
    pub const MAGIC: u64 = 0;
    /// [`super::LAYOUT_VERSION`], u32.
    pub const LAYOUT_VERSION: u64 = 4;
    /// Frame width in pixels, u32.
    pub const WIDTH: u64 = 8;
    /// Frame height in pixels, u32.
    pub const HEIGHT: u64 = 12;
    /// Which buffer (0 or 1) holds the newest complete frame, u32.
    pub const FRONT_INDEX: u64 = 16;
    /// Samples accumulated into the front frame, u32.
    pub const SAMPLES: u64 = 20;
    /// 1 once accumulation has settled (the delegate's `IsConverged`
    /// read), else 0, u32.
    pub const CONVERGED: u64 = 24;
    /// Monotonic count of rejected edits, u32 — moves without a frame, so
    /// an idle client polls this to know a `Ping` has messages waiting.
    pub const REJECTED_EDITS: u64 = 28;
    /// Monotonic publish count, u64 — the tear protocol's clock.
    pub const FRAME_COUNTER: u64 = 32;
    /// Byte offset of buffer 0's beauty plane, u64.
    pub const BEAUTY_OFFSET_0: u64 = 40;
    /// Byte offset of buffer 0's depth plane, u64.
    pub const DEPTH_OFFSET_0: u64 = 48;
    /// Byte offset of buffer 1's beauty plane, u64.
    pub const BEAUTY_OFFSET_1: u64 = 56;
    /// Byte offset of buffer 1's depth plane, u64.
    pub const DEPTH_OFFSET_1: u64 = 64;
    /// The session epoch the front frame incorporates, u64 —
    /// written with the frame it stamps, so a reader that pairs it with
    /// `CONVERGED` can tell a settled *current* picture from a settled
    /// stale one.
    pub const EPOCH: u64 = 72;
}

/// Bytes of one beauty plane: RGBA f32 per pixel.
#[must_use]
pub fn beauty_bytes(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height) * 16
}

/// Bytes of one depth plane: f32 per pixel.
#[must_use]
pub fn depth_bytes(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height) * 4
}

/// Byte offset of buffer `index`'s beauty plane. The header also carries
/// these (a reader may use either); the writer fills the header *from*
/// this arithmetic, so they cannot disagree.
#[must_use]
pub fn beauty_offset(width: u32, height: u32, index: u32) -> u64 {
    let buffer = beauty_bytes(width, height) + depth_bytes(width, height);
    HEADER_BYTES + u64::from(index) * buffer
}

/// Byte offset of buffer `index`'s depth plane (right after its beauty).
#[must_use]
pub fn depth_offset(width: u32, height: u32, index: u32) -> u64 {
    beauty_offset(width, height, index) + beauty_bytes(width, height)
}

/// Total segment length: the header page plus both buffers.
#[must_use]
pub fn segment_bytes(width: u32, height: u32) -> u64 {
    HEADER_BYTES + 2 * (beauty_bytes(width, height) + depth_bytes(width, height))
}

// Compile-time layout guarantees: every header field fits inside the
// header page, and the u64 fields sit 8-aligned — the alignment the
// atomics need.
const _: () = {
    assert!(header::EPOCH + 8 <= HEADER_BYTES);
    assert!(header::FRAME_COUNTER.is_multiple_of(8));
    assert!(header::BEAUTY_OFFSET_0.is_multiple_of(8));
    assert!(header::EPOCH.is_multiple_of(8));
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The planes tile the segment exactly: header, then buffer 0's
    /// beauty and depth, then buffer 1's, back to back, ending at the
    /// segment length.
    #[test]
    fn the_planes_tile_the_segment() {
        let (w, h) = (1280, 721); // odd height: no accidental alignment
        assert_eq!(beauty_offset(w, h, 0), HEADER_BYTES);
        assert_eq!(depth_offset(w, h, 0), beauty_offset(w, h, 0) + beauty_bytes(w, h));
        assert_eq!(beauty_offset(w, h, 1), depth_offset(w, h, 0) + depth_bytes(w, h));
        assert_eq!(depth_offset(w, h, 1), beauty_offset(w, h, 1) + beauty_bytes(w, h));
        assert_eq!(
            segment_bytes(w, h),
            depth_offset(w, h, 1) + depth_bytes(w, h)
        );
    }
}
