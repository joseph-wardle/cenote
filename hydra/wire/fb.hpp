// The shared-memory framebuffer layout — mirror of cenote-wire's fb.rs:
// the constants and arithmetic both processes map the segment by. Layout
// only, deliberately POSIX-free: the reader that maps lives in
// transport/view.cpp, the writer in the server's shm.rs.
//
// The tear protocol this layout serves (fb.rs's module doc, abridged):
// the writer fills the back buffer, stores front_index, then increments
// frame_counter with release order. The reader loads frame_counter
// (acquire), copies the front planes, and loads it again — the copy is
// valid iff the counter advanced at most 1 across it.
#pragma once

#include <cstdint>

namespace cenote::wire::fb {

/// "CNFB" read as a little-endian u32 at offset 0 — the first check a
/// mapper makes, before trusting anything else on the page.
inline constexpr std::uint32_t MAGIC = std::uint32_t{'C'} | std::uint32_t{'N'} << 8 |
                                       std::uint32_t{'F'} << 16 | std::uint32_t{'B'} << 24;

/// Bumped on any change to the layout. A reader finding a version it
/// does not know must unmap and treat the server as incompatible,
/// exactly like a Hello/Welcome protocol mismatch.
inline constexpr std::uint32_t LAYOUT_VERSION = 2;

/// The header's page — plane data starts here.
inline constexpr std::uint64_t HEADER_BYTES = 4096;

/// Byte offsets of the header fields, from the start of the segment —
/// fb.rs's `header` module, name for name. Every field is little-endian;
/// the u64s are 8-aligned.
namespace header {
/// fb::MAGIC, u32.
inline constexpr std::uint64_t MAGIC = 0;
/// fb::LAYOUT_VERSION, u32.
inline constexpr std::uint64_t LAYOUT_VERSION = 4;
/// Frame width in pixels, u32.
inline constexpr std::uint64_t WIDTH = 8;
/// Frame height in pixels, u32.
inline constexpr std::uint64_t HEIGHT = 12;
/// Which buffer (0 or 1) holds the newest complete frame, u32.
inline constexpr std::uint64_t FRONT_INDEX = 16;
/// Samples accumulated into the front frame, u32.
inline constexpr std::uint64_t SAMPLES = 20;
/// 1 once accumulation has settled, else 0, u32.
inline constexpr std::uint64_t CONVERGED = 24;
/// Monotonic count of rejected edits, u32 — moves without a frame, so an
/// idle client polls this to know a Ping has messages waiting.
inline constexpr std::uint64_t REJECTED_EDITS = 28;
/// Monotonic publish count, u64 — the tear protocol's clock.
inline constexpr std::uint64_t FRAME_COUNTER = 32;
/// Byte offset of buffer 0's beauty plane, u64.
inline constexpr std::uint64_t BEAUTY_OFFSET_0 = 40;
/// Byte offset of buffer 0's depth plane, u64.
inline constexpr std::uint64_t DEPTH_OFFSET_0 = 48;
/// Byte offset of buffer 1's beauty plane, u64.
inline constexpr std::uint64_t BEAUTY_OFFSET_1 = 56;
/// Byte offset of buffer 1's depth plane, u64.
inline constexpr std::uint64_t DEPTH_OFFSET_1 = 64;
/// The session epoch the front frame incorporates, u64 —
/// written with the frame it stamps, so a reader that pairs it with
/// CONVERGED can tell a settled *current* picture from a settled stale
/// one.
inline constexpr std::uint64_t EPOCH = 72;
} // namespace header

/// Bytes of one beauty plane: RGBA f32 per pixel, row-major, linear
/// Rec.709 (converted server-side).
constexpr std::uint64_t beauty_bytes(std::uint32_t width, std::uint32_t height) {
    return std::uint64_t{width} * height * 16;
}

/// Bytes of one depth plane: f32 camera-plane depth per pixel, +inf
/// where every sample missed.
constexpr std::uint64_t depth_bytes(std::uint32_t width, std::uint32_t height) {
    return std::uint64_t{width} * height * 4;
}

/// Byte offset of buffer `index`'s beauty plane. The header also carries
/// these; the writer fills the header *from* this arithmetic, so they
/// cannot disagree.
constexpr std::uint64_t beauty_offset(std::uint32_t width, std::uint32_t height,
                                      std::uint32_t index) {
    return HEADER_BYTES + index * (beauty_bytes(width, height) + depth_bytes(width, height));
}

/// Byte offset of buffer `index`'s depth plane (right after its beauty).
constexpr std::uint64_t depth_offset(std::uint32_t width, std::uint32_t height,
                                     std::uint32_t index) {
    return beauty_offset(width, height, index) + beauty_bytes(width, height);
}

/// Total segment length: the header page plus both buffers.
constexpr std::uint64_t segment_bytes(std::uint32_t width, std::uint32_t height) {
    return HEADER_BYTES + 2 * (beauty_bytes(width, height) + depth_bytes(width, height));
}

// fb.rs's compile-time layout guarantees, verbatim: every header field
// fits inside the header page, and the u64 fields sit 8-aligned — the
// alignment the atomics need.
static_assert(header::EPOCH + 8 <= HEADER_BYTES);
static_assert(header::FRAME_COUNTER % 8 == 0);
static_assert(header::BEAUTY_OFFSET_0 % 8 == 0);
static_assert(header::EPOCH % 8 == 0);

} // namespace cenote::wire::fb
