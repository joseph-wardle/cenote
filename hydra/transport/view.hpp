// The shm framebuffer reader: the C++ mirror of the server-side View in
// shm.rs, over the layout in wire/fb.hpp. Maps the segment a FbDesc
// names read-only, validates the header before trusting anything on the
// page, and copies planes out under the tear protocol.
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "wire/protocol.hpp"

namespace cenote::transport {

/// One mapped framebuffer segment. Created by the Client from the FbDesc
/// a Welcome or Resized reply carries, and replaced wholesale on resize.
/// Not thread-safe, like everything in transport/: one caller at a time.
class View {
public:
    /// shm_open + mmap (read-only) + header validation: the magic, the
    /// layout version, and the dimensions must all agree with the
    /// descriptor — drift is caught at map time, never as garbage
    /// pixels. Returns null after a warning when anything disagrees.
    [[nodiscard]] static std::unique_ptr<View> open(const wire::FbDesc& desc);

    ~View();
    View(const View&) = delete;
    View& operator=(const View&) = delete;

    [[nodiscard]] std::uint32_t width() const { return width_; }
    [[nodiscard]] std::uint32_t height() const { return height_; }

    /// Copies the front buffer's beauty plane (width*height RGBA f32,
    /// linear Rec.709) into `dst` under the tear protocol. False when no
    /// frame has been published yet (`dst` untouched), or when the
    /// writer outran every copy attempt — `dst` may then hold a torn
    /// frame, and the caller decides whether that shows for the one
    /// repaint before the next copy replaces it.
    bool copy_beauty(std::span<float> dst) const;

    /// The depth twin of copy_beauty: width*height f32 camera-plane
    /// depth, +inf where every sample missed.
    bool copy_depth(std::span<float> dst) const;

    /// The header's converged flag: whether accumulation into the front
    /// frame has settled at the server's sample cap. False before the
    /// first publish.
    [[nodiscard]] bool converged() const;

    /// The session epoch the front frame incorporates; 0 before
    /// the first publish. Convergence is honest only when this has
    /// reached the epoch the last picture-changing reply carried —
    /// converged() alone can describe a stale picture.
    [[nodiscard]] std::uint64_t epoch() const;

    /// The header's monotonic rejected-edit count. Moves without a frame
    /// — a client that sees it advance sends Ping to collect the
    /// messages.
    [[nodiscard]] std::uint32_t rejected_edits() const;

private:
    View(std::uint8_t* base, std::size_t bytes, std::uint32_t width, std::uint32_t height)
        : base_(base), bytes_(bytes), width_(width), height_(height) {}

    std::uint32_t u32(std::uint64_t offset, std::memory_order order) const;
    std::uint64_t u64(std::uint64_t offset, std::memory_order order) const;
    bool copy_plane(std::span<float> dst, bool beauty) const;

    std::uint8_t* base_;
    std::size_t bytes_;
    std::uint32_t width_;
    std::uint32_t height_;
};

} // namespace cenote::transport
