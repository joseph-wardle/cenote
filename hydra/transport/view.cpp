// transport/'s other deliberately-POSIX file (client.cpp is the first):
// shm_open and mmap live here, mirroring the reader half of the server's
// shm.rs. One pragmatic note carried over from there: the plane copies
// race the writer's stores by design — plain memcpy, the same pragmatism
// every shared-memory seqlock ships — and the counter check discards any
// copy the race could have torn.
#include "view.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "pxr/base/tf/diagnostic.h"

#include "wire/fb.hpp"

PXR_NAMESPACE_USING_DIRECTIVE

namespace cenote::transport {

namespace {

// The header fields are how the two processes' loads and stores are
// ordered; the layout 4- and 8-aligns them, and the mapping starts on a
// page, so atomic_ref's alignment demand always holds.
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);

/// A dimension bound far beyond any real viewport, keeping the layout
/// arithmetic far from u64 overflow no matter what a drifted server
/// claims — the shm cousin of MAX_MESSAGE_BYTES.
constexpr std::uint32_t MAX_DIMENSION = 65536;

} // namespace

std::unique_ptr<View> View::open(const wire::FbDesc& desc) {
    if (desc.width == 0 || desc.height == 0 || desc.width > MAX_DIMENSION ||
        desc.height > MAX_DIMENSION) {
        TF_WARN("the framebuffer descriptor names an implausible %ux%u segment", desc.width,
                desc.height);
        return nullptr;
    }
    // The mapping length comes from our own arithmetic; a descriptor
    // that disagrees is layout drift, refused before a byte is mapped —
    // a plane copy must never run past the end of the mapping.
    if (desc.bytes != wire::fb::segment_bytes(desc.width, desc.height)) {
        TF_WARN("the framebuffer descriptor claims %llu bytes where the layout says %llu",
                static_cast<unsigned long long>(desc.bytes),
                static_cast<unsigned long long>(wire::fb::segment_bytes(desc.width, desc.height)));
        return nullptr;
    }

    const int fd = ::shm_open(desc.shm_name.c_str(), O_RDONLY, 0);
    if (fd == -1) {
        TF_WARN("opening the framebuffer segment \"%s\": %s", desc.shm_name.c_str(),
                std::strerror(errno));
        return nullptr;
    }
    void* base =
        ::mmap(nullptr, static_cast<std::size_t>(desc.bytes), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        TF_WARN("mapping the framebuffer segment \"%s\": %s", desc.shm_name.c_str(),
                std::strerror(errno));
        return nullptr;
    }

    std::unique_ptr<View> view(new View(static_cast<std::uint8_t*>(base),
                                        static_cast<std::size_t>(desc.bytes), desc.width,
                                        desc.height));
    const auto check = [&view](const char* what, std::uint64_t offset, std::uint32_t expected) {
        const std::uint32_t found = view->u32(offset, std::memory_order_relaxed);
        if (found != expected) {
            TF_WARN("shm header %s: expected %u, found %u", what, expected, found);
        }
        return found == expected;
    };
    if (!(check("magic", wire::fb::header::MAGIC, wire::fb::MAGIC) &&
          check("layout version", wire::fb::header::LAYOUT_VERSION, wire::fb::LAYOUT_VERSION) &&
          check("width", wire::fb::header::WIDTH, desc.width) &&
          check("height", wire::fb::header::HEIGHT, desc.height))) {
        return nullptr; // The unique_ptr unmaps on the way out.
    }
    return view;
}

View::~View() { ::munmap(base_, bytes_); }

std::uint32_t View::u32(std::uint64_t offset, std::memory_order order) const {
    return std::atomic_ref<std::uint32_t>(*reinterpret_cast<std::uint32_t*>(base_ + offset))
        .load(order);
}

std::uint64_t View::u64(std::uint64_t offset, std::memory_order order) const {
    return std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(base_ + offset))
        .load(order);
}

bool View::copy_beauty(std::span<float> dst) const { return copy_plane(dst, true); }

bool View::copy_depth(std::span<float> dst) const { return copy_plane(dst, false); }

bool View::converged() const {
    // The writer stores the flag before its release increment of the
    // counter, so acquiring the counter first orders this load behind a
    // real publish; an unpublished (zero-filled) segment reads false.
    if (u64(wire::fb::header::FRAME_COUNTER, std::memory_order_acquire) == 0) {
        return false;
    }
    return u32(wire::fb::header::CONVERGED, std::memory_order_relaxed) != 0;
}

bool View::copy_plane(std::span<float> dst, bool beauty) const {
    const std::uint64_t bytes =
        beauty ? wire::fb::beauty_bytes(width_, height_) : wire::fb::depth_bytes(width_, height_);
    if (dst.size_bytes() != bytes) {
        TF_CODING_ERROR("a %zu-byte destination for a %llu-byte plane", dst.size_bytes(),
                        static_cast<unsigned long long>(bytes));
        return false;
    }
    // The tear protocol, fb.rs's reader steps verbatim. One advance
    // across the copy means the writer published into the *other*
    // buffer; two or more means it may have wrapped back into ours, so
    // retry — a writer fast enough to outrun three copies would have to
    // publish six frames during three memcpys.
    for (int attempt = 0; attempt < 3; ++attempt) {
        const std::uint64_t before =
            u64(wire::fb::header::FRAME_COUNTER, std::memory_order_acquire);
        if (before == 0) {
            return false; // Nothing published yet.
        }
        // Masked so a corrupt index can never push the copy off the mapping.
        const std::uint32_t front =
            u32(wire::fb::header::FRONT_INDEX, std::memory_order_relaxed) & 1U;
        const std::uint64_t offset = beauty ? wire::fb::beauty_offset(width_, height_, front)
                                            : wire::fb::depth_offset(width_, height_, front);
        std::memcpy(dst.data(), base_ + offset, static_cast<std::size_t>(bytes));
        const std::uint64_t after = u64(wire::fb::header::FRAME_COUNTER, std::memory_order_acquire);
        if (after - before <= 1) {
            return true;
        }
    }
    return false;
}

} // namespace cenote::transport
