// The render buffer: a plain CPU allocation the viewer-side tasks read with
// Map(). Two formats exist for it — f32 RGBA for color, f32 for depth — the
// same planes the shm framebuffer carries, so Map() refreshes the pixels
// with one tear-protocol plane copy. Color crosses untouched; depth is
// remapped from the server's camera-plane meters into the projection's
// [0, 1], the semantic Hydra's depth consumers assume (D-110).
// Allocation doubles as the resize lane: usdview allocates viewport-sized
// buffers, and the server's framebuffer follows (Allocate → Resize → remap).
#pragma once

#include "pxr/base/gf/matrix4d.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/pxr.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace cenote::transport {
class Client;
}

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteRenderBuffer final : public HdRenderBuffer {
public:
    HdCenoteRenderBuffer(SdfPath const& id, cenote::transport::Client* client);

    bool Allocate(GfVec3i const& dimensions, HdFormat format, bool multiSampled) override;

    unsigned int GetWidth() const override;
    unsigned int GetHeight() const override;
    unsigned int GetDepth() const override;
    HdFormat GetFormat() const override;
    bool IsMultiSampled() const override;

    void* Map() override;
    void Unmap() override;
    bool IsMapped() const override;

    void Resolve() override;
    bool IsConverged() const override;

    /// Called by the render pass for the buffers its AOVs bind: exactly
    /// those pull pixels from the server's framebuffer in Map(). Sticky —
    /// everything else (fallback prims, stray allocations) stays plain
    /// local memory forever.
    void MarkBound() { _bound = true; }

    /// The camera projection under which this frame is read — the render
    /// pass refreshes it every execute, so the depth remap always speaks
    /// the current camera's [0, 1]. Bound buffers always have one: the
    /// same execute that marks them bound sets it.
    void SetProjection(const GfMatrix4d& projection) { _projection = projection; }

private:
    void _Deallocate() override;
    void _Refresh();

    cenote::transport::Client* const _client;
    GfMatrix4d _projection{1.0};
    bool _bound = false;
    unsigned int _width = 0;
    unsigned int _height = 0;
    HdFormat _format = HdFormatInvalid;
    std::vector<uint8_t> _pixels;
    std::atomic<int> _mappers = 0;
};

PXR_NAMESPACE_CLOSE_SCOPE
