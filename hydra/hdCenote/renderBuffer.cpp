#include "renderBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/diagnostic.h"

#include "transport/client.hpp"
#include "transport/view.hpp"

PXR_NAMESPACE_OPEN_SCOPE

HdCenoteRenderBuffer::HdCenoteRenderBuffer(SdfPath const& id, cenote::transport::Client* client)
    : HdRenderBuffer(id), _client(client) {}

bool HdCenoteRenderBuffer::Allocate(GfVec3i const& dimensions, HdFormat format,
                                    bool /*multiSampled*/) {
    // The delegate's AOV descriptors advertise exactly two formats; anything
    // else reaching here is a wiring error, not a request to satisfy.
    if (format != HdFormatFloat32Vec4 && format != HdFormatFloat32) {
        TF_WARN("HdCenoteRenderBuffer supports only f32 RGBA and f32 depth, got format %d",
                static_cast<int>(format));
        _Deallocate();
        return false;
    }
    if (dimensions[0] < 0 || dimensions[1] < 0 || dimensions[2] != 1) {
        TF_WARN("HdCenoteRenderBuffer expects non-negative 2D dimensions, got (%d, %d, %d)",
                dimensions[0], dimensions[1], dimensions[2]);
        _Deallocate();
        return false;
    }

    _width = static_cast<unsigned int>(dimensions[0]);
    _height = static_cast<unsigned int>(dimensions[1]);
    _format = format;
    // Zero-filled: until a frame arrives from the server, readers see black.
    _pixels.assign(static_cast<size_t>(_width) * _height * HdDataSizeOfFormat(format), 0);
    // The resize lane: the server renders at whatever size was allocated
    // last. A no-op when the framebuffer already matches (the depth buffer
    // allocates right after color, same dimensions); a failure means
    // degraded, and the zeroed pixels above are the degraded picture.
    _client->resize(_width, _height);
    return true;
}

unsigned int HdCenoteRenderBuffer::GetWidth() const { return _width; }

unsigned int HdCenoteRenderBuffer::GetHeight() const { return _height; }

unsigned int HdCenoteRenderBuffer::GetDepth() const { return 1; }

HdFormat HdCenoteRenderBuffer::GetFormat() const { return _format; }

// The server hands back fully resolved pixels; there is never a sample buffer
// on this side to collapse, so multisampling is declined wholesale.
bool HdCenoteRenderBuffer::IsMultiSampled() const { return false; }

void* HdCenoteRenderBuffer::Map() {
    ++_mappers;
    _Refresh();
    return _pixels.data();
}

void HdCenoteRenderBuffer::Unmap() { --_mappers; }

bool HdCenoteRenderBuffer::IsMapped() const { return _mappers.load() != 0; }

void HdCenoteRenderBuffer::Resolve() {}

// Convergence, read live from the shm header (D-112): this buffer's picture
// is final once the server's accumulation has settled. Degraded means the
// zeroed pixels *are* the final picture — true, so a render-until-converged
// host (usdrecord) never spins against a dead server. A segment that trails
// the allocation is a resize still settling: frames are coming, not
// converged.
bool HdCenoteRenderBuffer::IsConverged() const {
    const cenote::transport::View* view = _client->view();
    if (view == nullptr) {
        return true;
    }
    if (view->width() != _width || view->height() != _height) {
        return false;
    }
    return view->converged();
}

// Pull the newest frame out of the server's framebuffer — one plane copy
// under the tear protocol, chosen by format: this buffer is either the
// beauty or the depth, never both. Skipped while unbound, degraded, or
// whenever the segment size trails the allocation (a resize settling);
// the pixels then keep whatever they last held, black at birth.
void HdCenoteRenderBuffer::_Refresh() {
    if (!_bound) {
        return;
    }
    const cenote::transport::View* view = _client->view();
    if (view == nullptr || view->width() != _width || view->height() != _height) {
        return;
    }
    const std::span<float> plane{reinterpret_cast<float*>(_pixels.data()),
                                 _pixels.size() / sizeof(float)};
    if (_format == HdFormatFloat32Vec4) {
        view->copy_beauty(plane);
    } else {
        view->copy_depth(plane);
        // The shm plane holds camera-plane z in meters, +inf where every
        // sample missed; Hydra's depth semantic is the projection's
        // [0, 1] (D-110). Only z is needed — the plane already carries
        // camera-forward distance, so the remap is the projection's z/w
        // rows and a divide, hdEmbree's conversion without the full
        // point transform. Misses land exactly on the 1.0 clear value.
        const double zz = _projection[2][2];
        const double zw = _projection[2][3];
        const double wz = _projection[3][2];
        const double ww = _projection[3][3];
        for (float& depth : plane) {
            if (!std::isfinite(depth)) {
                depth = 1.0f;
                continue;
            }
            // Gf composes row vectors and eye space looks down -Z.
            const double eyeZ = -static_cast<double>(depth);
            const double ndc = (eyeZ * zz + wz) / (eyeZ * zw + ww);
            depth = static_cast<float>(std::clamp(0.5 * (ndc + 1.0), 0.0, 1.0));
        }
    }
}

void HdCenoteRenderBuffer::_Deallocate() {
    _width = 0;
    _height = 0;
    _format = HdFormatInvalid;
    _pixels.clear();
}

PXR_NAMESPACE_CLOSE_SCOPE
