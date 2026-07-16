#include "renderPass.hpp"

#include "renderBuffer.hpp"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/imaging/cameraUtil/framing.h"
#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/hd/renderIndex.h"
#include "pxr/imaging/hd/renderPassState.h"
#include "pxr/imaging/hd/tokens.h"

#include "transport/client.hpp"

#include <array>
#include <cmath>
#include <numbers>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// Gf's doubles to the wire's floats.
std::array<float, 3> _ToArray(GfVec3d const& vector) {
    return {static_cast<float>(vector[0]), static_cast<float>(vector[1]),
            static_cast<float>(vector[2])};
}

/// Exact comparison on purpose: this decides "did the camera move", and
/// any bit of movement should reach the server.
bool _Same(cenote::wire::Camera const& a, cenote::wire::Camera const& b) {
    return a.position == b.position && a.look_at == b.look_at && a.up == b.up &&
           a.vfov_degrees == b.vfov_degrees && a.focus_distance == b.focus_distance &&
           a.aperture_radius == b.aperture_radius;
}

} // namespace

HdCenoteRenderPass::HdCenoteRenderPass(HdRenderIndex* index, HdRprimCollection const& collection,
                                       cenote::transport::Client* client)
    : HdRenderPass(index, collection), _client(client) {}

// Convergence, read live from the shm header and qualified by the epoch
// (D-113): settled counts only once the front frame has incorporated
// everything this client sent, so a stale picture never claims to be
// final — or the client is degraded and the picture will never improve.
// The bound buffers answer the same question with a resize guard on top.
bool HdCenoteRenderPass::IsConverged() const { return _client->converged(); }

void HdCenoteRenderPass::_Execute(HdRenderPassStateSharedPtr const& renderPassState,
                                  TfTokenVector const& /*renderTags*/) {
    // The per-frame health checks: between requests the socket must be
    // silent (anything else degrades, with the warning naming the
    // recovery), and a moved rejected-edit counter means the server has
    // messages waiting for a Ping.
    _client->check_liveness();
    _client->collect_rejections();
    // Mark the bound AOVs: exactly the buffers the viewer reads pull
    // pixels from the server's framebuffer in Map(). Each also learns the
    // frame's projection, which the depth remap reads (D-110).
    const GfMatrix4d projection = renderPassState->GetProjectionMatrix();
    for (HdRenderPassAovBinding const& binding : renderPassState->GetAovBindings()) {
        HdRenderBuffer* buffer = binding.renderBuffer;
        if (buffer == nullptr && !binding.renderBufferId.IsEmpty()) {
            buffer = static_cast<HdRenderBuffer*>(
                GetRenderIndex()->GetBprim(HdPrimTypeTokens->renderBuffer, binding.renderBufferId));
        }
        if (buffer != nullptr) {
            // Every renderBuffer in this index came from the delegate's
            // own factory, so the downcast is sound.
            auto* cenoteBuffer = static_cast<HdCenoteRenderBuffer*>(buffer);
            cenoteBuffer->MarkBound();
            cenoteBuffer->SetProjection(projection);
        }
    }
    _UpdateCamera(renderPassState, projection);
}

void HdCenoteRenderPass::_UpdateCamera(HdRenderPassStateSharedPtr const& renderPassState,
                                       GfMatrix4d const& projection) {
    const HdCamera* camera = renderPassState->GetCamera();
    if (camera == nullptr) {
        return;
    }
    // The vertical field of view comes from the *conformed* projection —
    // the same matrix the depth remap reads (D-110) — so the frame the
    // server renders and the depth read back from it share one camera
    // (D-114). HdCamera still supplies what a projection cannot: the
    // transform, the perspective check, and the lens (focus distance,
    // fStop, focal length — the last only to turn f-number into an
    // aperture radius).
    const double yScale = projection[1][1];
    if (camera->GetProjection() != HdCamera::Perspective || !std::isfinite(yScale) ||
        yScale <= 0.0) {
        if (!_warnedNonPerspective) {
            _warnedNonPerspective = true;
            TF_WARN("only perspective cameras reach cenote-server; the view will not follow %s",
                    camera->GetId().GetText());
        }
        return;
    }
    // Framing beyond a plain full-frame viewport carries more than one
    // field of view can say: cenote renders the full frame at whatever
    // vfov the conformed projection implies. Say so once.
    if (!_warnedExoticFraming) {
        const CameraUtilFraming& framing = renderPassState->GetFraming();
        if (framing.IsValid() && framing != CameraUtilFraming(framing.dataWindow)) {
            _warnedExoticFraming = true;
            TF_WARN("exotic framing (a data window apart from the display window, or non-square "
                    "pixels); cenote renders the full frame");
        }
    }
    const GfMatrix4d& transform = camera->GetTransform();
    const GfVec3d position = transform.ExtractTranslation();
    const GfVec3d forward = transform.TransformDir(GfVec3d(0.0, 0.0, -1.0)).GetNormalized();
    const GfVec3d up = transform.TransformDir(GfVec3d(0.0, 1.0, 0.0)).GetNormalized();
    const float focal = camera->GetFocalLength();
    const float focusDistance = camera->GetFocusDistance();
    const float fStop = camera->GetFStop();
    const bool focused = focusDistance > 0.0f;
    const cenote::wire::Camera current{
        .position = _ToArray(position),
        // The look-at pins the view direction; its distance is arbitrary
        // unless a focus distance makes it the focal plane.
        .look_at = _ToArray(position + forward * (focused ? focusDistance : 1.0f)),
        .up = _ToArray(up),
        // P[1][1] is 1/tan(vfov/2) for any conformed perspective
        // projection, off-center included.
        .vfov_degrees =
            static_cast<float>(2.0 * std::atan(1.0 / yScale) * 180.0 / std::numbers::pi),
        .focus_distance = focused ? std::optional<float>(focusDistance) : std::nullopt,
        // f-number to lens radius, in the same world units as the focal
        // length; an fStop of 0 means "no lens" on both sides.
        .aperture_radius = fStop > 0.0f && focal > 0.0f ? focal / fStop / 2.0f : 0.0f,
    };
    if (_lastCamera && _Same(*_lastCamera, current)) {
        return;
    }
    if (_client->set_camera(current)) {
        _lastCamera = current;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
