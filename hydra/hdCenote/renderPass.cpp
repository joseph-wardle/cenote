#include "renderPass.hpp"

#include "renderBuffer.hpp"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
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

// Convergence, read live from the shm header (D-112): the server's
// accumulation settled at its sample cap, or the client is degraded and
// the picture will never improve. The bound buffers answer the same
// question with a resize guard on top.
bool HdCenoteRenderPass::IsConverged() const { return _client->converged(); }

void HdCenoteRenderPass::_Execute(HdRenderPassStateSharedPtr const& renderPassState,
                                  TfTokenVector const& /*renderTags*/) {
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
    _UpdateCamera(renderPassState);
}

void HdCenoteRenderPass::_UpdateCamera(HdRenderPassStateSharedPtr const& renderPassState) {
    const HdCamera* camera = renderPassState->GetCamera();
    if (camera == nullptr) {
        return;
    }
    // HdCamera pre-scales focal length and apertures to world units, so
    // the field-of-view ratio below is unit-free. cenote's camera is a
    // perspective one; framing and conform policy are step 2's business.
    const float focal = camera->GetFocalLength();
    const float aperture = camera->GetVerticalAperture();
    if (camera->GetProjection() != HdCamera::Perspective || focal <= 0.0f || aperture <= 0.0f) {
        if (!_warnedNonPerspective) {
            _warnedNonPerspective = true;
            TF_WARN("only perspective cameras reach cenote-server; the view will not follow %s",
                    camera->GetId().GetText());
        }
        return;
    }
    const GfMatrix4d& transform = camera->GetTransform();
    const GfVec3d position = transform.ExtractTranslation();
    const GfVec3d forward = transform.TransformDir(GfVec3d(0.0, 0.0, -1.0)).GetNormalized();
    const GfVec3d up = transform.TransformDir(GfVec3d(0.0, 1.0, 0.0)).GetNormalized();
    const float focusDistance = camera->GetFocusDistance();
    const float fStop = camera->GetFStop();
    const bool focused = focusDistance > 0.0f;
    const cenote::wire::Camera current{
        .position = _ToArray(position),
        // The look-at pins the view direction; its distance is arbitrary
        // unless a focus distance makes it the focal plane.
        .look_at = _ToArray(position + forward * (focused ? focusDistance : 1.0f)),
        .up = _ToArray(up),
        .vfov_degrees = static_cast<float>(2.0 * std::atan2(0.5 * aperture, double{focal}) * 180.0 /
                                           std::numbers::pi),
        .focus_distance = focused ? std::optional<float>(focusDistance) : std::nullopt,
        // f-number to lens radius, in the same world units as the focal
        // length; an fStop of 0 means "no lens" on both sides.
        .aperture_radius = fStop > 0.0f ? focal / fStop / 2.0f : 0.0f,
    };
    if (_lastCamera && _Same(*_lastCamera, current)) {
        return;
    }
    if (_client->set_camera(current)) {
        _lastCamera = current;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
