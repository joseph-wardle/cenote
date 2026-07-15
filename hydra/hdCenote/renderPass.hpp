// The render pass: the hook Hydra's render task executes each frame. Two
// duties, both thin: mark the AOV-bound render buffers so their Map()
// pulls from the server's framebuffer, and keep the server's view current
// — decompose the pass state's camera and send it down the SetCamera lane
// when it changes. Never converged (D-107): usdview keeps repainting,
// which is what streams the frames.
#pragma once

#include "pxr/imaging/hd/renderPass.h"
#include "pxr/pxr.h"

#include "wire/protocol.hpp"

#include <optional>

namespace cenote::transport {
class Client;
}

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteRenderPass final : public HdRenderPass {
public:
    HdCenoteRenderPass(HdRenderIndex* index, HdRprimCollection const& collection,
                       cenote::transport::Client* client);

    bool IsConverged() const override;

protected:
    void _Execute(HdRenderPassStateSharedPtr const& renderPassState,
                  TfTokenVector const& renderTags) override;

private:
    void _UpdateCamera(HdRenderPassStateSharedPtr const& renderPassState);

    cenote::transport::Client* const _client;
    /// The camera the server last acknowledged; nothing is resent until
    /// the decomposition differs from it.
    std::optional<cenote::wire::Camera> _lastCamera;
    bool _warnedNonPerspective = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
