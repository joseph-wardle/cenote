// The render pass: the hook Hydra's render task executes each frame. Thin
// duties, all steady-state: run the client's per-frame health checks
// (silent-socket liveness, rejected-edit collection), mark the AOV-bound
// render buffers so their Map() pulls from the server's framebuffer, and
// keep the server's view current — decompose the pass state's camera and
// send it down the SetCamera lane when it changes.
#pragma once

#include "pxr/base/gf/matrix4d.h"
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
    void _UpdateCamera(HdRenderPassStateSharedPtr const& renderPassState,
                       GfMatrix4d const& projection);

    cenote::transport::Client* const _client;
    /// The camera the server last acknowledged; nothing is resent until
    /// the decomposition differs from it.
    std::optional<cenote::wire::Camera> _lastCamera;
    bool _warnedNonPerspective = false;
    bool _warnedExoticFraming = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
