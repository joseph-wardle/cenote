// The delegate shell: zero Rprims — scene geometry reaches cenote through the
// scene index, never through Rprim hydration — plus the two prims the
// viewer side genuinely needs: the camera Sprim (stock HdCamera) and the
// renderBuffer Bprim the render task draws from. Everything the base class
// defaults sensibly stays defaulted: GetRenderParam() is nullptr,
// capability flags are stock.
//
// It does export three render settings (renderSettings.hpp) — the sample
// budget, the noise threshold, the bounce limit. They travel to the server
// as scene data, not as delegate state: Update() resolves the map into a
// wire SettingsPatch whenever the host has moved it, and the patch rides
// out with everything else the same frame.
//
// Scene content travels the Hydra 2.0 lane instead: SetTerminalSceneIndex
// hangs the observer chain (observer.hpp) off the terminal scene index, and
// Update() — Hydra's serial per-frame hook, run before prim sync — flushes
// the batched notices through the translators and drains whatever they
// appended onto the wire: nothing pending → no send, the first flush →
// Replace (genesis — "the scene is now exactly this", which resets a
// reloaded stage for free), every later flush → Apply.
//
// The delegate also owns the render server's lifetime: constructing it
// spawns cenote-server (transport/client.hpp), destroying it shuts the
// server down. A failed birth degrades — the delegate stays up and renders
// nothing — rather than taking the host with it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/pxr.h"

#include "observer.hpp"
#include "transport/client.hpp"
#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteRenderDelegate final : public HdRenderDelegate {
public:
    HdCenoteRenderDelegate();
    /// The construction-time settings a host resolved before it had a
    /// delegate to set them on — husk's path (rendererPlugin.cpp).
    explicit HdCenoteRenderDelegate(HdRenderSettingsMap const& settingsMap);

    const TfTokenVector& GetSupportedRprimTypes() const override;
    const TfTokenVector& GetSupportedSprimTypes() const override;
    const TfTokenVector& GetSupportedBprimTypes() const override;

    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index,
                                           HdRprimCollection const& collection) override;

    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) override;
    void DestroyInstancer(HdInstancer* instancer) override;

    HdRprim* CreateRprim(TfToken const& typeId, SdfPath const& rprimId) override;
    void DestroyRprim(HdRprim* rPrim) override;

    HdSprim* CreateSprim(TfToken const& typeId, SdfPath const& sprimId) override;
    HdSprim* CreateFallbackSprim(TfToken const& typeId) override;
    void DestroySprim(HdSprim* sprim) override;

    HdBprim* CreateBprim(TfToken const& typeId, SdfPath const& bprimId) override;
    HdBprim* CreateFallbackBprim(TfToken const& typeId) override;
    void DestroyBprim(HdBprim* bprim) override;

    void CommitResources(HdChangeTracker* tracker) override;

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const& name) const override;

    void SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& terminalSceneIndex) override;
    void Update() override;

private:
    /// Resolves the settings map onto `_pending` when the host has moved
    /// it since the last drain, and posts whatever the resolution had to
    /// complain about.
    void _UpdateSettings();

    // First member on purpose: the server is up (or the client is
    // degraded, warning posted) before anything else exists, and it is
    // the last thing torn down.
    cenote::transport::Client _client;
    HdResourceRegistrySharedPtr _resourceRegistry;
    // The edits the translators have appended since the last drain.
    // Declared before the observer that writes into it.
    cenote::wire::ChangeSet _pending;
    std::unique_ptr<HdCenoteObserver> _observer;
    bool _sentGenesis = false;
    /// The settings version whose resolution is already on the wire. The
    /// base class starts its own counter at 1, so 0 means "nothing sent
    /// yet" and the settings ride genesis without a special case.
    unsigned int _settingsSent = 0;
    /// What the last resolution complained about, so a standing
    /// complaint stays quiet while an unrelated knob moves.
    std::vector<std::string> _settingsWarnings;
};

PXR_NAMESPACE_CLOSE_SCOPE
