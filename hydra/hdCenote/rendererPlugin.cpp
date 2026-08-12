#include "rendererPlugin.hpp"

#include "pxr/imaging/hd/rendererPluginRegistry.h"

#include "renderDelegate.hpp"

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType) { HdRendererPluginRegistry::Define<HdCenoteRendererPlugin>(); }

HdRenderDelegate* HdCenoteRendererPlugin::CreateRenderDelegate() {
    return new HdCenoteRenderDelegate();
}

// The overload husk reaches. HdRendererPluginRegistry::CreateRenderDelegate
// calls the no-arg factory when the settings map is empty (usdview's path)
// and this one when it is populated (husk resolves a stage's RenderSettings
// and passes them here); the base default for it returns nullptr, so a
// delegate that overrides only the no-arg form never appears in husk. The
// map is settings authored before there was a delegate to set them on, so
// it goes to the constructor that seeds the base class with it — the
// delegate resolves it on its first Update() like any later edit
// (renderSettings.hpp).
HdRenderDelegate*
HdCenoteRendererPlugin::CreateRenderDelegate(HdRenderSettingsMap const& settingsMap) {
    return new HdCenoteRenderDelegate(settingsMap);
}

void HdCenoteRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate) {
    delete renderDelegate;
}

// Rendering happens in cenote-server's process; the delegate needs neither the
// local GPU nor the Hgi the args would describe, so every parameter is ignored
// (and unnamed, per usdCompat.hpp). If the library loaded, it is supported.
bool HdCenoteRendererPlugin::IsSupported(CENOTE_ISSUPPORTED_PARAMS) const { return true; }

PXR_NAMESPACE_CLOSE_SCOPE
