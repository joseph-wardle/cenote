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
// delegate that overrides only the no-arg form never appears in husk. Cenote
// carries resolution, camera, and sampling to the server as scene data over
// the wire, not as delegate settings, so there is nothing to honour here —
// forward to the no-arg factory and let the map fall away.
HdRenderDelegate*
HdCenoteRendererPlugin::CreateRenderDelegate(HdRenderSettingsMap const& /*settingsMap*/) {
    return CreateRenderDelegate();
}

void HdCenoteRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate) {
    delete renderDelegate;
}

// Rendering happens in cenote-server's process; the delegate needs neither the
// local GPU nor the Hgi the args would describe, so every parameter is ignored
// (and unnamed, per usdCompat.hpp). If the library loaded, it is supported.
bool HdCenoteRendererPlugin::IsSupported(CENOTE_ISSUPPORTED_PARAMS) const { return true; }

PXR_NAMESPACE_CLOSE_SCOPE
