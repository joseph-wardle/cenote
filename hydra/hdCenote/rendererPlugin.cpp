#include "rendererPlugin.hpp"

#include "pxr/imaging/hd/rendererPluginRegistry.h"

#include "renderDelegate.hpp"

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType) { HdRendererPluginRegistry::Define<HdCenoteRendererPlugin>(); }

HdRenderDelegate* HdCenoteRendererPlugin::CreateRenderDelegate() {
    return new HdCenoteRenderDelegate();
}

void HdCenoteRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate) {
    delete renderDelegate;
}

bool HdCenoteRendererPlugin::IsSupported(HdRendererCreateArgs const& /*rendererCreateArgs*/,
                                         std::string* /*reasonWhyNot*/) const {
    // Rendering happens in cenote-server's process; the delegate needs neither
    // the local GPU nor the Hgi the args describe. If the library loaded, it
    // is supported.
    return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
