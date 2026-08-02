// The renderer-plugin bootstrap, isolated in this one thin file: Plug loads
// the library by the name in plugInfo.json, the registry function announces
// the type, and the only thing the class knows how to do is factory the
// delegate. Everything about rendering stays behind CreateRenderDelegate.
#pragma once

#include "pxr/imaging/hd/rendererPlugin.h"
#include "pxr/pxr.h"

#include "usdCompat.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate* CreateRenderDelegate() override;
    // husk creates the delegate through this settings-map overload; the base
    // default returns nullptr, so a renderer that omits it is invisible to
    // husk. Cenote honours no delegate settings, so it forwards (usdCompat
    // note: both USD versions declare this overload identically — no shim).
    HdRenderDelegate* CreateRenderDelegate(HdRenderSettingsMap const& settingsMap) override;
    void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;
    // The pure IsSupported override differs by USD version; the parameter
    // list comes from usdCompat.hpp.
    bool IsSupported(CENOTE_ISSUPPORTED_PARAMS) const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
