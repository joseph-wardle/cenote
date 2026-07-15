// The renderer-plugin bootstrap, isolated in this one thin file: Plug loads
// the library by the name in plugInfo.json, the registry function announces
// the type, and the only thing the class knows how to do is factory the
// delegate. Everything about rendering stays behind CreateRenderDelegate.
#pragma once

#include "pxr/imaging/hd/rendererPlugin.h"
#include "pxr/pxr.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate* CreateRenderDelegate() override;
    void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;
    bool IsSupported(HdRendererCreateArgs const& rendererCreateArgs,
                     std::string* reasonWhyNot = nullptr) const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
