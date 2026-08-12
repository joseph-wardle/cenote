#include "settingsPrim.hpp"

#include <utility>

#include "pxr/base/tf/stringUtils.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/imaging/hd/renderSettingsSchema.h"

#include "usdCompat.hpp"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// The two lanes worth re-reading. Deliberately not the whole
/// renderSettings locator: the filtering scene index also forwards the
/// current frame through this prim, so an animated playback would
/// otherwise re-resolve — and resend — the settings on every frame.
const HdDataSourceLocatorSet& _Locators() {
    static const HdDataSourceLocatorSet locators{
        HdRenderSettingsSchema::GetActiveLocator(),
        HdRenderSettingsSchema::GetNamespacedSettingsLocator()};
    return locators;
}

/// Whether the prim is the one the scene globals named. The flag is
/// computed by the filtering scene index against a single path, so at most
/// one prim in the stage can answer true.
bool _IsActive(const HdRenderSettingsSchema& settings) {
    const HdBoolDataSourceHandle active = settings.GetActive();
    return active && active->GetTypedValue(0.0f);
}

} // namespace

HdCenoteSettingsPrim::HdCenoteSettingsPrim(const SdfPath& path,
                                           const HdsiPrimManagingSceneIndexObserver* observer,
                                           std::shared_ptr<Active> active)
    : _path(path), _active(std::move(active)) {
    _Refresh(observer->GetSceneIndex()->GetPrim(_path));
}

HdCenoteSettingsPrim::~HdCenoteSettingsPrim() { _Release(); }

void HdCenoteSettingsPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                                  const HdsiPrimManagingSceneIndexObserver* observer) {
    if (!entry.dirtyLocators.Intersects(_Locators())) {
        return;
    }
    _Refresh(observer->GetSceneIndex()->GetPrim(_path));
}

void HdCenoteSettingsPrim::_Refresh(const HdSceneIndexPrim& prim) {
    const HdRenderSettingsSchema settings = HdRenderSettingsSchema::GetFromParent(prim.dataSource);
    if (!settings || !_IsActive(settings)) {
        _Release();
        return;
    }
    HdCenoteResolvedSettings resolved =
        HdCenoteResolveNamespacedSettings(cenote::NamespacedSettings(settings));
    if (!resolved.patch.spp && !resolved.patch.noise_threshold && !resolved.patch.max_bounces &&
        !resolved.patch.denoise) {
        // The Karma-or-RenderMan-authored stage: a settings prim is
        // active, it is full of somebody's settings, and none of them are
        // ours. Silence there reads as "cenote ignored my render
        // settings", which is exactly what happened and worth saying once.
        resolved.warnings.push_back(
            TfStringPrintf("the active render settings prim <%s> authors no cenote: setting; "
                           "the host's own render settings still apply",
                           _path.GetText()));
    }
    _Claim(std::move(resolved));
}

void HdCenoteSettingsPrim::_Claim(HdCenoteResolvedSettings resolved) {
    _active->owner = this;
    _active->resolved = std::move(resolved);
    ++_active->version;
}

void HdCenoteSettingsPrim::_Release() {
    if (_active->owner != this) {
        // Either never held, or handed on already: a resync constructs the
        // successor before destroying this one, and a hand-over between
        // two prims lands the claim before the loser is told it lost.
        return;
    }
    _active->owner = nullptr;
    _active->resolved = {};
    ++_active->version;
}

PXR_NAMESPACE_CLOSE_SCOPE
