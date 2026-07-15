#include "lightPrim.hpp"

#include <cmath>
#include <optional>
#include <utility>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/lightSchema.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A float light parameter, read leniently: the stage path (usdImaging)
/// serves float, and anything absent or oddly typed falls back to the
/// UsdLux default the caller names.
float _FloatOr(const HdContainerDataSourceHandle& light, const TfToken& name,
               const float fallback) {
    if (!light) {
        return fallback;
    }
    const auto source = HdSampledDataSource::Cast(light->Get(name));
    if (!source) {
        return fallback;
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<float>()) {
        return value.UncheckedGet<float>();
    }
    if (value.IsHolding<double>()) {
        return static_cast<float>(value.UncheckedGet<double>());
    }
    return fallback;
}

/// The light's color, defaulting to white like UsdLux.
GfVec3f _ColorOr(const HdContainerDataSourceHandle& light, const TfToken& name) {
    if (!light) {
        return GfVec3f(1.0f);
    }
    const auto source = HdSampledDataSource::Cast(light->Get(name));
    if (!source) {
        return GfVec3f(1.0f);
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<GfVec3f>()) {
        return value.UncheckedGet<GfVec3f>();
    }
    if (value.IsHolding<GfVec3d>()) {
        return GfVec3f(value.UncheckedGet<GfVec3d>());
    }
    return GfVec3f(1.0f);
}

/// Resolved (flattened) visibility; absent means visible.
bool _ReadVisibility(const HdSceneIndexPrim& prim) {
    const HdBoolDataSourceHandle visibility =
        HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
    return visibility ? visibility->GetTypedValue(0.0f) : true;
}

/// The wire payload. Direction is the world -Z of the flattened
/// transform — the axis UsdLux emits along — which for usdview's
/// camera light (transform = the camera's) is the view direction.
/// Irradiance is intensity·2^exposure·color per D-108; the 0.53° angle
/// collapses to the delta, and the rest of UsdLux (color temperature
/// among it) waits for step 4. Nullopt mirrors the server's rejection
/// of a zero direction: a degenerate transform must cost this light,
/// never the atomic flush it rides in.
std::optional<cenote::wire::Distant> _ReadLight(const HdSceneIndexPrim& prim) {
    GfMatrix4d world(1.0);
    if (const HdMatrixDataSourceHandle matrix =
            HdXformSchema::GetFromParent(prim.dataSource).GetMatrix()) {
        world = matrix->GetTypedValue(0.0f);
    }
    // Gf composes row vectors: world -Z is minus the third basis row.
    const GfVec3d direction = GfVec3d(-world[2][0], -world[2][1], -world[2][2]).GetNormalized();
    if (!(std::isfinite(direction[0]) && std::isfinite(direction[1]) &&
          std::isfinite(direction[2])) ||
        direction == GfVec3d(0.0)) {
        return std::nullopt;
    }
    const HdContainerDataSourceHandle light =
        HdLightSchema::GetFromParent(prim.dataSource).GetContainer();
    const float intensity = _FloatOr(light, HdLightTokens->intensity, 1.0f);
    const float exposure = _FloatOr(light, HdLightTokens->exposure, 0.0f);
    const GfVec3f color = _ColorOr(light, HdLightTokens->color);
    const float scale = intensity * std::exp2(exposure);
    return cenote::wire::Distant{
        .direction = {static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                      static_cast<float>(direction[2])},
        .irradiance = {scale * color[0], scale * color[1], scale * color[2]},
    };
}

/// Everything that reshapes the wire Light value: the light params and
/// the transform the direction comes from.
const HdDataSourceLocatorSet& _LightLocators() {
    static const HdDataSourceLocatorSet locators{HdLightSchema::GetDefaultLocator(),
                                                 HdXformSchema::GetDefaultLocator()};
    return locators;
}

} // namespace

HdCenoteLightPrim::HdCenoteLightPrim(const SdfPath& path,
                                     const HdsiPrimManagingSceneIndexObserver* observer,
                                     cenote::wire::ChangeSet* pending,
                                     std::shared_ptr<Registry> live)
    : _path(path), _name(path.GetString()), _pending(pending), _live(std::move(live)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: inherit the previous translator's ledger and take
        // the registry slot so its destructor, which runs after this
        // constructor, goes quietly.
        _sent = it->second->_sent;
        it->second = this;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _Withdraw();
        return;
    }
    _Reconcile(prim, _Dirt{.light = true, .visibility = true}, /*born=*/true);
}

HdCenoteLightPrim::~HdCenoteLightPrim() {
    const auto it = _live->find(_path);
    if (it == _live->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _live->erase(it);
    _Withdraw();
}

void HdCenoteLightPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                               const HdsiPrimManagingSceneIndexObserver* observer) {
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        return;
    }
    _Reconcile(
        prim,
        _Dirt{
            .light = entry.dirtyLocators.Intersects(_LightLocators()),
            .visibility = entry.dirtyLocators.Intersects(HdVisibilitySchema::GetDefaultLocator()),
        },
        /*born=*/false);
}

void HdCenoteLightPrim::_Reconcile(const HdSceneIndexPrim& prim, const _Dirt dirt,
                                   const bool born) {
    if (!born && !dirt.light && !dirt.visibility) {
        return;
    }
    const bool visible = _ReadVisibility(prim);
    std::optional<cenote::wire::Distant> light;
    if (visible) {
        light = _ReadLight(prim);
        if (!light && (born || _sent)) {
            TF_WARN("<%s> has a degenerate transform, so no light direction; light %s",
                    _path.GetText(), _sent ? "removed" : "not sent");
        }
    }
    if (light) {
        // A LightPatch is a handful of floats — always resend whole
        // rather than track which of direction and irradiance moved.
        if (dirt.light || !_sent) {
            _pending->ops.push_back(
                cenote::wire::LightPatch{.name = _name, .light = cenote::wire::Light{*light}});
            _sent = true;
        }
    } else {
        _Withdraw();
    }
}

void HdCenoteLightPrim::_Withdraw() {
    if (_sent) {
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Light, .name = _name});
        _sent = false;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
