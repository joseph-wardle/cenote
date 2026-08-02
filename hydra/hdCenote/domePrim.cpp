#include "domePrim.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/lightSchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/usdLux/blackbody.h"
#include "pxr/usd/usdLux/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A float light parameter, read leniently — restated from the light
/// translator, like the rest of these readers: file-local helpers stay
/// file-local, so each translator reads top to bottom on its own.
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

/// A bool light parameter, same leniency.
bool _BoolOr(const HdContainerDataSourceHandle& light, const TfToken& name, const bool fallback) {
    if (!light) {
        return fallback;
    }
    const auto source = HdSampledDataSource::Cast(light->Get(name));
    if (!source) {
        return fallback;
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<bool>()) {
        return value.UncheckedGet<bool>();
    }
    if (value.IsHolding<int>()) {
        return value.UncheckedGet<int>() != 0;
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

/// A token light parameter — texture:format — read leniently.
TfToken _TokenOr(const HdContainerDataSourceHandle& light, const TfToken& name,
                 const TfToken& fallback) {
    if (!light) {
        return fallback;
    }
    const auto source = HdSampledDataSource::Cast(light->Get(name));
    if (!source) {
        return fallback;
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<TfToken>()) {
        return value.UncheckedGet<TfToken>();
    }
    if (value.IsHolding<std::string>()) {
        return TfToken(value.UncheckedGet<std::string>());
    }
    return fallback;
}

/// A matrix light parameter — domeOffset, which usdImaging serves for
/// DomeLight_1's poleAxis and not at all for the original DomeLight,
/// whose pole is world +Y unconditionally; identity covers both.
GfMatrix4d _MatrixOr(const HdContainerDataSourceHandle& light, const TfToken& name) {
    if (!light) {
        return GfMatrix4d(1.0);
    }
    const auto source = HdSampledDataSource::Cast(light->Get(name));
    if (!source) {
        return GfMatrix4d(1.0);
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<GfMatrix4d>()) {
        return value.UncheckedGet<GfMatrix4d>();
    }
    return GfMatrix4d(1.0);
}

/// Resolved (flattened) visibility; absent means visible.
bool _ReadVisibility(const HdSceneIndexPrim& prim) {
    const HdBoolDataSourceHandle visibility =
        HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
    return visibility ? visibility->GetTypedValue(0.0f) : true;
}

/// The flattened world matrix; identity when unauthored.
GfMatrix4d _WorldMatrix(const HdSceneIndexPrim& prim) {
    if (const HdMatrixDataSourceHandle matrix =
            HdXformSchema::GetFromParent(prim.dataSource).GetMatrix()) {
        return matrix->GetTypedValue(0.0f);
    }
    return GfMatrix4d(1.0);
}

/// The dome's radiance multiplier (Q8, the shared UsdLux radiometry):
/// color times the luminance-normalized blackbody when color temperature
/// is enabled, times intensity·2^exposure. The whole sky is tinted by it
/// — the image never changes, the multiplier rides the wire.
GfVec3f _Tint(const HdContainerDataSourceHandle& light) {
    GfVec3f tint = _ColorOr(light, HdLightTokens->color);
    if (_BoolOr(light, HdLightTokens->enableColorTemperature, false)) {
        tint = GfCompMult(tint, UsdLuxBlackbodyTemperatureAsRgb(
                                    _FloatOr(light, HdLightTokens->colorTemperature, 6500.0f)));
    }
    const float scale = _FloatOr(light, HdLightTokens->intensity, 1.0f) *
                        std::exp2(_FloatOr(light, HdLightTokens->exposure, 0.0f));
    return tint * scale;
}

/// Mirror of the server's validate rule, restated from the mesh
/// translator: every entry of the f32 affine inverse must be finite —
/// the environment maps world directions back through it. A ChangeSet is
/// atomic, so a degenerate placement has to degrade one dome, never the
/// whole flush.
bool _Invertible(const cenote::wire::Matrix& matrix) {
    const auto& rows = matrix.rows;
    const GfVec3f c0(rows[0][0], rows[1][0], rows[2][0]);
    const GfVec3f c1(rows[0][1], rows[1][1], rows[2][1]);
    const GfVec3f c2(rows[0][2], rows[1][2], rows[2][2]);
    const GfVec3f translation(rows[0][3], rows[1][3], rows[2][3]);
    const float determinant = GfDot(c0, GfCross(c1, c2));
    const std::array<GfVec3f, 3> inverseRows = {GfCross(c1, c2) / determinant,
                                                GfCross(c2, c0) / determinant,
                                                GfCross(c0, c1) / determinant};
    for (const GfVec3f& row : inverseRows) {
        if (!(std::isfinite(row[0]) && std::isfinite(row[1]) && std::isfinite(row[2]) &&
              std::isfinite(GfDot(row, translation)))) {
            return false;
        }
    }
    return true;
}

/// The world matrix as the wire's Transform::Matrix — never decomposed.
/// Gf composes against row vectors, the wire against column vectors, so
/// the rows are the transpose's. Nullopt means the server would reject
/// the placement (see _Invertible).
std::optional<cenote::wire::Matrix> _WireTransform(const GfMatrix4d& world) {
    cenote::wire::Matrix transform;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            transform.rows[row][column] = static_cast<float>(world[column][row]);
        }
    }
    if (!_Invertible(transform)) {
        return std::nullopt;
    }
    return transform;
}

/// The 180° yaw between the two equirect conventions, baked innermost
/// into every placement this translator sends: USD maps the image's
/// horizontal center to +Z (the OpenEXR latlong rule the DomeLight
/// schema cites), cenote's environment maps it to −Z — a pure rotation
/// about the dome's own up axis, x → −x and z → −z, so the wire always
/// speaks cenote's convention and the server never learns USD's.
/// Function-local for the same reason as the delegate's type lists: a
/// namespace-scope initializer could throw where nothing catches.
const GfMatrix4d& _Yaw() {
    static const GfMatrix4d yaw(-1.0, 0.0, 0.0, 0.0, //
                                0.0, 1.0, 0.0, 0.0,  //
                                0.0, 0.0, -1.0, 0.0, //
                                0.0, 0.0, 0.0, 1.0);
    return yaw;
}

/// _Yaw() as the wire already spells it — the fallback placement when the
/// authored one is degenerate (a diagonal matrix is its own transpose).
cenote::wire::Matrix _BareYaw() {
    return {{{{-1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 0.0f}}}};
}

/// Everything that reshapes the wire payload: the light params and the
/// transform — one lane, total resend (see _Dirt).
const HdDataSourceLocatorSet& _LightLocators() {
    static const HdDataSourceLocatorSet locators{HdLightSchema::GetDefaultLocator(),
                                                 HdXformSchema::GetDefaultLocator()};
    return locators;
}

} // namespace

HdCenoteDomePrim::HdCenoteDomePrim(const SdfPath& path,
                                   const HdsiPrimManagingSceneIndexObserver* observer,
                                   cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live)
    : _path(path), _name(path.GetString()), _pending(pending), _live(std::move(live)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: inherit the previous translator's ledger — the slot,
        // the warn latches, a resync being no new degradation — and take
        // the registry entry so its destructor, which runs after this
        // constructor, goes quietly.
        _eligible = it->second->_eligible;
        _published = it->second->_published;
        _degenerate = it->second->_degenerate;
        _warned = it->second->_warned;
        it->second = this;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _eligible = false;
        _Withdraw();
        _Arbitrate(nullptr);
        return;
    }
    _Refresh(prim, _Dirt{.light = true, .visibility = true}, /*born=*/true);
}

HdCenoteDomePrim::~HdCenoteDomePrim() {
    const auto it = _live->find(_path);
    if (it == _live->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _live->erase(it);
    _Withdraw();
    // The failover (Q10): with this dome out of the registry, the
    // next-lowest eligible dome takes the slot in the same flush the
    // Remove rides — the sky never goes dark between them.
    _Arbitrate(nullptr);
}

void HdCenoteDomePrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                              const HdsiPrimManagingSceneIndexObserver* observer) {
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        return;
    }
    _Refresh(
        prim,
        _Dirt{
            .light = entry.dirtyLocators.Intersects(_LightLocators()),
            .visibility = entry.dirtyLocators.Intersects(HdVisibilitySchema::GetDefaultLocator()),
        },
        /*born=*/false);
}

void HdCenoteDomePrim::_Refresh(const HdSceneIndexPrim& prim, const _Dirt dirt, const bool born) {
    if (!born && !dirt.light && !dirt.visibility) {
        return;
    }
    _eligible = _ReadVisibility(prim);
    const bool rebuilt = born || dirt.light;
    if (rebuilt) {
        // Total re-read, total resend : any dirt in the
        // lane rebuilds the whole payload, every field explicitly set —
        // path included, Set or Clear, so a de-authored texture really
        // clears server-side.
        const HdContainerDataSourceHandle light =
            HdLightSchema::GetFromParent(prim.dataSource).GetContainer();
        const GfVec3f tint = _Tint(light);
        const GfMatrix4d world =
            _Yaw() * _MatrixOr(light, HdLightTokens->domeOffset) * _WorldMatrix(prim);
        std::optional<cenote::wire::Matrix> matrix = _WireTransform(world);
        if (!matrix) {
            // Unlike a mesh or an area light, a dome with a degenerate
            // placement still lights the scene — the sky has no size to
            // collapse — so the placement falls back to identity (in
            // cenote's convention: the bare yaw) instead of withdrawing.
            // The warning gates on the transition into degeneracy and
            // re-arms on the way out — a latch would go quiet forever,
            // and no gate at all would fire on every unrelated drag.
            if (!_degenerate) {
                TF_WARN("<%s> has a non-invertible transform; the sky stays unturned",
                        _path.GetText());
            }
            _degenerate = true;
            matrix = _BareYaw();
        } else {
            _degenerate = false;
        }
        std::optional<cenote::wire::Reset<std::string>> path = cenote::wire::Clear{};
        if (const std::optional<std::string> file = _SkyImage(light)) {
            path = cenote::wire::Set<std::string>{*file};
        }
        _payload = cenote::wire::EnvironmentPatch{
            .name = _name,
            .path = std::move(path),
            .tint = std::array<float, 3>{tint[0], tint[1], tint[2]},
            .transform = *matrix,
        };
    }
    _Arbitrate(rebuilt ? this : nullptr);
}

std::optional<std::string> HdCenoteDomePrim::_SkyImage(const HdContainerDataSourceHandle& light) {
    if (!light) {
        return std::nullopt;
    }
    const auto source = HdSampledDataSource::Cast(light->Get(HdLightTokens->textureFile));
    if (!source) {
        return std::nullopt;
    }
    const VtValue value = source->GetValue(0.0f);
    if (!value.IsHolding<SdfAssetPath>()) {
        return std::nullopt;
    }
    const SdfAssetPath& asset = value.UncheckedGet<SdfAssetPath>();
    if (asset.GetAssetPath().empty()) {
        return std::nullopt;
    }
    // Ar has already run; the server additionally demands the resolved
    // path be absolute and an existing file, checked here where failing
    // costs one sky instead of the flush.
    const std::string& resolved = asset.GetResolvedPath();
    std::error_code unused;
    if (resolved.empty() || !std::filesystem::path(resolved).is_absolute() ||
        !std::filesystem::is_regular_file(resolved, unused)) {
        if (!_warned.texture) {
            _warned.texture = true;
            TF_WARN("<%s>'s texture:file references \"%s\", which is not a readable file; "
                    "the sky stays constant",
                    _path.GetText(), asset.GetAssetPath().c_str());
        }
        return std::nullopt;
    }
    // Float formats only — the exact inverse of the rect light's rule:
    // the environment is radiance, decoded server-side as .exr or
    // Radiance .hdr, and LDR skies are not supported.
    std::string extension = std::filesystem::path(resolved).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".exr" && extension != ".hdr") {
        if (!_warned.texture) {
            _warned.texture = true;
            TF_WARN("<%s>'s texture:file is not a float-format image (.exr or .hdr), which is "
                    "all the environment decodes; the sky stays constant",
                    _path.GetText());
        }
        return std::nullopt;
    }
    // texture:format gates last, once a file is actually at stake:
    // latlong is the mapping the server implements, and automatic
    // resolves to it for the equirect files admitted above. The ball,
    // angular, and cube-cross unwraps are not supported.
    const TfToken format = _TokenOr(light, HdLightTokens->textureFormat, UsdLuxTokens->automatic);
    if (format != UsdLuxTokens->latlong && format != UsdLuxTokens->automatic) {
        if (!_warned.format) {
            _warned.format = true;
            TF_WARN("<%s> maps its texture as \"%s\", which the environment cannot unwrap "
                    "(latlong only); the sky stays constant",
                    _path.GetText(), format.GetText());
        }
        return std::nullopt;
    }
    return resolved;
}

void HdCenoteDomePrim::_Arbitrate(const HdCenoteDomePrim* const fresh) {
    HdCenoteDomePrim* winner = nullptr;
    for (const auto& [path, dome] : *_live) {
        if (dome->_eligible) {
            winner = dome;
            break;
        }
    }
    // Demotions first, so a hand-over reads causally in the flush:
    // Remove, then the successor's patch — one atomic wave, never two
    // environments merged, never none while a contender stands.
    for (const auto& [path, dome] : *_live) {
        if (dome->_published && dome != winner) {
            dome->_Withdraw();
        }
    }
    if (!winner) {
        return;
    }
    if (!winner->_published || winner == fresh) {
        winner->_Publish();
    }
    for (const auto& [path, dome] : *_live) {
        if (dome->_eligible && dome != winner && !dome->_warned.parked) {
            dome->_warned.parked = true;
            TF_WARN("<%s> is parked: the scene renders one environment, and <%s> — the "
                    "lowest dome path — holds it",
                    dome->_path.GetText(), winner->_path.GetText());
        }
    }
}

void HdCenoteDomePrim::_Publish() {
    _pending->ops.push_back(_payload);
    _published = true;
    // A later demotion is a fresh parking, worth its own warning.
    _warned.parked = false;
}

void HdCenoteDomePrim::_Withdraw() {
    if (!_published) {
        return;
    }
    _pending->ops.push_back(
        cenote::wire::Remove{.kind = cenote::wire::Kind::Environment, .name = _name});
    _published = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
