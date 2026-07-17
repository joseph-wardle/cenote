#include "lightPrim.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/lightSchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/usdLux/blackbody.h"

PXR_NAMESPACE_OPEN_SCOPE

// treatAsPoint is the one UsdLux input read here without the inputs:
// prefix, so it crosses usdImaging verbatim and HdLightTokens never
// spells it.
TF_DEFINE_PRIVATE_TOKENS(_tokens, (treatAsPoint));

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

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

/// The radiometric inputs every UsdLux light shares (Q8): the linear
/// Rec.709 tint — color times the luminance-normalized blackbody when
/// color temperature is enabled — and intensity·2^exposure, D-108's
/// scale. Fallbacks are the UsdLux schema defaults; intensity is the one
/// that differs per type, so the caller names it.
struct _Common {
    GfVec3f tint;
    float scale;
    bool normalize;
};

_Common _ReadCommon(const HdContainerDataSourceHandle& light, const float intensityFallback) {
    GfVec3f tint = _ColorOr(light, HdLightTokens->color);
    if (_BoolOr(light, HdLightTokens->enableColorTemperature, false)) {
        tint = GfCompMult(tint, UsdLuxBlackbodyTemperatureAsRgb(
                                    _FloatOr(light, HdLightTokens->colorTemperature, 6500.0f)));
    }
    const float intensity = _FloatOr(light, HdLightTokens->intensity, intensityFallback);
    const float exposure = _FloatOr(light, HdLightTokens->exposure, 0.0f);
    return {tint, intensity * std::exp2(exposure), _BoolOr(light, HdLightTokens->normalize, false)};
}

/// Mirror of the server's validate_instance rule, restated from the mesh
/// translator: every entry of the f32 affine inverse must be finite. A
/// ChangeSet is atomic — one rejected op takes every edit in the flush
/// down with it — so a degenerate placement has to cost one light, never
/// the whole flush.
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

/// The delta payload for a distant light. Direction is the world -Z of
/// the flattened transform — the axis UsdLux emits along — which for
/// usdview's camera light (transform = the camera's) is the view
/// direction. The 0.53° disk collapses to that delta (the locked floor),
/// but its π·sin²(angle/2) steradians still scale an unnormalized
/// intensity — UsdLux luminance — down to the irradiance the disk
/// delivers; normalize (or a zero angle) makes intensity the irradiance
/// itself. Nullopt mirrors the server's rejection of a zero direction: a
/// degenerate transform must cost this light, never the atomic flush it
/// rides in.
std::optional<cenote::wire::Light> _Distant(const GfMatrix4d& world, const _Common& common,
                                            const HdContainerDataSourceHandle& light) {
    // Gf composes row vectors: world -Z is minus the third basis row.
    const GfVec3d direction = GfVec3d(-world[2][0], -world[2][1], -world[2][2]).GetNormalized();
    if (!(std::isfinite(direction[0]) && std::isfinite(direction[1]) &&
          std::isfinite(direction[2])) ||
        direction == GfVec3d(0.0)) {
        return std::nullopt;
    }
    const float angle = std::max(_FloatOr(light, HdLightTokens->angle, 0.53f), 0.0f);
    const double sinHalf = std::sin(static_cast<double>(angle) * std::numbers::pi / 360.0);
    const float solidAngle = common.normalize || angle == 0.0f
                                 ? 1.0f
                                 : static_cast<float>(std::numbers::pi * sinHalf * sinHalf);
    const GfVec3f irradiance = common.tint * (common.scale * solidAngle);
    return cenote::wire::Distant{
        .direction = {static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                      static_cast<float>(direction[2])},
        .irradiance = {irradiance[0], irradiance[1], irradiance[2]},
    };
}

/// The delta payload for a sphere marked treatAsPoint: position is the
/// flattened translation, and intensity is the radiant intensity the
/// sphere's surface delivers — luminance times the projected area π·r²,
/// or intensity/4 when normalize has already divided the 4π·r² surface
/// out. The radius rides an anisotropic world scale as |det|^(1/3), the
/// one isotropic reading of a stretch a point cannot express (amendment
/// E). Nullopt keeps a non-finite position off the wire, like _Distant's
/// degenerate direction.
std::optional<cenote::wire::Light> _Point(const GfMatrix4d& world, const _Common& common,
                                          const HdContainerDataSourceHandle& light) {
    const GfVec3d position(world[3][0], world[3][1], world[3][2]);
    if (!(std::isfinite(position[0]) && std::isfinite(position[1]) && std::isfinite(position[2]))) {
        return std::nullopt;
    }
    const float radius = std::max(_FloatOr(light, HdLightTokens->radius, 0.5f), 0.0f);
    const float rWorld = radius * static_cast<float>(std::cbrt(std::abs(world.GetDeterminant3())));
    const float projected = common.normalize ? 0.25f : kPi * rWorld * rWorld;
    const GfVec3f intensity = common.tint * (common.scale * projected);
    return cenote::wire::Point{
        .position = {static_cast<float>(position[0]), static_cast<float>(position[1]),
                     static_cast<float>(position[2])},
        .intensity = {intensity[0], intensity[1], intensity[2]},
    };
}

// -- The synthesized area geometry -------------------------------------------
// Each UsdLux area shape becomes object-space triangles with the authored
// dimensions baked into the vertices, wound counter-clockwise-outward on
// the side UsdLux emits from — the server's emitters are one-sided,
// winding-front. The renderer keeps no analytic light shapes: emissive
// triangles are its native area light, so the segment counts below are
// the shapes' whole cost — light sampling sees only triangles, and the
// silhouette is camera-invisible anyway.

/// Rect: two triangles in the XY plane facing -Z. The UVs restate the
/// schema's texture frame — image minimum at the (+X,+Y) corner, maximum
/// at (-X,-Y) — in cenote's convention, where v=0 samples the image
/// file's first (top) row while USD's (0,0) is the displayed image's
/// lower left: u = ½ − x/w, v = ½ + y/h.
cenote::wire::Inline _RectSource(const float width, const float height) {
    const float w = width / 2.0f;
    const float h = height / 2.0f;
    cenote::wire::Inline source;
    source.positions = {{w, h, 0.0f}, {-w, h, 0.0f}, {-w, -h, 0.0f}, {w, -h, 0.0f}};
    source.uvs = {{{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}}};
    source.triangles = {{0, 2, 1}, {0, 3, 2}};
    return source;
}

/// Disk: a 64-segment fan in the XY plane facing -Z.
cenote::wire::Inline _DiskSource(const float radius) {
    constexpr std::uint32_t kSegments = 64;
    cenote::wire::Inline source;
    source.positions.reserve(kSegments + 1);
    source.positions.push_back({0.0f, 0.0f, 0.0f});
    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float theta = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kSegments);
        source.positions.push_back({radius * std::cos(theta), radius * std::sin(theta), 0.0f});
    }
    source.triangles.reserve(kSegments);
    for (std::uint32_t i = 0; i < kSegments; ++i) {
        source.triangles.push_back({0, 1 + (i + 1) % kSegments, 1 + i});
    }
    return source;
}

/// Sphere: a 32×16 UV sphere, poles on Z.
cenote::wire::Inline _SphereSource(const float radius) {
    constexpr std::uint32_t kSegments = 32;
    constexpr std::uint32_t kStacks = 16;
    cenote::wire::Inline source;
    source.positions.reserve(std::size_t{kStacks - 1} * kSegments + 2);
    source.positions.push_back({0.0f, 0.0f, radius});
    for (std::uint32_t k = 1; k < kStacks; ++k) {
        const float phi = kPi * static_cast<float>(k) / static_cast<float>(kStacks);
        for (std::uint32_t j = 0; j < kSegments; ++j) {
            const float theta = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(kSegments);
            source.positions.push_back({radius * std::sin(phi) * std::cos(theta),
                                        radius * std::sin(phi) * std::sin(theta),
                                        radius * std::cos(phi)});
        }
    }
    source.positions.push_back({0.0f, 0.0f, -radius});
    const auto ring = [](const std::uint32_t k, const std::uint32_t j) {
        return 1 + (k - 1) * kSegments + j % kSegments;
    };
    const std::uint32_t south = 1 + (kStacks - 1) * kSegments;
    source.triangles.reserve(2 * std::size_t{kStacks - 1} * kSegments);
    for (std::uint32_t j = 0; j < kSegments; ++j) {
        source.triangles.push_back({0, ring(1, j), ring(1, j + 1)});
    }
    for (std::uint32_t k = 1; k + 1 < kStacks; ++k) {
        for (std::uint32_t j = 0; j < kSegments; ++j) {
            source.triangles.push_back({ring(k, j), ring(k + 1, j), ring(k + 1, j + 1)});
            source.triangles.push_back({ring(k, j), ring(k + 1, j + 1), ring(k, j + 1)});
        }
    }
    for (std::uint32_t j = 0; j < kSegments; ++j) {
        source.triangles.push_back({south, ring(kStacks - 1, j + 1), ring(kStacks - 1, j)});
    }
    return source;
}

/// Cylinder: a 64-segment tube along X — no end caps, which is UsdLux's
/// own reading of the shape.
cenote::wire::Inline _TubeSource(const float radius, const float length) {
    constexpr std::uint32_t kSegments = 64;
    cenote::wire::Inline source;
    source.positions.reserve(2 * std::size_t{kSegments});
    for (const float x : {-length / 2.0f, length / 2.0f}) {
        for (std::uint32_t j = 0; j < kSegments; ++j) {
            const float theta = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(kSegments);
            source.positions.push_back({x, radius * std::cos(theta), radius * std::sin(theta)});
        }
    }
    source.triangles.reserve(2 * std::size_t{kSegments});
    for (std::uint32_t j = 0; j < kSegments; ++j) {
        const std::uint32_t next = (j + 1) % kSegments;
        source.triangles.push_back({j, kSegments + next, kSegments + j});
        source.triangles.push_back({j, next, kSegments + next});
    }
    return source;
}

/// The object-space triangles for one area light, dimensions read at
/// their UsdLux schema defaults, clamped non-negative — a negative width
/// would silently flip the emitting side.
cenote::wire::Inline _AreaSource(const TfToken& type, const HdContainerDataSourceHandle& light) {
    const auto dimension = [&light](const TfToken& name, const float fallback) {
        return std::max(_FloatOr(light, name, fallback), 0.0f);
    };
    if (type == HdPrimTypeTokens->rectLight) {
        return _RectSource(dimension(HdLightTokens->width, 1.0f),
                           dimension(HdLightTokens->height, 1.0f));
    }
    if (type == HdPrimTypeTokens->diskLight) {
        return _DiskSource(dimension(HdLightTokens->radius, 0.5f));
    }
    if (type == HdPrimTypeTokens->sphereLight) {
        return _SphereSource(dimension(HdLightTokens->radius, 0.5f));
    }
    return _TubeSource(dimension(HdLightTokens->radius, 0.5f),
                       dimension(HdLightTokens->length, 1.0f));
}

/// The synthesized triangles' summed area under the world placement —
/// the exact quantity `normalize` divides by, measured on the very
/// geometry being sent.
double _WorldArea(const cenote::wire::Inline& source, const GfMatrix4d& world) {
    const auto point = [&](const std::uint32_t index) {
        const std::array<float, 3>& p = source.positions[index];
        return world.Transform(GfVec3d(p[0], p[1], p[2]));
    };
    double area = 0.0;
    for (const std::array<std::uint32_t, 3>& triangle : source.triangles) {
        const GfVec3d a = point(triangle[0]);
        area += 0.5 * GfCross(point(triangle[1]) - a, point(triangle[2]) - a).GetLength();
    }
    return area;
}

/// Everything that reshapes the wire payload: the light params and the
/// transform — one lane, total resend (see _Dirt).
const HdDataSourceLocatorSet& _LightLocators() {
    static const HdDataSourceLocatorSet locators{HdLightSchema::GetDefaultLocator(),
                                                 HdXformSchema::GetDefaultLocator()};
    return locators;
}

} // namespace

HdCenoteLightPrim::HdCenoteLightPrim(const SdfPath& path, const TfToken& type,
                                     const HdsiPrimManagingSceneIndexObserver* observer,
                                     cenote::wire::ChangeSet* pending,
                                     std::shared_ptr<Registry> live)
    : _path(path), _name(path.GetString()), _type(type), _pending(pending), _live(std::move(live)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: inherit the previous translator's ledger — the warn
        // latches included, a resync being no new approximation — and
        // take the registry slot so its destructor, which runs after
        // this constructor, goes quietly.
        _sent = it->second->_sent;
        _warned = it->second->_warned;
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
    if (!_ReadVisibility(prim)) {
        _Withdraw();
        return;
    }
    const HdContainerDataSourceHandle light =
        HdLightSchema::GetFromParent(prim.dataSource).GetContainer();
    const bool delta =
        _type == HdPrimTypeTokens->distantLight ||
        (_type == HdPrimTypeTokens->sphereLight && _BoolOr(light, _tokens->treatAsPoint, false));
    const _Spelling spelling = delta ? _Spelling::Delta : _Spelling::Area;
    if (!dirt.light && _sent == spelling) {
        // A visibility wobble around objects already standing.
        return;
    }
    // Total re-read, total resend (D-113/D-115): any dirt in the lane
    // rebuilds the whole payload — no per-field diffing, no
    // identical-patch suppression — and a spelling flip (a sphere
    // toggling treatAsPoint) withdraws the old objects in the same
    // atomic flush the new ones ride.
    const GfMatrix4d world = _WorldMatrix(prim);
    const _Common common =
        _ReadCommon(light, _type == HdPrimTypeTokens->distantLight ? 50000.0f : 1.0f);
    if (!_warned.diffuse && _FloatOr(light, HdLightTokens->diffuse, 1.0f) != 1.0f) {
        _warned.diffuse = true;
        TF_WARN("<%s> scales its diffuse contribution, an artistic split cenote's one "
                "transport does not carry; ignored",
                _path.GetText());
    }
    if (!_warned.specular && _FloatOr(light, HdLightTokens->specular, 1.0f) != 1.0f) {
        _warned.specular = true;
        TF_WARN("<%s> scales its specular contribution, an artistic split cenote's one "
                "transport does not carry; ignored",
                _path.GetText());
    }
    if (delta) {
        const std::optional<cenote::wire::Light> payload = _type == HdPrimTypeTokens->distantLight
                                                               ? _Distant(world, common, light)
                                                               : _Point(world, common, light);
        if (!payload) {
            if (born || _sent != _Spelling::None) {
                TF_WARN("<%s> has a degenerate transform; light %s", _path.GetText(),
                        _sent != _Spelling::None ? "removed" : "not sent");
            }
            _Withdraw();
            return;
        }
        if (_sent == _Spelling::Area) {
            _Withdraw();
        }
        _pending->ops.push_back(cenote::wire::LightPatch{.name = _name, .light = payload});
        _sent = _Spelling::Delta;
        return;
    }
    // The area triple (Q5/Q6): geometry, absorber material, instance —
    // the flattened matrix rides the instance untouched, and the
    // instance is never camera-visible (the light illuminates; its
    // shape is a sampling device, not scenery).
    const std::optional<cenote::wire::Matrix> matrix = _WireTransform(world);
    if (!matrix) {
        if (born || _sent != _Spelling::None) {
            TF_WARN("<%s> has a non-invertible transform; light %s", _path.GetText(),
                    _sent != _Spelling::None ? "removed" : "not sent");
        }
        _Withdraw();
        return;
    }
    cenote::wire::Inline source = _AreaSource(_type, light);
    if (world.GetDeterminant3() < 0.0) {
        // A mirroring placement reverses winding in world space; flipping
        // the triples keeps the emitting face on the side UsdLux lights
        // (amendment B).
        for (std::array<std::uint32_t, 3>& triangle : source.triangles) {
            std::swap(triangle[1], triangle[2]);
        }
    }
    // normalize divides the luminance by world-space area — measured on
    // the triangles being sent, so what the renderer integrates is
    // exactly what the divisor weighed. Zero area (a zero dimension or a
    // collapsing scale) keeps divisor 1, silently: the shape is already
    // dark by geometry, the same reading UsdLux prescribes for its
    // sizeFactor (amendment D).
    double luminance = common.scale;
    if (common.normalize) {
        const double area = _WorldArea(source, world);
        luminance /= area > 0.0 ? area : 1.0;
    }
    cenote::wire::MaterialPatch material =
        _EmissionMaterial(light, common.tint, static_cast<float>(luminance));
    if (_sent == _Spelling::Delta) {
        _Withdraw();
    }
    _pending->ops.push_back(cenote::wire::MeshPatch{
        .name = _name, .source = cenote::wire::MeshSource{std::move(source)}});
    _pending->ops.push_back(std::move(material));
    _pending->ops.push_back(cenote::wire::InstancePatch{.name = _name,
                                                        .mesh = _name,
                                                        .material = _name,
                                                        .transform = *matrix,
                                                        .camera_visible = false});
    _sent = _Spelling::Area;
}

cenote::wire::MaterialPatch
HdCenoteLightPrim::_EmissionMaterial(const HdContainerDataSourceHandle& light, const GfVec3f tint,
                                     float luminance) {
    // The black absorber (amendment C): the camera never sees this
    // surface — the instance is camera-invisible — but reflections and
    // shadow rays still hit it, and the wire default, an 80% gray,
    // would bounce light the light never emitted. Black diffuse, no
    // specular, emission only.
    cenote::wire::MaterialPatch material{
        .name = _name,
        .base_color = cenote::wire::Constant<std::array<float, 3>>{{0.0f, 0.0f, 0.0f}},
        .specular_weight = 0.0f,
    };
    cenote::wire::Texturable<std::array<float, 3>> emission =
        cenote::wire::Constant<std::array<float, 3>>{{tint[0], tint[1], tint[2]}};
    if (_type == HdPrimTypeTokens->rectLight) {
        if (const std::optional<std::string> file = _TextureFile(light)) {
            // The map replaces the constant color on the wire, and the
            // shader multiplies it onto the scalar luminance alone — so
            // the tint has one lane left: its Rec.709 luminance folds
            // into the scalar, and any hue is dropped under a latched
            // warning (Q7).
            emission = cenote::wire::TextureRef{*file, std::nullopt, std::nullopt};
            luminance *= 0.2126f * tint[0] + 0.7152f * tint[1] + 0.0722f * tint[2];
            if (!_warned.tint && !(tint[0] == tint[1] && tint[1] == tint[2])) {
                _warned.tint = true;
                TF_WARN("<%s> tints its textured emission; only the tint's luminance "
                        "survives, and the hue is dropped",
                        _path.GetText());
            }
        }
    }
    material.emission_luminance = luminance;
    material.emission_color = std::move(emission);
    return material;
}

std::optional<std::string>
HdCenoteLightPrim::_TextureFile(const HdContainerDataSourceHandle& light) {
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
    // costs one texture instead of the flush (D-117).
    const std::string& resolved = asset.GetResolvedPath();
    std::error_code unused;
    if (resolved.empty() || !std::filesystem::path(resolved).is_absolute() ||
        !std::filesystem::is_regular_file(resolved, unused)) {
        if (!_warned.texture) {
            _warned.texture = true;
            TF_WARN("<%s>'s texture:file references \"%s\", which is not a readable file; "
                    "emission stays constant",
                    _path.GetText(), asset.GetAssetPath().c_str());
        }
        return std::nullopt;
    }
    // LDR only: the emission slot rides the BC7 color pipeline, which
    // would clip a float source to display range rather than reject it.
    // HDR emission maps are deferred beyond M4.
    std::string extension = std::filesystem::path(resolved).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".exr" || extension == ".hdr") {
        if (!_warned.texture) {
            _warned.texture = true;
            TF_WARN("<%s>'s texture:file is a float-format image, which the emission slot "
                    "cannot carry; emission stays constant",
                    _path.GetText());
        }
        return std::nullopt;
    }
    return resolved;
}

void HdCenoteLightPrim::_Withdraw() {
    switch (_sent) {
    case _Spelling::None:
        return;
    case _Spelling::Delta:
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Light, .name = _name});
        break;
    case _Spelling::Area:
        // The instance — the one holding references — goes first, then
        // the material, then the mesh; the server validates post-merge
        // either way, but the list should read causally.
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Instance, .name = _name});
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Material, .name = _name});
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Mesh, .name = _name});
        break;
    }
    _sent = _Spelling::None;
}

PXR_NAMESPACE_CLOSE_SCOPE
