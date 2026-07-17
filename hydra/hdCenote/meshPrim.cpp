#include "meshPrim.hpp"

#include "instancerPrim.hpp"
#include "materialPrim.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/instancedBySchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopology.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/types.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/imaging/pxOsd/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (st));

namespace {

/// The server's own neutral default base color (description.rs). Sent
/// explicitly rather than left to get-or-create defaulting, so a
/// displayColor that disappears across a resync cannot leave a stale
/// color on the companion material.
constexpr std::array<float, 3> kNeutralColor{0.8f, 0.8f, 0.8f};

TfToken _TokenOr(const HdTokenDataSourceHandle& source, const TfToken& fallback) {
    return source ? source->GetTypedValue(0.0f) : fallback;
}

VtIntArray _IntsOr(const HdIntArrayDataSourceHandle& source) {
    return source ? source->GetTypedValue(0.0f) : VtIntArray();
}

/// Mirror of the server's validate_instance rule: every entry of the
/// f32 affine inverse must be finite. The mirror matters because a
/// ChangeSet is atomic — one rejected op takes every edit in the flush
/// down with it — so a degenerate transform has to cost one instance,
/// never the whole flush.
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

/// A primvar's values and their domain: per corner of the triangulation
/// (an un-welded faceVarying read, three per triangle) or per point.
template <typename Array> struct _Primvar {
    Array values;
    bool perCorner;
};

/// A primvar the wire can carry, or the reasons there is none: a count
/// mismatch warns and drops, and anything else (absent, uniform,
/// constant, an unexpected type) quietly reads as absent. faceVarying
/// values come back per corner, re-fanned by HdMeshUtil in the same
/// order ComputeTriangleIndices walked, so index i lands on triangle
/// i/3's corner i%3; vertex and varying values pass through per point.
template <typename Array>
std::optional<_Primvar<Array>> _ReadPrimvar(const SdfPath& path, const HdPrimvarsSchema& primvars,
                                            const TfToken& name, const size_t pointCount,
                                            const size_t cornerCount, const HdMeshUtil& util) {
    const HdPrimvarSchema primvar = primvars.GetPrimvar(name);
    const HdSampledDataSourceHandle value = primvar.GetPrimvarValue();
    if (!value) {
        return std::nullopt;
    }
    const TfToken interpolation = _TokenOr(primvar.GetInterpolation(), TfToken());
    const bool faceVarying = interpolation == HdPrimvarSchemaTokens->faceVarying;
    if (!faceVarying && interpolation != HdPrimvarSchemaTokens->vertex &&
        interpolation != HdPrimvarSchemaTokens->varying) {
        return std::nullopt;
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (!sampled.IsHolding<Array>()) {
        return std::nullopt;
    }
    Array array = sampled.UncheckedGet<Array>();
    const size_t expected = faceVarying ? cornerCount : pointCount;
    if (array.size() != expected) {
        TF_WARN("<%s> has %zu %s for %zu %s; dropped", path.GetText(), array.size(), name.GetText(),
                expected, faceVarying ? "corners" : "points");
        return std::nullopt;
    }
    if (!faceVarying) {
        return _Primvar<Array>{std::move(array), false};
    }
    VtValue triangulated;
    const HdMeshComputationResult result = util.ComputeTriangulatedFaceVaryingPrimvar(
        HdGetValueData(sampled), static_cast<int>(array.size()), HdGetValueTupleType(sampled).type,
        &triangulated);
    if (result == HdMeshComputationResult::Error) {
        TF_WARN("<%s>: triangulating the faceVarying %s primvar failed; dropped", path.GetText(),
                name.GetText());
        return std::nullopt;
    }
    if (result == HdMeshComputationResult::Success) {
        array = triangulated.UncheckedGet<Array>();
    }
    // Unchanged: the cage is already all triangles, right-handed, with
    // nothing skipped, so the authored corners sit in fan order as-is.
    return _Primvar<Array>{std::move(array), true};
}

/// points — always vertex, the one primvar geometry cannot do without.
VtVec3fArray _Points(const SdfPath& path, const HdPrimvarsSchema& primvars) {
    const HdSampledDataSourceHandle value = primvars.GetPrimvar(HdTokens->points).GetPrimvarValue();
    if (!value) {
        return {};
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (!sampled.IsHolding<VtVec3fArray>()) {
        TF_WARN("<%s> has points of type %s, not GfVec3f[]; mesh dropped", path.GetText(),
                sampled.GetTypeName().c_str());
        return {};
    }
    return sampled.UncheckedGet<VtVec3fArray>();
}

/// The wire payload: the base cage fan-triangulated by HdMeshUtil,
/// which honors orientation (leftHanded fans come out reversed), so the
/// triples land counter-clockwise-outward as Inline requires. cenote's
/// format is single-indexed — attributes live per position — so a
/// welded mesh copies straight through, and any faceVarying primvar
/// un-welds the mesh to three vertices per triangle so every per-corner
/// value has a position to sit on. Nullopt is "nothing the server would
/// accept" — an empty mesh is legal USD, but validation rejects a Mesh
/// without geometry.
std::optional<cenote::wire::Inline> _ReadGeometry(const SdfPath& path,
                                                  const HdSceneIndexPrim& prim) {
    const HdMeshSchema mesh = HdMeshSchema::GetFromParent(prim.dataSource);
    const HdMeshTopologySchema topologySchema = mesh.GetTopology();
    const VtIntArray counts = _IntsOr(topologySchema.GetFaceVertexCounts());
    const VtIntArray indices = _IntsOr(topologySchema.GetFaceVertexIndices());
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    const VtVec3fArray points = _Points(path, primvars);
    if (points.empty() || counts.empty() || indices.empty()) {
        return std::nullopt;
    }
    for (const int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= points.size()) {
            TF_WARN("<%s> indexes vertex %d of %zu; mesh dropped", path.GetText(), index,
                    points.size());
            return std::nullopt;
        }
    }
    const HdMeshTopology topology(
        _TokenOr(mesh.GetSubdivisionScheme(), PxOsdOpenSubdivTokens->none),
        _TokenOr(topologySchema.GetOrientation(), HdTokens->rightHanded), counts, indices,
        _IntsOr(topologySchema.GetHoleIndices()));
    const HdMeshUtil util(&topology, path);
    VtVec3iArray triangles;
    VtIntArray primitiveParams;
    util.ComputeTriangleIndices(&triangles, &primitiveParams);
    if (triangles.empty()) {
        return std::nullopt;
    }
    const auto normals = _ReadPrimvar<VtVec3fArray>(path, primvars, HdTokens->normals,
                                                    points.size(), indices.size(), util);
    const auto st = _ReadPrimvar<VtVec2fArray>(path, primvars, _tokens->st, points.size(),
                                               indices.size(), util);
    cenote::wire::Inline source;
    if ((normals && normals->perCorner) || (st && st->perCorner)) {
        // The un-weld: positions gather through the triangle indices,
        // per-corner values copy straight across, and per-point values
        // gather alongside the positions. Only meshes that carry
        // faceVarying data pay the duplication.
        source.positions.reserve(3 * triangles.size());
        source.triangles.reserve(triangles.size());
        if (normals) {
            source.normals.emplace();
            source.normals->reserve(3 * triangles.size());
        }
        if (st) {
            source.uvs.emplace();
            source.uvs->reserve(3 * triangles.size());
        }
        for (size_t t = 0; t < triangles.size(); ++t) {
            const GfVec3i& triangle = triangles[t];
            const auto base = static_cast<std::uint32_t>(3 * t);
            source.triangles.push_back({base, base + 1, base + 2});
            for (int k = 0; k < 3; ++k) {
                const GfVec3f& position = points[triangle[k]];
                source.positions.push_back({position[0], position[1], position[2]});
                if (normals) {
                    const GfVec3f& normal = normals->perCorner ? normals->values[base + k]
                                                               : normals->values[triangle[k]];
                    source.normals->push_back({normal[0], normal[1], normal[2]});
                }
                if (st) {
                    const GfVec2f& uv =
                        st->perCorner ? st->values[base + k] : st->values[triangle[k]];
                    source.uvs->push_back({uv[0], uv[1]});
                }
            }
        }
        return source;
    }
    source.positions.reserve(points.size());
    for (const GfVec3f& point : points) {
        source.positions.push_back({point[0], point[1], point[2]});
    }
    source.triangles.reserve(triangles.size());
    for (const GfVec3i& triangle : triangles) {
        source.triangles.push_back({static_cast<std::uint32_t>(triangle[0]),
                                    static_cast<std::uint32_t>(triangle[1]),
                                    static_cast<std::uint32_t>(triangle[2])});
    }
    if (normals) {
        source.normals.emplace();
        source.normals->reserve(normals->values.size());
        for (const GfVec3f& normal : normals->values) {
            source.normals->push_back({normal[0], normal[1], normal[2]});
        }
    }
    if (st) {
        source.uvs.emplace();
        source.uvs->reserve(st->values.size());
        for (const GfVec2f& uv : st->values) {
            source.uvs->push_back({uv[0], uv[1]});
        }
    }
    return source;
}

/// What the companion material wears: a constant displayColor is used
/// directly, a vertex one is approximated by its first element — the
/// wire carries no per-vertex color, so the approximation is the
/// contract, not a stopgap — and anything else is the neutral default.
std::array<float, 3> _ReadDisplayColor(const HdSceneIndexPrim& prim) {
    const HdPrimvarSchema primvar =
        HdPrimvarsSchema::GetFromParent(prim.dataSource).GetPrimvar(HdTokens->displayColor);
    const HdSampledDataSourceHandle value = primvar.GetPrimvarValue();
    if (!value) {
        return kNeutralColor;
    }
    const TfToken interpolation = _TokenOr(primvar.GetInterpolation(), TfToken());
    if (interpolation != HdPrimvarSchemaTokens->constant &&
        interpolation != HdPrimvarSchemaTokens->vertex &&
        interpolation != HdPrimvarSchemaTokens->varying) {
        return kNeutralColor;
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (sampled.IsHolding<GfVec3f>()) {
        const GfVec3f color = sampled.UncheckedGet<GfVec3f>();
        return {color[0], color[1], color[2]};
    }
    if (sampled.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& colors = sampled.UncheckedGet<VtVec3fArray>();
        if (!colors.empty()) {
            return {colors[0][0], colors[0][1], colors[0][2]};
        }
    }
    return kNeutralColor;
}

/// The all-purpose binding target; empty when unbound. The flattening
/// stack has already resolved binding inheritance and the registered
/// resolving filter (sceneIndexPlugins.cpp) collapsed
/// material:binding:preview onto the all-purpose slot, so this one
/// read is the final word.
SdfPath _ReadBinding(const HdSceneIndexPrim& prim) {
    const HdPathDataSourceHandle path =
        HdMaterialBindingsSchema::GetFromParent(prim.dataSource).GetMaterialBinding().GetPath();
    return path ? path->GetTypedValue(0.0f) : SdfPath();
}

/// Resolved (flattened) visibility; absent means visible.
bool _ReadVisibility(const HdSceneIndexPrim& prim) {
    const HdBoolDataSourceHandle visibility =
        HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
    return visibility ? visibility->GetTypedValue(0.0f) : true;
}

/// The prim's flattened transform; identity when unauthored. For an
/// un-instanced mesh this is the world matrix; for a prim inside a point-
/// instancer prototype it is prototype-root relative (the prototype root
/// carries resetXformStack, so the instancer's placement is never folded
/// in here — that is ComputePlacements's job).
GfMatrix4d _WorldMatrix(const HdSceneIndexPrim& prim) {
    if (const HdMatrixDataSourceHandle matrix =
            HdXformSchema::GetFromParent(prim.dataSource).GetMatrix()) {
        return matrix->GetTypedValue(0.0f);
    }
    return GfMatrix4d(1.0);
}

/// A world matrix as Transform::Matrix — never decomposed. Gf composes
/// against row vectors, the wire against column vectors, so the rows are
/// the transpose's. Nullopt means the server would reject the placement
/// (see _Invertible).
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

/// Everything that reshapes the wire Mesh payload: topology plus the
/// three primvars Inline carries.
const HdDataSourceLocatorSet& _GeometryLocators() {
    static const HdDataSourceLocatorSet locators{
        HdMeshSchema::GetDefaultLocator(), HdPrimvarsSchema::GetPointsLocator(),
        HdPrimvarsSchema::GetDefaultLocator().Append(HdTokens->normals),
        HdPrimvarsSchema::GetDefaultLocator().Append(_tokens->st)};
    return locators;
}

const HdDataSourceLocator& _DisplayColorLocator() {
    static const HdDataSourceLocator locator =
        HdPrimvarsSchema::GetDefaultLocator().Append(HdTokens->displayColor);
    return locator;
}

/// Everything that reshapes the placements array: the flattened transform,
/// visibility, and instancedBy (the set of instancers that copy this mesh,
/// which turns a lone placement into an array and back).
const HdDataSourceLocatorSet& _PlacementLocators() {
    static const HdDataSourceLocatorSet locators{HdXformSchema::GetDefaultLocator(),
                                                 HdVisibilitySchema::GetDefaultLocator(),
                                                 HdInstancedBySchema::GetDefaultLocator()};
    return locators;
}

} // namespace

HdCenoteMeshPrim::HdCenoteMeshPrim(const SdfPath& path,
                                   const HdsiPrimManagingSceneIndexObserver* observer,
                                   cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live,
                                   std::shared_ptr<const MaterialRegistry> materials,
                                   std::shared_ptr<const InstancerRegistry> instancers)
    : _path(path), _name(path.GetString()), _material(path.GetString() + "/displayColor"),
      _pending(pending), _live(std::move(live)), _materials(std::move(materials)),
      _instancers(std::move(instancers)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: the previous translator still holds server objects.
        // Inherit its ledger — the reconcile below then updates in
        // place — and take the registry slot so its destructor, which
        // runs after this constructor, goes quietly.
        _sent = it->second->_sent;
        _instanceLive = it->second->_instanceLive;
        _binding = it->second->_binding;
        _wearsBinding = it->second->_wearsBinding;
        _warnedElement = it->second->_warnedElement;
        it->second = this;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _Withdraw();
        return;
    }
    _Reconcile(prim, _Dirt{.geometry = true, .color = true, .placement = true, .binding = true});
}

HdCenoteMeshPrim::~HdCenoteMeshPrim() {
    const auto it = _live->find(_path);
    if (it == _live->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _live->erase(it);
    _Withdraw();
}

void HdCenoteMeshPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                              const HdsiPrimManagingSceneIndexObserver* observer) {
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        return;
    }
    _Reconcile(prim, _Dirt{
                         .geometry = entry.dirtyLocators.Intersects(_GeometryLocators()),
                         .color = entry.dirtyLocators.Intersects(_DisplayColorLocator()),
                         .placement = entry.dirtyLocators.Intersects(_PlacementLocators()),
                         .binding = entry.dirtyLocators.Intersects(
                             HdMaterialBindingsSchema::GetDefaultLocator()),
                     });
}

void HdCenoteMeshPrim::_Reconcile(const HdSceneIndexPrim& prim, const _Dirt dirt) {
    bool born = false;
    if (dirt.geometry) {
        std::optional<cenote::wire::Inline> source = _ReadGeometry(_path, prim);
        if (!source) {
            _Withdraw();
            return;
        }
        _pending->ops.push_back(cenote::wire::MeshPatch{
            .name = _name, .source = cenote::wire::MeshSource{std::move(*source)}});
        if (!_sent) {
            _pending->ops.push_back(cenote::wire::MaterialPatch{
                .name = _material,
                .base_color =
                    cenote::wire::Constant<std::array<float, 3>>{_ReadDisplayColor(prim)}});
            _sent = true;
            born = true;
        }
    }
    if (!_sent) {
        // Nothing on the wire yet; a future geometry dirty re-enters.
        return;
    }
    if (dirt.color && !born) {
        _pending->ops.push_back(cenote::wire::MaterialPatch{
            .name = _material,
            .base_color = cenote::wire::Constant<std::array<float, 3>>{_ReadDisplayColor(prim)}});
    }
    if (born || dirt.binding) {
        // Materials flush ahead of meshes (the batching priorities in
        // observer.cpp), so a target this sync cannot see is truly not
        // there — absent from the stage or arriving in a later wave.
        const SdfPath binding = _ReadBinding(prim);
        if (binding != _binding) {
            _binding = binding;
            if (!_binding.IsEmpty() && !_Bindable()) {
                TF_WARN("<%s> binds <%s>, which is not a material on the wire; wearing "
                        "displayColor until it arrives",
                        _path.GetText(), _binding.GetText());
            }
        }
    }
    if (born || dirt.placement) {
        // Refresh the cached placement inputs, then place from them. A
        // later instancer poke replaces the placements from these same
        // inputs without touching the scene index.
        _visible = _ReadVisibility(prim);
        _protoXform = _WorldMatrix(prim);
        _instancedBy = _ReadInstancedBy(prim);
        _Place();
    }
    // Every reconcile ends by squaring the instance's wear with the
    // registry — a binding edit repoints here, and everything else
    // no-ops.
    ResolveBinding();
}

void HdCenoteMeshPrim::RecomposeInstancing() {
    if (_instancedBy.empty() || !_sent) {
        // A mesh no instancer copies ignores instancer pokes, and a mesh
        // whose geometry has not reached the wire has no instance to
        // place. The cached inputs are still current — only the instancer
        // moved — so _Place re-reads nothing.
        return;
    }
    _Place();
}

void HdCenoteMeshPrim::_Place() {
    std::optional<std::vector<cenote::wire::Transform>> placements = _ComposePlacements();
    if (!placements) {
        _WithdrawInstance();
        return;
    }
    if (!_instanceLive) {
        _wearsBinding = _Bindable();
        _pending->ops.push_back(cenote::wire::InstancePatch{
            .name = _name,
            .mesh = _name,
            .material = _wearsBinding ? _binding.GetString() : _material,
            .transforms = std::move(placements)});
        _instanceLive = true;
    } else {
        // The whole array replaces (D-073); the mesh and the binding
        // stay put.
        _pending->ops.push_back(
            cenote::wire::InstancePatch{.name = _name, .transforms = std::move(placements)});
    }
}

std::optional<std::vector<cenote::wire::Transform>> HdCenoteMeshPrim::_ComposePlacements() {
    if (!_visible) {
        return std::nullopt;
    }
    if (_instancedBy.empty()) {
        // Un-instanced: the mesh stands once at its own transform.
        const std::optional<cenote::wire::Matrix> matrix = _WireTransform(_protoXform);
        if (!matrix) {
            TF_WARN("<%s> has a non-invertible transform; instance %s", _path.GetText(),
                    _instanceLive ? "removed" : "not placed");
            return std::nullopt;
        }
        return std::vector<cenote::wire::Transform>{*matrix};
    }
    // Instanced: concatenate the placements every instancer contributes,
    // each composed as (this prim's own transform) · (the instancer chain).
    std::vector<cenote::wire::Transform> placements;
    for (const _Instancer& instancer : _instancedBy) {
        const auto it = _instancers->find(instancer.path);
        if (it == _instancers->end()) {
            // The instancer is not on the registry yet; park until its
            // birth pokes this mesh. An empty placement would say the
            // opposite — resident, placed nowhere — so honesty is nullopt.
            return std::nullopt;
        }
        for (const GfMatrix4d& chain : it->second->ComputePlacements(instancer.prototypeRoot)) {
            if (const std::optional<cenote::wire::Matrix> matrix =
                    _WireTransform(_protoXform * chain)) {
                placements.push_back(*matrix);
            } else if (!_warnedElement) {
                _warnedElement = true;
                TF_WARN("<%s> composes a non-invertible instance placement; the copies it would "
                        "make are dropped",
                        _path.GetText());
            }
        }
    }
    return placements;
}

std::vector<HdCenoteMeshPrim::_Instancer>
HdCenoteMeshPrim::_ReadInstancedBy(const HdSceneIndexPrim& prim) const {
    const HdInstancedBySchema schema = HdInstancedBySchema::GetFromParent(prim.dataSource);
    const HdPathArrayDataSourceHandle pathsSource = schema.GetPaths();
    if (!pathsSource) {
        return {};
    }
    const VtArray<SdfPath> paths = pathsSource->GetTypedValue(0.0f);
    VtArray<SdfPath> roots;
    if (const HdPathArrayDataSourceHandle rootsSource = schema.GetPrototypeRoots()) {
        roots = rootsSource->GetTypedValue(0.0f);
    }
    std::vector<_Instancer> instancers;
    instancers.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        // prototypeRoots runs parallel to paths; a missing entry leaves the
        // root empty, and ComputePlacements then matches no prototype —
        // placing nothing, the honest reading of malformed instancedBy.
        instancers.push_back({paths[i], i < roots.size() ? roots[i] : SdfPath()});
    }
    return instancers;
}

void HdCenoteMeshPrim::ResolveBinding() {
    if (!_instanceLive) {
        return;
    }
    const bool bindable = _Bindable();
    if (bindable == _wearsBinding) {
        return;
    }
    _pending->ops.push_back(cenote::wire::InstancePatch{
        .name = _name, .material = bindable ? _binding.GetString() : _material});
    _wearsBinding = bindable;
}

bool HdCenoteMeshPrim::_Bindable() const {
    if (_binding.IsEmpty()) {
        return false;
    }
    const auto it = _materials->find(_binding);
    return it != _materials->end() && it->second->Published();
}

void HdCenoteMeshPrim::_WithdrawInstance() {
    if (_instanceLive) {
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Instance, .name = _name});
        _instanceLive = false;
    }
}

void HdCenoteMeshPrim::_Withdraw() {
    _WithdrawInstance();
    if (_sent) {
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Mesh, .name = _name});
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Material, .name = _material});
        _sent = false;
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
