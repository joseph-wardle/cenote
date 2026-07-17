#include "instancerPrim.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/stringUtils.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/instancedBySchema.h"
#include "pxr/imaging/hd/instancerTopologySchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// Resolved (flattened) visibility; absent means visible. An invisible
/// instancer composes to nothing, the same reading hdEmbree gives it.
bool _ReadVisibility(const HdSceneIndexPrim& prim) {
    const HdBoolDataSourceHandle visibility =
        HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
    return visibility ? visibility->GetTypedValue(0.0f) : true;
}

/// The instancer's flattened transform; identity when unauthored. World
/// space for a top-level instancer, prototype-root relative for a nested
/// one — the parent chain places the latter.
GfMatrix4d _WorldMatrix(const HdSceneIndexPrim& prim) {
    if (const HdMatrixDataSourceHandle matrix =
            HdXformSchema::GetFromParent(prim.dataSource).GetMatrix()) {
        return matrix->GetTypedValue(0.0f);
    }
    return GfMatrix4d(1.0);
}

/// The value data source of one instance-rate primvar, or null when the
/// instancer does not carry it — every hydra:instance* input is optional
/// and defaults to the identity component.
HdSampledDataSourceHandle _InstancePrimvar(const HdSceneIndexPrim& prim, const TfToken& name) {
    return HdPrimvarsSchema::GetFromParent(prim.dataSource).GetPrimvar(name).GetPrimvarValue();
}

/// hydra:instanceTranslations / instanceScales as doubles, both precisions
/// admitted; empty when absent or oddly typed (read as all-identity).
std::vector<GfVec3d> _ReadVec3(const HdSampledDataSourceHandle& source) {
    if (!source) {
        return {};
    }
    const VtValue value = source->GetValue(0.0f);
    std::vector<GfVec3d> out;
    if (value.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& array = value.UncheckedGet<VtVec3fArray>();
        out.reserve(array.size());
        for (const GfVec3f& element : array) {
            out.emplace_back(element);
        }
    } else if (value.IsHolding<VtVec3dArray>()) {
        const VtVec3dArray& array = value.UncheckedGet<VtVec3dArray>();
        out.assign(array.begin(), array.end());
    }
    return out;
}

/// hydra:instanceRotations widened to double quaternions — usdImaging
/// emits half (orientations) or single (orientationsf) precision, and the
/// composition wants double. Empty when absent or oddly typed.
std::vector<GfQuatd> _ReadQuat(const HdSampledDataSourceHandle& source) {
    if (!source) {
        return {};
    }
    const VtValue value = source->GetValue(0.0f);
    std::vector<GfQuatd> out;
    if (value.IsHolding<VtQuathArray>()) {
        const VtQuathArray& array = value.UncheckedGet<VtQuathArray>();
        out.reserve(array.size());
        for (const GfQuath& element : array) {
            out.emplace_back(element);
        }
    } else if (value.IsHolding<VtQuatfArray>()) {
        const VtQuatfArray& array = value.UncheckedGet<VtQuatfArray>();
        out.reserve(array.size());
        for (const GfQuatf& element : array) {
            out.emplace_back(element);
        }
    } else if (value.IsHolding<VtQuatdArray>()) {
        const VtQuatdArray& array = value.UncheckedGet<VtQuatdArray>();
        out.assign(array.begin(), array.end());
    }
    return out;
}

/// hydra:instanceTransforms — the aggregated 4×4 form native instancing
/// emits. Empty when absent or oddly typed.
std::vector<GfMatrix4d> _ReadMatrix(const HdSampledDataSourceHandle& source) {
    if (!source) {
        return {};
    }
    const VtValue value = source->GetValue(0.0f);
    if (value.IsHolding<VtMatrix4dArray>()) {
        const VtMatrix4dArray& array = value.UncheckedGet<VtMatrix4dArray>();
        return {array.begin(), array.end()};
    }
    return {};
}

/// The four instance-rate transform channels, each indexed by instance
/// index; any absent channel is the identity for that component.
struct _Samplers {
    std::vector<GfVec3d> translations;
    std::vector<GfQuatd> rotations;
    std::vector<GfVec3d> scales;
    std::vector<GfMatrix4d> matrices;
};

_Samplers _ReadSamplers(const HdSceneIndexPrim& prim) {
    return {_ReadVec3(_InstancePrimvar(prim, HdInstancerTokens->instanceTranslations)),
            _ReadQuat(_InstancePrimvar(prim, HdInstancerTokens->instanceRotations)),
            _ReadVec3(_InstancePrimvar(prim, HdInstancerTokens->instanceScales)),
            _ReadMatrix(_InstancePrimvar(prim, HdInstancerTokens->instanceTransforms))};
}

/// One instance's placement, hdEmbree's exact order (instancer.cpp): the
/// instancer transform, then translate · rotate · scale · matrix
/// pre-multiplied in turn, so a point standing at the origin lands at
/// matrix · S · R · T · instancerXform (Gf's row-vector convention). A
/// channel the instancer omits, or an index past its end, is the identity.
GfMatrix4d _Compose(const _Samplers& samplers, const int index, const GfMatrix4d& instancerXform) {
    GfMatrix4d placement = instancerXform;
    const auto i = static_cast<std::size_t>(index);
    if (index >= 0 && i < samplers.translations.size()) {
        GfMatrix4d translate;
        translate.SetTranslate(samplers.translations[i]);
        placement = translate * placement;
    }
    if (index >= 0 && i < samplers.rotations.size()) {
        GfMatrix4d rotate;
        rotate.SetRotate(samplers.rotations[i]);
        placement = rotate * placement;
    }
    if (index >= 0 && i < samplers.scales.size()) {
        GfMatrix4d scale;
        scale.SetScale(samplers.scales[i]);
        placement = scale * placement;
    }
    if (index >= 0 && i < samplers.matrices.size()) {
        placement = samplers.matrices[i] * placement;
    }
    return placement;
}

/// The instancers that copy this instancer (its own nesting), paired with
/// the prototype root each copies it under; empty for a top-level
/// instancer.
std::vector<std::pair<SdfPath, SdfPath>> _ReadInstancedBy(const HdSceneIndexPrim& prim) {
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
    std::vector<std::pair<SdfPath, SdfPath>> parents;
    parents.reserve(paths.size());
    for (std::size_t i = 0; i < paths.size(); ++i) {
        parents.emplace_back(paths[i], i < roots.size() ? roots[i] : SdfPath());
    }
    return parents;
}

/// Whether a primvar name is a transform channel this translator consumes
/// or a motion-blur channel it drops in silence (velocity family).
bool _IsHandledPrimvar(const TfToken& name) {
    return name == HdInstancerTokens->instanceTranslations ||
           name == HdInstancerTokens->instanceRotations ||
           name == HdInstancerTokens->instanceScales ||
           name == HdInstancerTokens->instanceTransforms || name == HdTokens->velocities ||
           name == HdTokens->accelerations || name == HdTokens->angularVelocities;
}

/// The instance-rate primvars cenote cannot carry — per-instance shading
/// data (a color, a width) has no home when an instance is only a
/// placement wearing a shared material. Names them for the one warning.
std::string _UnsupportedPrimvars(const HdSceneIndexPrim& prim) {
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    std::vector<std::string> dropped;
    for (const TfToken& name : primvars.GetPrimvarNames()) {
        if (_IsHandledPrimvar(name)) {
            continue;
        }
        const HdPrimvarSchema primvar = primvars.GetPrimvar(name);
        const HdTokenDataSourceHandle interpolation = primvar.GetInterpolation();
        if (interpolation &&
            interpolation->GetTypedValue(0.0f) == HdPrimvarSchemaTokens->instance) {
            dropped.push_back(name.GetString());
        }
    }
    return TfStringJoin(dropped, ", ");
}

} // namespace

HdCenoteInstancerPrim::HdCenoteInstancerPrim(
    const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
    std::shared_ptr<Registry> instancers, std::shared_ptr<const HdCenoteMeshPrim::Registry> meshes)
    : _path(path), _observer(observer), _instancers(std::move(instancers)),
      _meshes(std::move(meshes)) {
    const auto [it, inserted] = _instancers->try_emplace(_path, this);
    if (!inserted) {
        // A resync: there is no server-side ledger to inherit — the latch
        // aside — only the registry slot to take, so the superseded
        // destructor pokes nobody.
        _warnedPrimvars = it->second->_warnedPrimvars;
        it->second = this;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        // A phantom add: nothing to read, but any mesh already awaiting
        // this instancer should recompose (and find it absent).
        _PokeMeshes();
        return;
    }
    _Refresh(prim);
}

HdCenoteInstancerPrim::~HdCenoteInstancerPrim() {
    const auto it = _instancers->find(_path);
    if (it == _instancers->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _instancers->erase(it);
    // The dependents must recompose without this instancer — they will
    // find it gone and park.
    _PokeMeshes();
}

void HdCenoteInstancerPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& /*entry*/,
                                   const HdsiPrimManagingSceneIndexObserver* observer) {
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _PokeMeshes();
        return;
    }
    // Every locator this translator reads — topology, the instance
    // primvars, the instancer's transform, its own instancedBy,
    // visibility — reshapes the placements the same way: the dependents
    // recompose wholesale. So any dirt at all pokes, no per-locator
    // surgery (the placements are a pure function of the whole prim).
    _Refresh(prim);
}

void HdCenoteInstancerPrim::_Refresh(const HdSceneIndexPrim& prim) {
    if (!_warnedPrimvars) {
        const std::string dropped = _UnsupportedPrimvars(prim);
        if (!dropped.empty()) {
            _warnedPrimvars = true;
            TF_WARN("<%s> carries the instance-rate primvar(s) %s, which cenote cannot vary per "
                    "copy; dropped",
                    _path.GetText(), dropped.c_str());
        }
    }
    _PokeMeshes();
}

void HdCenoteInstancerPrim::_PokeMeshes() const {
    for (const auto& [path, mesh] : *_meshes) {
        mesh->RecomposeInstancing();
    }
}

std::vector<GfMatrix4d>
HdCenoteInstancerPrim::ComputePlacements(const SdfPath& prototypeRoot) const {
    const HdSceneIndexPrim prim = _observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource || !_ReadVisibility(prim)) {
        // Gone or invisible: an invisible instancer places nothing, and
        // its whole subtree with it (nested chains compose empty).
        return {};
    }
    // ComputeInstanceIndicesForProto folds the mask in already: the result
    // is the surviving instance indices for this prototype, in order.
    HdInstancerTopologySchema topology = HdInstancerTopologySchema::GetFromParent(prim.dataSource);
    const VtIntArray indices = topology.ComputeInstanceIndicesForProto(prototypeRoot);
    if (indices.empty()) {
        return {};
    }
    const _Samplers samplers = _ReadSamplers(prim);
    const GfMatrix4d instancerXform = _WorldMatrix(prim);
    std::vector<GfMatrix4d> placements;
    placements.reserve(indices.size());
    for (const int index : indices) {
        placements.push_back(_Compose(samplers, index, instancerXform));
    }
    // Nesting (hdEmbree's cartesian product): each parent copy places a
    // whole set of this instancer's copies, child transform innermost, and
    // several parents concatenate. A top-level instancer skips all of this.
    const std::vector<std::pair<SdfPath, SdfPath>> parents = _ReadInstancedBy(prim);
    if (parents.empty()) {
        return placements;
    }
    std::vector<GfMatrix4d> nested;
    for (const auto& [parentPath, parentRoot] : parents) {
        const auto it = _instancers->find(parentPath);
        if (it == _instancers->end()) {
            // An ancestor not yet on the registry contributes no copies;
            // its birth will poke the leaf meshes to recompose.
            continue;
        }
        const std::vector<GfMatrix4d> above = it->second->ComputePlacements(parentRoot);
        nested.reserve(nested.size() + above.size() * placements.size());
        for (const GfMatrix4d& parent : above) {
            for (const GfMatrix4d& child : placements) {
                nested.push_back(child * parent);
            }
        }
    }
    return nested;
}

PXR_NAMESPACE_CLOSE_SCOPE
