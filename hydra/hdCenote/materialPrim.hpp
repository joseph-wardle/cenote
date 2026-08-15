// The material translator: one scene-index material prim becomes one
// server Material, named by the prim path. A UsdPreviewSurface surface
// maps input by input onto the wire's OpenPBR fields; any other surface
// — MaterialX, glslfx, none at all — wears OpenPBR defaults under one
// warning. Every sync sends a total patch, each mapped field explicitly
// set, so the wire material is a pure function of the network and a
// de-authored input resets its field instead of stranding stale state.
// The warn policy throughout: silent where the core's behavior is the
// authored behavior, one warning naming the material and input where
// fidelity is lost, and never an op the server would reject. Birth and
// death walk the geometry registry: Hydra emits no binding dirt for
// either, so each bound prim repoints its own instance — onto this
// material when it appears, back to its companion before the Remove
// lands. Like every translator it never sends: wire ops go to the
// delegate's pending ChangeSet, and Update() drains it. Removal is RAII
// — the prim-managing observer drops the handle and the destructor
// emits the Remove.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "geometryPrim.hpp"
#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteMaterialPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each prim path — the same
    /// resync tie-breaker the geometry translator carries (see
    /// HdCenoteGeometryPrim::Registry): the newcomer inherits the ledger and
    /// the superseded destructor goes quietly.
    using Registry = std::map<SdfPath, HdCenoteMaterialPrim*>;

    /// Reads the prim at `path` from the observer's scene index and
    /// appends its add to `pending`. `pending`, `live`, and `geometry`
    /// all outlive every translator — the delegate and the factory own
    /// them.
    HdCenoteMaterialPrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                         cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live,
                         std::shared_ptr<const HdCenoteGeometryPrim::Registry> geometry);

    /// RAII removal: Op::Remove for the Material if it is still up —
    /// unless a resync already handed the path to a successor.
    ~HdCenoteMaterialPrim() override;

    /// Whether the Material is on the wire right now — what a prim's
    /// bindable check reads. False only for the pathological birth that
    /// found no data source; a readable network always publishes.
    bool Published() const { return _sent; }

private:
    /// Any dirty under the material locator triggers the full re-read —
    /// the network is one lane, and the total patch needs no per-field
    /// locator surgery.
    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Re-reads the whole network and appends the total MaterialPatch.
    /// A material is always publishable — an unmappable network merely
    /// wears defaults — so unlike geometry and light there is no validity
    /// branch: the delegate pre-checks every texture path it forwards.
    void _Reconcile(const HdSceneIndexPrim& prim);

    /// Removes the Material if it is currently on the server — after
    /// every bound prim has repointed to its companion, so the Remove
    /// and the repoints land in one flush and nothing dangles.
    void _Withdraw();

    /// The lifecycle walk: every prim squares its instance's wear with
    /// the registries via ResolveBinding (a no-op for the unaffected).
    void _RepointGeometry();

    const SdfPath _path;
    /// The server name: the prim path, Kind::Material.
    const std::string _name;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;
    const std::shared_ptr<const HdCenoteGeometryPrim::Registry> _geometry;

    // The ledger: what exists server-side right now.
    bool _sent = false; //< the Material is up
};

// The geometry header spells this map out (it cannot include this header —
// the include runs the other way); the two spellings must not drift.
static_assert(
    std::is_same_v<HdCenoteGeometryPrim::MaterialRegistry, HdCenoteMaterialPrim::Registry>);

PXR_NAMESPACE_CLOSE_SCOPE
