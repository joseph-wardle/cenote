// The instancer translator: unlike every other prim, it puts nothing on
// the wire. cenote has no instancer object — an instance is geometry plus a
// placements array — so a Hydra instancer becomes pure arithmetic
// the geometry translators consume. Its whole job is ComputePlacements: given
// one of its prototypes, return the world-space placement matrices that
// prototype's prims should be copied to, with the instance-rate transforms
// (the hydra:instance* primvars), the instancer's own transform, the
// topology mask, and any parent instancers all folded in. A prim inside a
// prototype reads its own instancedBy, asks each named instancer here, and
// authors the concatenated array.
//
// Because an instancer edit dirties the instancer prim but not the
// prototype prims it moves (their flattened transform is prototype-root
// relative and never sees the instancer), the translator pokes its
// dependents the way a material does: birth, edit, and death all
// walk the geometry registry so every instanced prim recomposes. Removal is
// RAII, but there is nothing to remove server-side — the destructor only
// pokes.
#pragma once

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "geometryPrim.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteInstancerPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each instancer path — the
    /// same resync tie-breaker the other translators carry (see
    /// HdCenoteGeometryPrim::Registry). There is no server-side ledger to
    /// inherit; the newcomer only takes the slot so the superseded
    /// destructor pokes nobody.
    using Registry = std::map<SdfPath, HdCenoteInstancerPrim*>;

    /// Reads the instancer at `path` and registers as its transform
    /// provider. `instancers` is this registry (self-registration and the
    /// walk up to parent instancers); `geometry` is the geometry translators'
    /// registry, poked whenever this instancer changes. Both outlive every
    /// translator — the factory owns them. The observer back-pointer lets
    /// ComputePlacements read this prim on demand, when a dependent prim
    /// asks; it outlives the prims it manages.
    HdCenoteInstancerPrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                          std::shared_ptr<Registry> instancers,
                          std::shared_ptr<const HdCenoteGeometryPrim::Registry> geometry);

    /// RAII: nothing to withdraw, but the dependents must recompose
    /// without this instancer — unless a resync already handed the path to
    /// a successor.
    ~HdCenoteInstancerPrim() override;

    /// The world-space placements for the prototype rooted at
    /// `prototypeRoot`: one matrix per surviving instance, with the
    /// instance-rate transforms, this instancer's transform, the mask, and
    /// any parent instancers folded in — hdEmbree's exact composition. The
    /// caller pre-multiplies each prototype prim's own prototype-root
    /// relative transform. Empty when this instancer (or an ancestor) is
    /// invisible or masks the prototype away.
    std::vector<GfMatrix4d> ComputePlacements(const SdfPath& prototypeRoot) const;

private:
    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Warns once for each instance-rate primvar cenote cannot carry (any
    /// beyond the four transform ones and the silently-dropped velocity
    /// family), then tells every instanced prim to recompose. The read
    /// happens here, off the dependents' path, so ComputePlacements
    /// stays a pure function.
    void _Refresh(const HdSceneIndexPrim& prim);

    /// Walks the geometry registry so every instanced prim recomposes.
    /// Broad on purpose — an instanced prim reached through a chain of
    /// instancers recomposes from scratch, so poking every one is what
    /// makes nested edits correct without tracking who nests whom. The
    /// un-instanced prims it also reaches return immediately.
    void _PokeGeometry() const;

    const SdfPath _path;
    const HdsiPrimManagingSceneIndexObserver* const _observer;
    const std::shared_ptr<Registry> _instancers;
    const std::shared_ptr<const HdCenoteGeometryPrim::Registry> _geometry;

    /// One-shot latch: the unsupported instance-rate primvars have been
    /// named. A resync inherits it — a resync is no new approximation.
    bool _warnedPrimvars = false;
};

// The geometry header spells this map out (it cannot include this header —
// the include runs the other way); the two spellings must not drift.
static_assert(
    std::is_same_v<HdCenoteGeometryPrim::InstancerRegistry, HdCenoteInstancerPrim::Registry>);

PXR_NAMESPACE_CLOSE_SCOPE
