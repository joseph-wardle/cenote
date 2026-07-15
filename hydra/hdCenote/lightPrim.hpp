// The distant light translator: one scene-index distantLight prim
// becomes one server Light, named by the prim path (D-108 pulls this
// slice of step 4 forward so the checkpoint frame is honestly lit).
// Direction is the world -Z of the flattened transform, irradiance is
// intensity·2^exposure·color, and the angle collapses to the delta per
// the locked floor. Like every translator it never sends: wire ops go
// to the delegate's pending ChangeSet, and Update() drains it. Removal
// is RAII — the prim-managing observer drops the handle and the
// destructor emits the Remove.
#pragma once

#include <map>
#include <memory>
#include <string>

#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteLightPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each prim path — the same
    /// resync tie-breaker the mesh translator carries (see
    /// HdCenoteMeshPrim::Registry): the newcomer inherits the ledger and
    /// the superseded destructor goes quietly.
    using Registry = std::map<SdfPath, HdCenoteLightPrim*>;

    /// Reads the prim at `path` from the observer's scene index and
    /// appends its add to `pending`. Both `pending` and `live` outlive
    /// every translator — the delegate and the factory own them.
    HdCenoteLightPrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                      cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live);

    /// RAII removal: Op::Remove for the Light if it is still up — unless
    /// a resync already handed the path to a successor.
    ~HdCenoteLightPrim() override;

private:
    /// Which lanes a change touches, from the dirty locators. Direction
    /// and irradiance both live on the one wire Light value, so params
    /// and transform share a lane.
    struct _Dirt {
        bool light;      //< params or transform → LightPatch.light
        bool visibility; //< visible ⟺ the Light exists (D-109's spirit)
    };

    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Brings the server-side Light in line with the prim. The validity
    /// mirror: the server rejects a zero-direction distant light, and a
    /// ChangeSet is atomic, so a degenerate transform withdraws the
    /// light instead of ever reaching the wire.
    void _Reconcile(const HdSceneIndexPrim& prim, _Dirt dirt, bool born);

    /// Removes the Light if it is currently on the server.
    void _Withdraw();

    const SdfPath _path;
    /// The server name: the prim path, Kind::Light.
    const std::string _name;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;

    // The ledger: what exists server-side right now.
    bool _sent = false; //< the Light is up
};

PXR_NAMESPACE_CLOSE_SCOPE
