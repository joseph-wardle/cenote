// The UsdRenderSettings translator: the stage's own opinion of how it
// wants to be rendered, which is the opinion that outranks the host's.
//
// A stage may carry any number of RenderSettings prims — a lookdev one, a
// final-frame one, one per shot — and Hydra names at most one of them
// active, through the scene globals (`usdrecord --renderSettingsPrimPath`,
// the `renderSettingsPrimPath` stage metadatum, usdview's context menu).
// So this translator arbitrates like the dome translator does, but the
// arbiter is elsewhere: each prim simply reads its own `active` flag and
// the one that finds it true claims the slot. Nothing else can be true at
// the same time, because the flag is computed by comparing against a
// single path.
//
// It puts nothing on the wire. What it publishes is a resolution — the
// same {patch, warnings} the settings map resolves to — into a slot the
// delegate reads at drain, where the two are merged in one place under one
// precedence rule (renderSettings.hpp). Appending an op here instead would
// have made precedence a question of who ran first in the flush, which is
// not a thing to make load-bearing.
//
// The keys arrive already filtered to `cenote:` by the filtering scene
// index the delegate registers (sceneIndexPlugins.cpp), so anything left
// that is not one of ours is a typo, and says so.
#pragma once

#include <memory>

#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "renderSettings.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteSettingsPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// The one active prim's contribution, shared between every settings
    /// translator and the delegate. `version` is what the delegate polls:
    /// the base class has its own counter for the settings map, and this
    /// is the same idea for the source Hydra has no counter for. It starts
    /// at 0 and the delegate starts having sent 0, so a stage with no
    /// render settings prim never sends on its account.
    struct Active {
        /// Whose resolution `resolved` is — identity, not just a flag, so
        /// a hand-over and a resync both stay unambiguous about who may
        /// release the slot.
        const HdCenoteSettingsPrim* owner = nullptr;
        HdCenoteResolvedSettings resolved;
        unsigned int version = 0;
    };

    /// Reads the prim at `path` from the observer's scene index and claims
    /// the slot if it is the active one. `active` outlives every
    /// translator — the delegate owns it.
    HdCenoteSettingsPrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                         std::shared_ptr<Active> active);

    /// Hands the slot back if this prim holds it, so a deleted or
    /// deactivated settings prim returns the host's map to authority in
    /// the same flush.
    ~HdCenoteSettingsPrim() override;

private:
    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Re-reads `active` and the namespaced settings, then claims or
    /// releases. Total re-read on any dirt in the renderSettings locator:
    /// the payload is three numbers, and a partial one would only be a
    /// second thing to keep true.
    void _Refresh(const HdSceneIndexPrim& prim);

    /// Takes the slot and bumps the version.
    void _Claim(HdCenoteResolvedSettings resolved);

    /// Gives the slot up, if it is this prim's to give.
    void _Release();

    const SdfPath _path;
    const std::shared_ptr<Active> _active;
};

PXR_NAMESPACE_CLOSE_SCOPE
