// The UsdLux dome translator: dome lights contend for THE environment —
// the server renders at most one — and the winner becomes the one
// EnvironmentPatch, named by its prim path. Arbitration is by path
// (Q10): the lowest-path visible dome publishes, the rest park under one
// warning each, and a winner dying, hiding, or losing to a lower-path
// newcomer hands the slot over in the same atomic flush — never two
// environments on the server, never a black sky while a parked dome
// could light the scene. No dome at all is the boot state: no
// environment, the renderer's native black sky. Radiometry folds color,
// blackbody temperature, and intensity·2^exposure into the wire's tint,
// over the equirect image when texture:file qualifies (absolute,
// existing, .exr/.hdr, an equirect texture:format) and over the constant
// white sky when it does not. Placement composes the flattened world
// matrix over domeOffset — DomeLight_1's poleAxis alignment, served by
// usdImaging — with the 180° yaw between USD's equirect convention
// (image center at +Z, the OpenEXR latlong rule) and cenote's (−Z) baked
// innermost, so the wire stays in cenote's convention. Like every
// translator it never sends: wire ops go to the delegate's pending
// ChangeSet, and Update() drains it. Removal is RAII — the prim-managing
// observer drops the handle and the destructor emits the Remove, after
// handing the slot on.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteDomePrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each dome path — the same
    /// resync tie-breaker the mesh translator carries (see
    /// HdCenoteMeshPrim::Registry), and here also the arbitration order:
    /// the map sorts by path, so the first eligible entry is the winner.
    using Registry = std::map<SdfPath, HdCenoteDomePrim*>;

    /// Reads the prim at `path` from the observer's scene index and
    /// joins the arbitration — publishing to `pending` if it wins the
    /// slot. Both `pending` and `live` outlive every translator — the
    /// delegate and the factory own them.
    HdCenoteDomePrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                     cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live);

    /// RAII removal: Op::Remove for the Environment if this dome holds
    /// it, then the failover — the next-lowest eligible dome publishes
    /// in the same flush. Unless a resync already handed the path to a
    /// successor, who answers for it instead.
    ~HdCenoteDomePrim() override;

private:
    /// Which lanes a change touches, from the dirty locators. Params and
    /// transform both reshape the one payload — total re-read, total
    /// resend (D-113/D-115) — so they share a lane.
    struct _Dirt {
        bool light;      //< params or transform → the payload rebuilds
        bool visibility; //< visible ⟺ eligible for the slot
    };

    /// One-shot latches for the degradations that warn: with every edit
    /// re-reading everything, an unlatched warning would fire on every
    /// drag of an unrelated handle.
    struct _Warned {
        bool parked = false;  //< lost the slot to a lower path (resets on winning)
        bool texture = false; //< texture:file dropped (unreadable or not .exr/.hdr)
        bool format = false;  //< texture:format names a mapping the server lacks
    };

    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Re-reads eligibility and rebuilds the cached payload, then
    /// arbitrates. The cache is what lets a parked dome publish at
    /// failover — inside the winner's destructor — without another trip
    /// through the scene index.
    void _Refresh(const HdSceneIndexPrim& prim, _Dirt dirt, bool born);

    /// The dome's texture:file down to a path the environment can wear:
    /// absolute, existing, .exr/.hdr, and mapped equirect per
    /// texture:format — anything else warns once and leaves the sky
    /// constant. Unauthored is silent: a bare dome is the constant white
    /// sky colored by tint, by design.
    std::optional<std::string> _SkyImage(const HdContainerDataSourceHandle& light);

    /// Squares the whole registry with the one-environment rule: demotes
    /// any published non-winner, then publishes the winner — freshly
    /// when `fresh` is the winner itself (total resend), or because the
    /// slot just opened — and parks the rest under their warnings. Every
    /// hand-over lands demotion and succession in one atomic flush.
    void _Arbitrate(const HdCenoteDomePrim* fresh);

    /// Appends the cached EnvironmentPatch and takes the slot.
    void _Publish();

    /// Removes the Environment if this dome currently holds it.
    void _Withdraw();

    const SdfPath _path;
    /// The server name: the prim path, Kind::Environment.
    const std::string _name;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;

    // The ledger: this dome's standing, right now.
    bool _eligible = false;   //< readable and visible — a contender
    bool _published = false;  //< holds the environment slot server-side
    bool _degenerate = false; //< placement fell back to the bare yaw (gates the warning)
    /// The payload as of the last read, ready for failover.
    cenote::wire::EnvironmentPatch _payload;
    _Warned _warned;
};

PXR_NAMESPACE_CLOSE_SCOPE
