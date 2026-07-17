// The UsdLux light translator: one scene-index light prim becomes the
// server objects its wire spelling needs. Distant lights — usdview's
// camera light among them (D-108) — and spheres marked treatAsPoint
// collapse to one delta Light; rect, disk, sphere, and cylinder lights
// synthesize the area triple — a mesh with the dimensions baked into its
// vertices, a black-absorber material carrying the emission, and a
// camera-invisible instance under the flattened world matrix — all three
// named by the prim path. The renderer keeps no analytic light shapes:
// emissive triangles are its native area light, ReSTIR's whole reason,
// so synthesis is the honest translation, not a workaround. Radiometry
// follows the UsdLux luminous spec: intensity·2^exposure with color and
// blackbody temperature as the tint, `normalize` dividing by world-space
// area (or the sphere/distant closed forms). Like every translator it
// never sends: wire ops go to the delegate's pending ChangeSet, and
// Update() drains it. Removal is RAII — the prim-managing observer drops
// the handle and the destructor emits the Removes.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/hd/dataSource.h"
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

    /// Reads the prim at `path` — of light `type`, one of the five the
    /// observer dispatches — from the observer's scene index and appends
    /// its add to `pending`. Both `pending` and `live` outlive every
    /// translator — the delegate and the factory own them.
    HdCenoteLightPrim(const SdfPath& path, const TfToken& type,
                      const HdsiPrimManagingSceneIndexObserver* observer,
                      cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live);

    /// RAII removal: Op::Remove for whatever spelling is still up —
    /// unless a resync already handed the path to a successor.
    ~HdCenoteLightPrim() override;

private:
    /// What the server holds for this prim right now: nothing, one delta
    /// Light, or the area triple. A live edit can flip the spelling — a
    /// sphere toggling treatAsPoint — and the flip withdraws the old
    /// objects and sends the new in the same atomic flush.
    enum class _Spelling { None, Delta, Area };

    /// Which lanes a change touches, from the dirty locators. Params and
    /// transform both reshape the one payload — total re-read, total
    /// resend (D-113/D-115) — so they share a lane.
    struct _Dirt {
        bool light;      //< params or transform → the whole spelling resends
        bool visibility; //< visible ⟺ the objects exist (D-109's spirit)
    };

    /// One-shot latches for the approximations that degrade under a
    /// warning: with every edit re-reading everything, an unlatched
    /// warning would fire on every drag of an unrelated handle.
    struct _Warned {
        bool diffuse = false;  //< diffuse ≠ 1 ignored
        bool specular = false; //< specular ≠ 1 ignored
        bool texture = false;  //< texture:file dropped (unreadable or non-LDR)
        bool tint = false;     //< chromatic tint folded to luminance over a texture
    };

    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Brings the server-side objects in line with the prim: reads the
    /// whole light fresh, withdraws a stale spelling, and resends. The
    /// validity mirror: the server rejects degenerate directions and
    /// non-invertible placements, and a ChangeSet is atomic, so a
    /// degenerate light withdraws instead of ever reaching the wire.
    void _Reconcile(const HdSceneIndexPrim& prim, _Dirt dirt, bool born);

    /// The area triple's black absorber (amendment C): only emission
    /// set, carrying the UsdLux radiometry — with the rect's texture
    /// worn as the emission map when it qualifies (Q7).
    cenote::wire::MaterialPatch _EmissionMaterial(const HdContainerDataSourceHandle& light,
                                                  GfVec3f tint, float luminance);

    /// The rect's texture:file down to a path the emission slot can
    /// wear: absolute, existing, and LDR — anything else warns once and
    /// keeps the emission constant. Unauthored is silent.
    std::optional<std::string> _TextureFile(const HdContainerDataSourceHandle& light);

    /// Removes whatever is currently on the server — for the area triple
    /// the instance first, then the material, then the mesh.
    void _Withdraw();

    const SdfPath _path;
    /// The server name, shared by every object this prim spells (they
    /// differ by Kind): the prim path.
    const std::string _name;
    /// The scene-index prim type the factory dispatched on; the branch
    /// every reconcile takes.
    const TfToken _type;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;

    // The ledger: what exists server-side right now.
    _Spelling _sent = _Spelling::None;
    _Warned _warned;
};

PXR_NAMESPACE_CLOSE_SCOPE
