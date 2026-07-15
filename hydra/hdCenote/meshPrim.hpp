// The mesh translator: one scene-index mesh prim becomes three server
// objects — a Mesh (the triangulated payload, named by the prim path),
// an Instance (its placement, same name), and a <primPath>/displayColor
// Material (what an unbound surface wears; in step 1 every mesh is
// unbound, since material translation lands in step 3). Translators
// never send: they append wire ops to the delegate's pending ChangeSet,
// and Update() drains it. Removal is RAII — the prim-managing observer
// drops the handle and the destructor emits the Removes.
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

class HdCenoteMeshPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each prim path. A resync
    /// hands the managing observer a fresh translator *before* the old
    /// handle is destroyed (map assignment order), so bare RAII would
    /// emit the old prim's Removes after the new prim's adds and tear
    /// down the freshly patched objects. The registry breaks the tie:
    /// the newcomer inherits the ledger and the superseded destructor
    /// goes quietly. Notices arrive serially (Flush runs inside
    /// Update()), so the map needs no lock.
    using Registry = std::map<SdfPath, HdCenoteMeshPrim*>;

    /// Reads the prim at `path` from the observer's scene index and
    /// appends its adds to `pending`. Both `pending` and `live` outlive
    /// every translator — the delegate and the factory own them.
    HdCenoteMeshPrim(const SdfPath& path, const HdsiPrimManagingSceneIndexObserver* observer,
                     cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live);

    /// RAII removal: Op::Remove for everything this translator still
    /// holds server-side — unless a resync already handed the path to
    /// a successor.
    ~HdCenoteMeshPrim() override;

private:
    /// Which server objects a change touches, from the dirty locators —
    /// the wire patch fields follow 1:1.
    struct _Dirt {
        bool geometry;   //< topology or points/normals/st → MeshPatch.source
        bool color;      //< displayColor → companion MaterialPatch.base_color
        bool xform;      //< flattened matrix → InstancePatch.transform
        bool visibility; //< visible ⟺ the instance exists (D-109)
    };

    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Brings the server-side objects in line with the prim, touching
    /// only what `dirt` marks. The one subtlety is validity: a ChangeSet
    /// is atomic server-side, so an op the server would reject (an empty
    /// mesh, a non-invertible placement) must never be emitted — the
    /// object is withdrawn or the instance suppressed instead.
    void _Reconcile(const HdSceneIndexPrim& prim, _Dirt dirt);

    /// Removes everything currently on the server, in reference order.
    void _Withdraw();

    const SdfPath _path;
    /// Server names: mesh and instance share the prim path; the
    /// companion material appends /displayColor. Kinds keep them apart.
    const std::string _name;
    const std::string _material;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;

    // The ledger: what exists server-side right now.
    bool _sent = false;         //< the Mesh and companion Material are up
    bool _instanceLive = false; //< the Instance is placed
};

PXR_NAMESPACE_CLOSE_SCOPE
