// The geometry translator: one scene-index geometry prim becomes three
// server objects — the payload (named by the prim path), an Instance (its
// placement, same name), and a <primPath>/displayColor Material (the
// permanent fallback: published unconditionally, worn whenever the prim's
// bound material is not on the wire). The instance wears the all-purpose
// binding when the material registry has it live, and the material
// lifecycle hooks call ResolveBinding so a late or dying material
// repoints the instance without any Hydra dirt.
// Translators never send: they append wire ops to the delegate's
// pending ChangeSet, and Update() drains it. Removal is RAII — the
// prim-managing observer drops the handle and the destructor emits the
// Removes.
//
// Two prim types come through here, and only two things separate them:
// how the payload is read (a triangulated mesh cage, or BasisCurves
// cells passed through verbatim for the server to evaluate) and which
// wire kind carries it. Everything else is the same work on one registry.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteMaterialPrim;
class HdCenoteInstancerPrim;

class HdCenoteGeometryPrim final : public HdsiPrimManagingSceneIndexObserver::PrimBase {
public:
    /// Which translator currently answers for each prim path. A resync
    /// hands the managing observer a fresh translator *before* the old
    /// handle is destroyed (map assignment order), so bare RAII would
    /// emit the old prim's Removes after the new prim's adds and tear
    /// down the freshly patched objects. The registry breaks the tie:
    /// the newcomer inherits the ledger and the superseded destructor
    /// goes quietly. Notices arrive serially (Flush runs inside
    /// Update()), so the map needs no lock.
    using Registry = std::map<SdfPath, HdCenoteGeometryPrim*>;

    /// The material translators' registry — HdCenoteMaterialPrim::
    /// Registry, spelled out here because the material header includes
    /// this one (its lifecycle hooks call ResolveBinding); it
    /// static_asserts that the two spellings stay the same type.
    using MaterialRegistry = std::map<SdfPath, HdCenoteMaterialPrim*>;

    /// The instancer translators' registry — HdCenoteInstancerPrim::
    /// Registry, spelled out here for the same reason (the instancer
    /// header includes this one, to poke geometry); its .cpp static_asserts
    /// that the two spellings stay the same type. An instanced prim looks
    /// its instancers up here to compose its placements.
    using InstancerRegistry = std::map<SdfPath, HdCenoteInstancerPrim*>;

    /// Reads the prim at `path` from the observer's scene index and
    /// appends its adds to `pending`. `primType` picks which payload is
    /// read — mesh or basisCurves, the two the factory routes here.
    /// `pending`, `live`, `materials`, and `instancers` all outlive every
    /// translator — the delegate and the factory own them.
    HdCenoteGeometryPrim(const SdfPath& path, const TfToken& primType,
                         const HdsiPrimManagingSceneIndexObserver* observer,
                         cenote::wire::ChangeSet* pending, std::shared_ptr<Registry> live,
                         std::shared_ptr<const MaterialRegistry> materials,
                         std::shared_ptr<const InstancerRegistry> instancers);

    /// RAII removal: Op::Remove for everything this translator still
    /// holds server-side — unless a resync already handed the path to
    /// a successor.
    ~HdCenoteGeometryPrim() override;

    /// Repoints the instance at whatever it should wear right now — the
    /// bound material when it is live on the wire, the companion
    /// otherwise — and no-ops when nothing changed. The material
    /// lifecycle hooks call this on birth and death (Hydra emits no
    /// binding dirt for either,  and every reconcile ends here.
    void ResolveBinding();

    /// Recomposes the placements array from the cached inputs (its own
    /// transform and instancedBy) against the current instancer state, and
    /// resends the instance. An instancer translator calls this when it
    /// changes — Hydra dirties the instancer, not the prototype prims it
    /// moves — so a poke is the only signal an instanced prim gets. A prim
    /// that no instancer instances returns at once.
    void RecomposeInstancing();

private:
    /// Which server objects a change touches, from the dirty locators —
    /// the wire patch fields follow 1:1.
    struct _Dirt {
        bool geometry;  //< the payload's own locators → MeshPatch/CurvesPatch.source
        bool color;     //< displayColor → companion MaterialPatch.base_color
        bool placement; //< transform, visibility, or instancedBy → the placements array
        bool binding;   //< resolved materialBindings → InstancePatch.material
    };

    /// One instancer that instances this prim: the instancer's path and
    /// the prototype root it copies this prim under (from instancedBy).
    struct _Instancer {
        SdfPath path;
        SdfPath prototypeRoot;
    };

    void _Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                const HdsiPrimManagingSceneIndexObserver* observer) override;

    /// Brings the server-side objects in line with the prim, touching
    /// only what `dirt` marks. The one subtlety is validity: a ChangeSet
    /// is atomic server-side, so an op the server would reject (an empty
    /// mesh, a periodic curve, a non-invertible placement) must never be
    /// emitted — the object is withdrawn or the placement dropped instead.
    void _Reconcile(const HdSceneIndexPrim& prim, _Dirt dirt);

    /// (Re)places the instance from the cached placement inputs, shared by
    /// the reconcile that refreshes those inputs and the instancer poke
    /// that leaves them be: composes the placements, then creates,
    /// replaces, or withdraws the instance to match.
    void _Place();

    /// The per-copy placements this prim should stand at, or nullopt to
    /// withdraw the instance (invisible, a degenerate lone transform, or
    /// an instancer not yet on the registry — honest absence). An empty
    /// vector is legal and distinct: resident, placed nowhere (fully
    /// masked). Degenerate elements of an instanced array drop under a
    /// one-shot warning; the atomic flush must never carry one.
    std::optional<std::vector<cenote::wire::Transform>> _ComposePlacements();

    /// The instancers that copy this prim, paired with the prototype root
    /// each copies it under; empty when the prim stands on its own.
    std::vector<_Instancer> _ReadInstancedBy(const HdSceneIndexPrim& prim) const;

    /// Removes everything currently on the server, in reference order.
    void _Withdraw();

    /// Removes just the instance, leaving the payload and companion up.
    void _WithdrawInstance();

    /// True when the bound material is live on the wire right now —
    /// registered and published, so an instance may reference it.
    bool _Bindable() const;

    /// The wire kind this prim's payload travels as.
    cenote::wire::Kind _Kind() const {
        return _curves ? cenote::wire::Kind::Curves : cenote::wire::Kind::Mesh;
    }

    const SdfPath _path;
    /// Whether this prim is BasisCurves rather than a Mesh.
    const bool _curves;
    /// Server names: payload and instance share the prim path; the
    /// companion material appends /displayColor. Kinds keep them apart.
    const std::string _name;
    const std::string _material;
    cenote::wire::ChangeSet* const _pending;
    const std::shared_ptr<Registry> _live;
    const std::shared_ptr<const MaterialRegistry> _materials;
    const std::shared_ptr<const InstancerRegistry> _instancers;

    // The ledger: what exists server-side right now.
    /// The kind of payload standing under _name, with its companion
    /// Material; empty when neither is up. The *kind* rather than a flag,
    /// because a resync inherits this ledger from the translator it
    /// supersedes and a prim may change type under one path — so the
    /// payload on the wire is not always the kind this translator
    /// publishes, and only a withdrawal that names what was created can
    /// be right.
    std::optional<cenote::wire::Kind> _payload;
    bool _instanceLive = false; //< the Instance is placed
    /// The all-purpose binding target from the last sync; empty when
    /// unbound.
    SdfPath _binding;
    /// Whether the instance wears _binding rather than the companion;
    /// meaningful only while _instanceLive.
    bool _wearsBinding = false;

    // The cached placement inputs, refreshed on every placement dirt and
    // reused verbatim when an instancer poke recomposes: the instancer
    // moved, not this prim, so re-reading the scene index would only
    // recover what is already here.
    bool _visible = true;        //< resolved visibility
    GfMatrix4d _protoXform{1.0}; //< own transform (prototype-root relative when instanced)
    std::vector<_Instancer>
        _instancedBy{}; //< the instancers that copy this prim; empty = un-instanced
    /// One-shot latch: a degenerate placement element has been named.
    bool _warnedElement = false;
    /// One-shot latch: this curves prim's dropped normals have been
    /// named. Curves sweep round tubes, so an authored orientation has
    /// nowhere to land.
    bool _warnedNormals = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
