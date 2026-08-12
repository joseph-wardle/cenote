#include "observer.hpp"

#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/tokens.h"

#include <utility>

#include "domePrim.hpp"
#include "instancerPrim.hpp"
#include "lightPrim.hpp"
#include "materialPrim.hpp"
#include "meshPrim.hpp"
#include "settingsPrim.hpp"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// The batching index orders each flush by this priority: materials and
/// instancers first, everything else after. Not for the server's sake —
/// it merges a whole ChangeSet and validates the result, so forward
/// references within one flush are already legal — but for the mesh
/// translator's: it reads both registries at sync. With materials ahead a
/// newborn instance wears its bound material from its first op; with
/// instancers ahead an instanced mesh composes its placements from a
/// registry already populated, so the common case needs no poke. (Cross-
/// flush arrivals still do: the material and instancer birth hooks poke.
/// Removals flush before everything regardless of priority; the death
/// hooks cover those.)
class _PriorityFunctor final
    : public HdsiPrimTypeNoticeBatchingSceneIndex::PrimTypePriorityFunctor {
public:
    size_t GetPriorityForPrimType(const TfToken& primType) const override {
        return primType == HdPrimTypeTokens->material || primType == HdPrimTypeTokens->instancer
                   ? 0
                   : 1;
    }
    size_t GetNumPriorities() const override { return 2; }
};

/// Hands each added prim its translator by type — and every type it
/// does not know gets null, which the observer simply never tracks: how
/// unknown stays non-fatal forever. Mesh prims (implicits included,
/// already converted by the registered filter stack) get the mesh
/// translator; instancers get the instancer translator, which puts
/// nothing on the wire and only feeds the meshes their placements; five
/// of the six UsdLux types the wire can spell — distant (usdview's camera
/// light among them), rect, disk, sphere, cylinder — get the
/// light translator; domes get their own, which arbitrates the one
/// environment slot among themselves (usdview's dome-toggle dome contends
/// like any other); materials get the material translator; render settings
/// prims get the settings translator, which puts nothing on the wire and
/// only publishes the active one's resolution for the delegate to merge.
class _PrimFactory final : public HdsiPrimManagingSceneIndexObserver::PrimFactoryBase {
public:
    _PrimFactory(cenote::wire::ChangeSet* pending,
                 std::shared_ptr<HdCenoteSettingsPrim::Active> settings)
        : _pending(pending), _settings(std::move(settings)),
          _meshes(std::make_shared<HdCenoteMeshPrim::Registry>()),
          _lights(std::make_shared<HdCenoteLightPrim::Registry>()),
          _domes(std::make_shared<HdCenoteDomePrim::Registry>()),
          _materials(std::make_shared<HdCenoteMaterialPrim::Registry>()),
          _instancers(std::make_shared<HdCenoteInstancerPrim::Registry>()) {}

    HdsiPrimManagingSceneIndexObserver::PrimBaseHandle
    CreatePrim(const HdSceneIndexObserver::AddedPrimEntry& entry,
               const HdsiPrimManagingSceneIndexObserver* observer) override {
        if (entry.primType == HdPrimTypeTokens->mesh) {
            return std::make_shared<HdCenoteMeshPrim>(entry.primPath, observer, _pending, _meshes,
                                                      _materials, _instancers);
        }
        if (entry.primType == HdPrimTypeTokens->instancer) {
            return std::make_shared<HdCenoteInstancerPrim>(entry.primPath, observer, _instancers,
                                                           _meshes);
        }
        if (entry.primType == HdPrimTypeTokens->distantLight ||
            entry.primType == HdPrimTypeTokens->rectLight ||
            entry.primType == HdPrimTypeTokens->diskLight ||
            entry.primType == HdPrimTypeTokens->sphereLight ||
            entry.primType == HdPrimTypeTokens->cylinderLight) {
            return std::make_shared<HdCenoteLightPrim>(entry.primPath, entry.primType, observer,
                                                       _pending, _lights);
        }
        if (entry.primType == HdPrimTypeTokens->domeLight) {
            return std::make_shared<HdCenoteDomePrim>(entry.primPath, observer, _pending, _domes);
        }
        if (entry.primType == HdPrimTypeTokens->material) {
            return std::make_shared<HdCenoteMaterialPrim>(entry.primPath, observer, _pending,
                                                          _materials, _meshes);
        }
        if (entry.primType == HdPrimTypeTokens->renderSettings) {
            return std::make_shared<HdCenoteSettingsPrim>(entry.primPath, observer, _settings);
        }
        return nullptr;
    }

private:
    /// The delegate's pending ChangeSet, shared with every translator.
    cenote::wire::ChangeSet* _pending;
    /// The delegate's slot for the active render settings prim.
    std::shared_ptr<HdCenoteSettingsPrim::Active> _settings;
    /// The translators' resync tie-breakers (see each Registry's doc).
    std::shared_ptr<HdCenoteMeshPrim::Registry> _meshes;
    std::shared_ptr<HdCenoteLightPrim::Registry> _lights;
    std::shared_ptr<HdCenoteDomePrim::Registry> _domes;
    std::shared_ptr<HdCenoteMaterialPrim::Registry> _materials;
    std::shared_ptr<HdCenoteInstancerPrim::Registry> _instancers;
};

} // namespace

HdCenoteObserver::HdCenoteObserver(HdSceneIndexBaseRefPtr const& terminal,
                                   cenote::wire::ChangeSet* pending,
                                   std::shared_ptr<HdCenoteSettingsPrim::Active> settings)
    : _batching(HdsiPrimTypeNoticeBatchingSceneIndex::New(
          terminal, HdRetainedContainerDataSource::New(
                        HdsiPrimTypeNoticeBatchingSceneIndexTokens->primTypePriorityFunctor,
                        HdRetainedTypedSampledDataSource<
                            HdsiPrimTypeNoticeBatchingSceneIndex::PrimTypePriorityFunctorHandle>::
                            New(std::make_shared<_PriorityFunctor>())))),
      _observer(HdsiPrimManagingSceneIndexObserver::New(
          _batching, HdRetainedContainerDataSource::New(
                         HdsiPrimManagingSceneIndexObserverTokens->primFactory,
                         HdRetainedTypedSampledDataSource<
                             HdsiPrimManagingSceneIndexObserver::PrimFactoryBaseHandle>::
                             New(std::make_shared<_PrimFactory>(pending, std::move(settings)))))) {}

void HdCenoteObserver::Flush() { _batching->Flush(); }

PXR_NAMESPACE_CLOSE_SCOPE
