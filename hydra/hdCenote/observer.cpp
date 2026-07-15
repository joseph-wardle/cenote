#include "observer.hpp"

#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/tokens.h"

#include "lightPrim.hpp"
#include "meshPrim.hpp"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// The batching index orders each flush by this priority. cenote needs
/// no cross-type ordering: the server merges a whole ChangeSet and
/// validates the result, so forward references within one flush are
/// already legal. One bucket; the functor exists because the batching
/// index requires one.
class _PriorityFunctor final
    : public HdsiPrimTypeNoticeBatchingSceneIndex::PrimTypePriorityFunctor {
public:
    size_t GetPriorityForPrimType(const TfToken& /*primType*/) const override { return 0; }
    size_t GetNumPriorities() const override { return 1; }
};

/// Hands each added prim its translator by type — and every type it
/// does not know gets null, which the observer simply never tracks: how
/// unknown stays non-fatal forever. Mesh prims (implicits included,
/// already converted by the registered filter stack) get the mesh
/// translator; distant lights (usdview's camera light among them, per
/// D-108) get the light translator. The rest of UsdLux is step 4.
class _PrimFactory final : public HdsiPrimManagingSceneIndexObserver::PrimFactoryBase {
public:
    explicit _PrimFactory(cenote::wire::ChangeSet* pending)
        : _pending(pending), _meshes(std::make_shared<HdCenoteMeshPrim::Registry>()),
          _lights(std::make_shared<HdCenoteLightPrim::Registry>()) {}

    HdsiPrimManagingSceneIndexObserver::PrimBaseHandle
    CreatePrim(const HdSceneIndexObserver::AddedPrimEntry& entry,
               const HdsiPrimManagingSceneIndexObserver* observer) override {
        if (entry.primType == HdPrimTypeTokens->mesh) {
            return std::make_shared<HdCenoteMeshPrim>(entry.primPath, observer, _pending, _meshes);
        }
        if (entry.primType == HdPrimTypeTokens->distantLight) {
            return std::make_shared<HdCenoteLightPrim>(entry.primPath, observer, _pending, _lights);
        }
        return nullptr;
    }

private:
    /// The delegate's pending ChangeSet, shared with every translator.
    cenote::wire::ChangeSet* _pending;
    /// The translators' resync tie-breakers (see each Registry's doc).
    std::shared_ptr<HdCenoteMeshPrim::Registry> _meshes;
    std::shared_ptr<HdCenoteLightPrim::Registry> _lights;
};

} // namespace

HdCenoteObserver::HdCenoteObserver(HdSceneIndexBaseRefPtr const& terminal,
                                   cenote::wire::ChangeSet* pending)
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
                             New(std::make_shared<_PrimFactory>(pending))))) {}

void HdCenoteObserver::Flush() { _batching->Flush(); }

PXR_NAMESPACE_CLOSE_SCOPE
