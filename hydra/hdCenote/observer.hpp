// The delegate's ear on the terminal scene index — hdPrman's observer
// shape (its renderDelegate.cpp documents the rationale), kept to its
// thinnest. Notices batch up in HdsiPrimTypeNoticeBatchingSceneIndex
// until Flush() releases them into HdsiPrimManagingSceneIndexObserver,
// which owns the per-prim lifecycle — populate-on-attach, dirty routing,
// recursive subtree removal — so that bookkeeping is deleted, not
// ported. Its prim factory decides which prim types get a translator;
// translators append wire ops to the delegate's pending ChangeSet and
// never send.
#pragma once

#include <memory>

#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hdsi/primManagingSceneIndexObserver.h"
#include "pxr/imaging/hdsi/primTypeNoticeBatchingSceneIndex.h"
#include "pxr/pxr.h"

#include "settingsPrim.hpp"
#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

class HdCenoteObserver final {
public:
    /// Hangs the batching index and the prim-managing observer off the
    /// terminal scene index. Both `pending` and `settings` outlive this
    /// observer (they are the delegate's members); translators append to
    /// the first at every Flush(), and the settings translator publishes
    /// into the second, which the delegate reads at drain.
    HdCenoteObserver(HdSceneIndexBaseRefPtr const& terminal, cenote::wire::ChangeSet* pending,
                     std::shared_ptr<HdCenoteSettingsPrim::Active> settings);

    HdCenoteObserver(const HdCenoteObserver&) = delete;
    HdCenoteObserver& operator=(const HdCenoteObserver&) = delete;

    /// Releases every notice batched since the last call into the
    /// translators — the delegate calls this from Update(), Hydra's
    /// serial per-frame hook. The first call also replays whatever the
    /// scene index already holds.
    void Flush();

private:
    HdsiPrimTypeNoticeBatchingSceneIndexRefPtr _batching;
    HdsiPrimManagingSceneIndexObserverRefPtr _observer;
};

PXR_NAMESPACE_CLOSE_SCOPE
