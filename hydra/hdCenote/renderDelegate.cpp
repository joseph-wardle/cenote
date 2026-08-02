#include "renderDelegate.hpp"

#include "renderBuffer.hpp"
#include "renderPass.hpp"

#include "pxr/base/gf/vec4f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/hd/instancer.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hd/resourceRegistry.h"
#include "pxr/imaging/hd/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

// Zero Rprims by design: with no Rprim types advertised, the render
// index never hydrates geometry, and the Rprim factory below stays
// unreachable. (The instancer factory does not: the scene-index bridge
// inserts instancer prims regardless, so CreateInstancer returns an inert
// one — see there.) The camera and the renderBuffer are the whole prim vocabulary —
// plus the light types the translators read, advertised because advertising
// gates population: distant and dome also satisfy HdxTaskController's
// built-in-light check, which demands domeLight AND a camera light type before
// it injects usdview's default camera light, without which every frame is
// black; rect, disk, sphere, and cylinder are the area lights —
// plus material, the honest contract for a delegate that reads materials (it
// is also what gates population on any non-scene-index path). The light and
// material sprims themselves are inert; the translators read the same prims
// from the terminal scene index. The non-empty lists live as function-local
// statics in their getters — a namespace-scope TfTokenVector initializer
// allocates, and would throw where nothing catches.
//
// Two inherited defaults are load-bearing and deliberately not overridden
// (restating them here would only invite drift): GetMaterialBindingPurpose()
// returns `preview` — the purpose where assets that split their bindings put
// the UsdPreviewSurface look, which the registered binding-resolving filter
// (sceneIndexPlugins.cpp) collapses onto the all-purpose slot the mesh
// translator reads — and GetMaterialRenderContexts() returns the universal
// empty token, exactly the context the material translator asks each network
// for.
static const TfTokenVector kNoTypes;

namespace {

/// The advertised light types, all backed by _NullLight below.
bool _IsLightType(const TfToken& typeId) {
    return typeId == HdPrimTypeTokens->distantLight || typeId == HdPrimTypeTokens->domeLight ||
           typeId == HdPrimTypeTokens->rectLight || typeId == HdPrimTypeTokens->diskLight ||
           typeId == HdPrimTypeTokens->sphereLight || typeId == HdPrimTypeTokens->cylinderLight;
}

/// The inert backing for the advertised light types — nothing on this
/// side ever reads it.
class _NullLight final : public HdLight {
public:
    explicit _NullLight(SdfPath const& id) : HdLight(id) {}
    void Sync(HdSceneDelegate* /*sceneDelegate*/, HdRenderParam* /*renderParam*/,
              HdDirtyBits* dirtyBits) override {
        *dirtyBits = Clean;
    }
    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
};

/// The inert backing for the advertised material type, likewise.
class _NullMaterial final : public HdMaterial {
public:
    explicit _NullMaterial(SdfPath const& id) : HdMaterial(id) {}
    void Sync(HdSceneDelegate* /*sceneDelegate*/, HdRenderParam* /*renderParam*/,
              HdDirtyBits* dirtyBits) override {
        *dirtyBits = Clean;
    }
    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
};

} // namespace

HdCenoteRenderDelegate::HdCenoteRenderDelegate()
    : _resourceRegistry(std::make_shared<HdResourceRegistry>()) {}

const TfTokenVector& HdCenoteRenderDelegate::GetSupportedRprimTypes() const { return kNoTypes; }

const TfTokenVector& HdCenoteRenderDelegate::GetSupportedSprimTypes() const {
    static const TfTokenVector kSprimTypes = {
        HdPrimTypeTokens->camera,        HdPrimTypeTokens->distantLight,
        HdPrimTypeTokens->domeLight,     HdPrimTypeTokens->rectLight,
        HdPrimTypeTokens->diskLight,     HdPrimTypeTokens->sphereLight,
        HdPrimTypeTokens->cylinderLight, HdPrimTypeTokens->material};
    return kSprimTypes;
}

const TfTokenVector& HdCenoteRenderDelegate::GetSupportedBprimTypes() const {
    static const TfTokenVector kBprimTypes = {HdPrimTypeTokens->renderBuffer};
    return kBprimTypes;
}

HdResourceRegistrySharedPtr HdCenoteRenderDelegate::GetResourceRegistry() const {
    return _resourceRegistry;
}

HdRenderPassSharedPtr
HdCenoteRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                         HdRprimCollection const& collection) {
    return std::make_shared<HdCenoteRenderPass>(index, collection, &_client);
}

// The scene-index-to-render-index bridge inserts an instancer prim into
// the render index whether or not any Rprim it instances is hydrated, then
// dirties it — and the change tracker verifies the instancer it dirties
// was inserted. So an inert base HdInstancer is returned to keep that
// bookkeeping consistent (its no-op Sync is never asked for anything); the
// real placement arithmetic lives in the scene-index translator
// (instancerPrim.cpp), off this path entirely.
HdInstancer* HdCenoteRenderDelegate::CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) {
    return new HdInstancer(delegate, id);
}

void HdCenoteRenderDelegate::DestroyInstancer(HdInstancer* instancer) { delete instancer; }

HdRprim* HdCenoteRenderDelegate::CreateRprim(TfToken const& /*typeId*/,
                                             SdfPath const& /*rprimId*/) {
    return nullptr;
}

void HdCenoteRenderDelegate::DestroyRprim(HdRprim* /*rPrim*/) {}

HdSprim* HdCenoteRenderDelegate::CreateSprim(TfToken const& typeId, SdfPath const& sprimId) {
    if (typeId == HdPrimTypeTokens->camera) {
        return new HdCamera(sprimId);
    }
    if (_IsLightType(typeId)) {
        return new _NullLight(sprimId);
    }
    if (typeId == HdPrimTypeTokens->material) {
        return new _NullMaterial(sprimId);
    }
    TF_CODING_ERROR("Unknown Sprim type %s", typeId.GetText());
    return nullptr;
}

HdSprim* HdCenoteRenderDelegate::CreateFallbackSprim(TfToken const& typeId) {
    if (typeId == HdPrimTypeTokens->camera) {
        return new HdCamera(SdfPath::EmptyPath());
    }
    if (_IsLightType(typeId)) {
        return new _NullLight(SdfPath::EmptyPath());
    }
    if (typeId == HdPrimTypeTokens->material) {
        return new _NullMaterial(SdfPath::EmptyPath());
    }
    TF_CODING_ERROR("Unknown fallback Sprim type %s", typeId.GetText());
    return nullptr;
}

void HdCenoteRenderDelegate::DestroySprim(HdSprim* sprim) { delete sprim; }

HdBprim* HdCenoteRenderDelegate::CreateBprim(TfToken const& typeId, SdfPath const& bprimId) {
    if (typeId == HdPrimTypeTokens->renderBuffer) {
        return new HdCenoteRenderBuffer(bprimId, &_client);
    }
    TF_CODING_ERROR("Unknown Bprim type %s", typeId.GetText());
    return nullptr;
}

HdBprim* HdCenoteRenderDelegate::CreateFallbackBprim(TfToken const& typeId) {
    if (typeId == HdPrimTypeTokens->renderBuffer) {
        return new HdCenoteRenderBuffer(SdfPath::EmptyPath(), &_client);
    }
    TF_CODING_ERROR("Unknown fallback Bprim type %s", typeId.GetText());
    return nullptr;
}

void HdCenoteRenderDelegate::DestroyBprim(HdBprim* bprim) { delete bprim; }

void HdCenoteRenderDelegate::CommitResources(HdChangeTracker* /*tracker*/) {}

// The render index calls this once, right after it assembles the scene
// index graph — the registered filter stack (sceneIndexPlugins.cpp)
// included — and before the stage populates anything.
void HdCenoteRenderDelegate::SetTerminalSceneIndex(
    const HdSceneIndexBaseRefPtr& terminalSceneIndex) {
    if (terminalSceneIndex && !_observer) {
        _observer = std::make_unique<HdCenoteObserver>(terminalSceneIndex, &_pending);
    }
}

// Hydra's serial per-frame hook, ahead of prim sync: flush the batched
// notices through the translators, then put whatever they appended on
// the wire. The first flush is genesis — a Replace, sent even when it
// carries nothing, so the server's scene is *this* scene by declaration
// and not by coincidence of both being empty. Every later flush is an
// Apply, skipped when there is nothing to say. Both block only on the
// local Ack — a receipt, not a render.
void HdCenoteRenderDelegate::Update() {
    if (!_observer) {
        return;
    }
    _observer->Flush();
    if (!_client.alive()) {
        // Degraded: the edits have nowhere to go, and holding them
        // would only grow a list no server will ever read.
        _pending.ops.clear();
        return;
    }
    if (!_sentGenesis) {
        if (_client.replace(_pending)) {
            _sentGenesis = true;
            _pending.ops.clear();
        }
        return;
    }
    if (_pending.ops.empty()) {
        return;
    }
    if (_client.apply(_pending)) {
        _pending.ops.clear();
    }
}

// The two channels the task controller asks about, in the formats the shm
// framebuffer will carry: f32 RGBA color and f32 depth. Never multisampled —
// the server hands back resolved pixels. Everything else gets the invalid
// descriptor, which reads as "not offered".
HdAovDescriptor HdCenoteRenderDelegate::GetDefaultAovDescriptor(TfToken const& name) const {
    if (name == HdAovTokens->color) {
        return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0f)));
    }
    if (name == HdAovTokens->depth) {
        return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0f));
    }
    return HdAovDescriptor();
}

PXR_NAMESPACE_CLOSE_SCOPE
