// The trimmed convenience filter stack, registered for the Cenote
// renderer via HdSceneIndexPluginRegistry (each class also has a Types
// entry in plugInfo.json.in — that metadata is what makes the registry
// load us for the render index it is assembling). Five stock filters,
// borrowed from the hdPrman/hdSt convenience stacks: purpose-split
// material bindings collapse onto the all-purpose slot, implicit
// surfaces become meshes, computed primvars (skinning) become plain
// primvars, sourceAsset shaders resolve to node identifiers, and
// declared dependencies forward dirtiness at the end of the chain.
#include "pxr/imaging/hd/dependencyForwardingSceneIndex.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexPlugin.h"
#include "pxr/imaging/hd/sceneIndexPluginRegistry.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hdsi/extComputationPrimvarPruningSceneIndex.h"
#include "pxr/imaging/hdsi/implicitSurfaceSceneIndex.h"
#include "pxr/imaging/hdsi/materialBindingResolvingSceneIndex.h"
#include "pxr/imaging/hdsi/nodeIdentifierResolvingSceneIndex.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (glslfx));

/// material:binding:preview bindings collapse onto the all-purpose slot
/// — preview is where assets that split their bindings put the
/// UsdPreviewSurface look, and the mesh translator reads only the
/// all-purpose binding. Exactly Storm's block
/// (hdSt/materialBindingResolvingSceneIndexPlugin.cpp).
class HdCenoteMaterialBindingResolvingSceneIndexPlugin final : public HdSceneIndexPlugin {
protected:
    HdSceneIndexBaseRefPtr
    _AppendSceneIndex(const HdSceneIndexBaseRefPtr& inputScene,
                      const HdContainerDataSourceHandle& /*inputArgs*/) override {
        return HdsiMaterialBindingResolvingSceneIndex::New(
            inputScene, {HdTokens->preview, HdMaterialBindingsSchemaTokens->allPurpose},
            HdMaterialBindingsSchemaTokens->allPurpose);
    }
};

/// USD sphere/cube/cone/cylinder/capsule/plane prims become meshes for
/// free — cenote's one geometry is the triangle mesh.
class HdCenoteImplicitSurfaceSceneIndexPlugin final : public HdSceneIndexPlugin {
protected:
    HdSceneIndexBaseRefPtr
    _AppendSceneIndex(const HdSceneIndexBaseRefPtr& inputScene,
                      const HdContainerDataSourceHandle& /*inputArgs*/) override {
        static const HdDataSourceBaseHandle toMesh = HdRetainedTypedSampledDataSource<TfToken>::New(
            HdsiImplicitSurfaceSceneIndexTokens->toMesh);
        return HdsiImplicitSurfaceSceneIndex::New(
            inputScene, HdRetainedContainerDataSource::New(
                            HdPrimTypeTokens->sphere, toMesh, HdPrimTypeTokens->cube, toMesh,
                            HdPrimTypeTokens->cone, toMesh, HdPrimTypeTokens->cylinder, toMesh,
                            HdPrimTypeTokens->capsule, toMesh, HdPrimTypeTokens->plane, toMesh));
    }
};

/// Skinning and friends become plain primvars — the mesh translator
/// reads points, never runs a computation.
class HdCenoteExtComputationPrimvarPruningSceneIndexPlugin final : public HdSceneIndexPlugin {
protected:
    HdSceneIndexBaseRefPtr
    _AppendSceneIndex(const HdSceneIndexBaseRefPtr& inputScene,
                      const HdContainerDataSourceHandle& /*inputArgs*/) override {
        return HdSiExtComputationPrimvarPruningSceneIndex::New(inputScene);
    }
};

/// glslfx:sourceAsset shaders (usdImaging's preview shaders among them)
/// resolve to plain node identifiers — the same reading Storm gets, so
/// the material translator switches on identifiers alone.
class HdCenoteNodeIdentifierResolvingSceneIndexPlugin final : public HdSceneIndexPlugin {
protected:
    HdSceneIndexBaseRefPtr
    _AppendSceneIndex(const HdSceneIndexBaseRefPtr& inputScene,
                      const HdContainerDataSourceHandle& /*inputArgs*/) override {
        return HdSiNodeIdentifierResolvingSceneIndex::New(inputScene, _tokens->glslfx);
    }
};

/// Dependencies declared by earlier filters forward dirtiness to their
/// dependents; last in the chain so every declaration is honored.
class HdCenoteDependencyForwardingSceneIndexPlugin final : public HdSceneIndexPlugin {
protected:
    HdSceneIndexBaseRefPtr
    _AppendSceneIndex(const HdSceneIndexBaseRefPtr& inputScene,
                      const HdContainerDataSourceHandle& /*inputArgs*/) override {
        return HdDependencyForwardingSceneIndex::New(inputScene);
    }
};

TF_REGISTRY_FUNCTION(TfType) {
    HdSceneIndexPluginRegistry::Define<HdCenoteMaterialBindingResolvingSceneIndexPlugin>();
    HdSceneIndexPluginRegistry::Define<HdCenoteImplicitSurfaceSceneIndexPlugin>();
    HdSceneIndexPluginRegistry::Define<HdCenoteExtComputationPrimvarPruningSceneIndexPlugin>();
    HdSceneIndexPluginRegistry::Define<HdCenoteNodeIdentifierResolvingSceneIndexPlugin>();
    HdSceneIndexPluginRegistry::Define<HdCenoteDependencyForwardingSceneIndexPlugin>();
}

TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
    // HD_CENOTE_DISPLAY_NAME comes from CMake — the same value
    // configure_file stamps into plugInfo.json, so the registration and
    // the renderer plugin's displayName cannot drift apart.
    const std::string renderer = HD_CENOTE_DISPLAY_NAME;
    HdSceneIndexPluginRegistry& registry = HdSceneIndexPluginRegistry::GetInstance();
    // Binding resolution takes Storm's exact placement: the start of
    // phase 0, ahead of the content filters.
    registry.RegisterSceneIndexForRenderer(
        renderer, TfToken("HdCenoteMaterialBindingResolvingSceneIndexPlugin"), nullptr,
        /*insertionPhase=*/0, HdSceneIndexPluginRegistry::InsertionOrderAtStart);
    // The three content filters share phase 0 in registration order;
    // dependency forwarding sits at hdPrman's customary arbitrary-large
    // phase so it lands after anything else that ever joins the chain.
    for (const char* pluginName : {"HdCenoteImplicitSurfaceSceneIndexPlugin",
                                   "HdCenoteExtComputationPrimvarPruningSceneIndexPlugin",
                                   "HdCenoteNodeIdentifierResolvingSceneIndexPlugin"}) {
        registry.RegisterSceneIndexForRenderer(renderer, TfToken(pluginName), nullptr,
                                               /*insertionPhase=*/0,
                                               HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
    }
    registry.RegisterSceneIndexForRenderer(
        renderer, TfToken("HdCenoteDependencyForwardingSceneIndexPlugin"), nullptr,
        /*insertionPhase=*/1000, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
}

PXR_NAMESPACE_CLOSE_SCOPE
