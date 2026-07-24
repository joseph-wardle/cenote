// The one place two USD versions are allowed to differ (D-122). cenote's
// delegate is a single source that must compile against both stock OpenUSD
// 26.05 (the everyday build and the pixel oracle) and the USD 25.05 baked
// into Houdini's HDK (so the same .so loads inside husk). Where an imaging
// API changed between those two, the difference is absorbed here and nowhere
// else — the delegate proper stays version-blind.
//
// Validated at exactly two points: PXR_VERSION 2505 (HDK) and 2605 (stock).
// The gate reads "anything newer than the HDK's 25.05 speaks the new API";
// intermediate quarterly releases are out of scope and untested.
#pragma once

#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/types.h"
#include "pxr/pxr.h"

#include <string>

// --- HdRendererPlugin::IsSupported ------------------------------------------
// 25.05 declares exactly one, pure: `IsSupported(bool gpuEnabled)`. 26.05
// deprecates that and makes `IsSupported(HdRendererCreateArgs const&,
// std::string*)` the pure override instead. The two differ in arity, so the
// override's parameter list cannot be a typedef — it is this macro, expanded
// inside PXR_NAMESPACE where the unqualified names resolve. The parameters are
// unnamed on purpose: the delegate renders out of process and ignores every
// one of them, and unnamed parameters draw no -Wunused-parameter under the
// tree's -Werror.
#if PXR_VERSION > 2505
#define CENOTE_ISSUPPORTED_PARAMS HdRendererCreateArgs const&, std::string*
#else
#define CENOTE_ISSUPPORTED_PARAMS bool
#endif

namespace cenote {

// --- HdMeshUtil::ComputeTriangulatedFaceVaryingPrimvar ----------------------
// 26.05 returns a three-state enum; 25.05 returns bool. The old bool API
// triangulates and writes on success, has no "already all triangles" fast
// path, and returns false only when it cannot resolve the type — so `true`
// maps to Success (read `triangulated`) and `false` to Error, and Unchanged
// simply never arises there.
enum class TriangulateResult { Error, Success, Unchanged };

inline TriangulateResult ComputeTriangulatedFaceVarying(const PXR_NS::HdMeshUtil& util,
                                                        const void* source, int numElements,
                                                        PXR_NS::HdType dataType,
                                                        PXR_NS::VtValue* triangulated) {
#if PXR_VERSION > 2505
    switch (
        util.ComputeTriangulatedFaceVaryingPrimvar(source, numElements, dataType, triangulated)) {
    case PXR_NS::HdMeshComputationResult::Success:
        return TriangulateResult::Success;
    case PXR_NS::HdMeshComputationResult::Unchanged:
        return TriangulateResult::Unchanged;
    case PXR_NS::HdMeshComputationResult::Error:
        return TriangulateResult::Error;
    }
    return TriangulateResult::Error;
#else
    return util.ComputeTriangulatedFaceVaryingPrimvar(source, numElements, dataType, triangulated)
               ? TriangulateResult::Success
               : TriangulateResult::Error;
#endif
}

} // namespace cenote
