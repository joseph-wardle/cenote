#include "geometryPrim.hpp"

#include "instancerPrim.hpp"
#include "materialPrim.hpp"
#include "usdCompat.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/basisCurvesSchema.h"
#include "pxr/imaging/hd/basisCurvesTopologySchema.h"
#include "pxr/imaging/hd/instancedBySchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopology.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/types.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/imaging/pxOsd/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (st));

namespace {

/// The server's own neutral default base color (description.rs). Sent
/// explicitly rather than left to get-or-create defaulting, so a
/// displayColor that disappears across a resync cannot leave a stale
/// color on the companion material. Achromatic, so it needs no colour
/// conversion — grey is grey in both Rec.709 and ACEScg.
constexpr std::array<float, 3> kNeutralColor{0.8f, 0.8f, 0.8f};

/// An authored linear Rec.709 colour (USD's displayColor convention),
/// expressed in ACEScg — the space the server renders in. This is the
/// C++ side of the one conversion `cenote::color::acescg_from_rec709`
/// (crates/cenote/src/color.rs) does for every other authoring front end:
/// the importer converts, so the delegate must too, or a displayColor is
/// silently treated as already-ACEScg and renders oversaturated (and, once
/// the server converts back to Rec.709 for display, a saturated primary
/// lands outside the 709 gamut — the negative lobe the server now clamps).
/// The matrix is `ACESCG_FROM_REC709` verbatim; the end-to-end golden
/// (hydra/tests/flip_golden.py) round-trips a primary through both sides and
/// pins the agreement, so a drift here fails there rather than silently.
std::array<float, 3> _AcescgFromRec709(const std::array<float, 3>& c) {
    const float r = c[0];
    const float g = c[1];
    const float b = c[2];
    return {
        0.6130974f * r + 0.33952314f * g + 0.04737945f * b,
        0.07019373f * r + 0.9163539f * g + 0.013452399f * b,
        0.020615594f * r + 0.10956978f * g + 0.86981466f * b,
    };
}

TfToken _TokenOr(const HdTokenDataSourceHandle& source, const TfToken& fallback) {
    return source ? source->GetTypedValue(0.0f) : fallback;
}

VtIntArray _IntsOr(const HdIntArrayDataSourceHandle& source) {
    return source ? source->GetTypedValue(0.0f) : VtIntArray();
}

/// Mirror of the server's validate_instance rule: every entry of the
/// f32 affine inverse must be finite. The mirror matters because a
/// ChangeSet is atomic — one rejected op takes every edit in the flush
/// down with it — so a degenerate transform has to cost one instance,
/// never the whole flush.
bool _Invertible(const cenote::wire::Matrix& matrix) {
    const auto& rows = matrix.rows;
    const GfVec3f c0(rows[0][0], rows[1][0], rows[2][0]);
    const GfVec3f c1(rows[0][1], rows[1][1], rows[2][1]);
    const GfVec3f c2(rows[0][2], rows[1][2], rows[2][2]);
    const GfVec3f translation(rows[0][3], rows[1][3], rows[2][3]);
    const float determinant = GfDot(c0, GfCross(c1, c2));
    const std::array<GfVec3f, 3> inverseRows = {GfCross(c1, c2) / determinant,
                                                GfCross(c2, c0) / determinant,
                                                GfCross(c0, c1) / determinant};
    for (const GfVec3f& row : inverseRows) {
        if (!(std::isfinite(row[0]) && std::isfinite(row[1]) && std::isfinite(row[2]) &&
              std::isfinite(GfDot(row, translation)))) {
            return false;
        }
    }
    return true;
}

/// A primvar's values and their domain: per corner of the triangulation
/// (an un-welded faceVarying read, three per triangle) or per point.
template <typename Array> struct _Primvar {
    Array values;
    bool perCorner;
};

/// A primvar the wire can carry, or the reasons there is none: a count
/// mismatch warns and drops, and anything else (absent, uniform,
/// constant, an unexpected type) quietly reads as absent. faceVarying
/// values come back per corner, re-fanned by HdMeshUtil in the same
/// order ComputeTriangleIndices walked, so index i lands on triangle
/// i/3's corner i%3; vertex and varying values pass through per point.
template <typename Array>
std::optional<_Primvar<Array>> _ReadPrimvar(const SdfPath& path, const HdPrimvarsSchema& primvars,
                                            const TfToken& name, const size_t pointCount,
                                            const size_t cornerCount, const HdMeshUtil& util) {
    const HdPrimvarSchema primvar = primvars.GetPrimvar(name);
    const HdSampledDataSourceHandle value = primvar.GetPrimvarValue();
    if (!value) {
        return std::nullopt;
    }
    const TfToken interpolation = _TokenOr(primvar.GetInterpolation(), TfToken());
    const bool faceVarying = interpolation == HdPrimvarSchemaTokens->faceVarying;
    if (!faceVarying && interpolation != HdPrimvarSchemaTokens->vertex &&
        interpolation != HdPrimvarSchemaTokens->varying) {
        return std::nullopt;
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (!sampled.IsHolding<Array>()) {
        return std::nullopt;
    }
    Array array = sampled.UncheckedGet<Array>();
    const size_t expected = faceVarying ? cornerCount : pointCount;
    if (array.size() != expected) {
        TF_WARN("<%s> has %zu %s for %zu %s; dropped", path.GetText(), array.size(), name.GetText(),
                expected, faceVarying ? "corners" : "points");
        return std::nullopt;
    }
    if (!faceVarying) {
        return _Primvar<Array>{std::move(array), false};
    }
    VtValue triangulated;
    const cenote::TriangulateResult result = cenote::ComputeTriangulatedFaceVarying(
        util, HdGetValueData(sampled), static_cast<int>(array.size()),
        HdGetValueTupleType(sampled).type, &triangulated);
    if (result == cenote::TriangulateResult::Error) {
        TF_WARN("<%s>: triangulating the faceVarying %s primvar failed; dropped", path.GetText(),
                name.GetText());
        return std::nullopt;
    }
    if (result == cenote::TriangulateResult::Success) {
        array = triangulated.UncheckedGet<Array>();
    }
    // Unchanged: the cage is already all triangles, right-handed, with
    // nothing skipped, so the authored corners sit in fan order as-is.
    return _Primvar<Array>{std::move(array), true};
}

/// points — always vertex, the one primvar geometry cannot do without,
/// and the one both payloads read.
VtVec3fArray _Points(const SdfPath& path, const HdPrimvarsSchema& primvars) {
    const HdSampledDataSourceHandle value = primvars.GetPrimvar(HdTokens->points).GetPrimvarValue();
    if (!value) {
        return {};
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (!sampled.IsHolding<VtVec3fArray>()) {
        TF_WARN("<%s> has points of type %s, not GfVec3f[]; geometry dropped", path.GetText(),
                sampled.GetTypeName().c_str());
        return {};
    }
    return sampled.UncheckedGet<VtVec3fArray>();
}

/// The wire Mesh payload: the base cage fan-triangulated by HdMeshUtil,
/// which honors orientation (leftHanded fans come out reversed), so the
/// triples land counter-clockwise-outward as Inline requires. cenote's
/// format is single-indexed — attributes live per position — so a
/// welded mesh copies straight through, and any faceVarying primvar
/// un-welds the mesh to three vertices per triangle so every per-corner
/// value has a position to sit on. Nullopt is "nothing the server would
/// accept" — an empty mesh is legal USD, but validation rejects a Mesh
/// without geometry.
std::optional<cenote::wire::Inline> _ReadMesh(const SdfPath& path, const HdSceneIndexPrim& prim) {
    const HdMeshSchema mesh = HdMeshSchema::GetFromParent(prim.dataSource);
    const HdMeshTopologySchema topologySchema = mesh.GetTopology();
    const VtIntArray counts = _IntsOr(topologySchema.GetFaceVertexCounts());
    const VtIntArray indices = _IntsOr(topologySchema.GetFaceVertexIndices());
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    const VtVec3fArray points = _Points(path, primvars);
    if (points.empty() || counts.empty() || indices.empty()) {
        return std::nullopt;
    }
    for (const int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= points.size()) {
            TF_WARN("<%s> indexes vertex %d of %zu; mesh dropped", path.GetText(), index,
                    points.size());
            return std::nullopt;
        }
    }
    const HdMeshTopology topology(
        _TokenOr(mesh.GetSubdivisionScheme(), PxOsdOpenSubdivTokens->none),
        _TokenOr(topologySchema.GetOrientation(), HdTokens->rightHanded), counts, indices,
        _IntsOr(topologySchema.GetHoleIndices()));
    const HdMeshUtil util(&topology, path);
    VtVec3iArray triangles;
    VtIntArray primitiveParams;
    util.ComputeTriangleIndices(&triangles, &primitiveParams);
    if (triangles.empty()) {
        return std::nullopt;
    }
    const auto normals = _ReadPrimvar<VtVec3fArray>(path, primvars, HdTokens->normals,
                                                    points.size(), indices.size(), util);
    const auto st = _ReadPrimvar<VtVec2fArray>(path, primvars, _tokens->st, points.size(),
                                               indices.size(), util);
    cenote::wire::Inline source;
    if ((normals && normals->perCorner) || (st && st->perCorner)) {
        // The un-weld: positions gather through the triangle indices,
        // per-corner values copy straight across, and per-point values
        // gather alongside the positions. Only meshes that carry
        // faceVarying data pay the duplication.
        source.positions.reserve(3 * triangles.size());
        source.triangles.reserve(triangles.size());
        if (normals) {
            source.normals.emplace();
            source.normals->reserve(3 * triangles.size());
        }
        if (st) {
            source.uvs.emplace();
            source.uvs->reserve(3 * triangles.size());
        }
        for (size_t t = 0; t < triangles.size(); ++t) {
            const GfVec3i& triangle = triangles[t];
            const auto base = static_cast<std::uint32_t>(3 * t);
            source.triangles.push_back({base, base + 1, base + 2});
            for (int k = 0; k < 3; ++k) {
                const GfVec3f& position = points[triangle[k]];
                source.positions.push_back({position[0], position[1], position[2]});
                if (normals) {
                    const GfVec3f& normal = normals->perCorner ? normals->values[base + k]
                                                               : normals->values[triangle[k]];
                    source.normals->push_back({normal[0], normal[1], normal[2]});
                }
                if (st) {
                    const GfVec2f& uv =
                        st->perCorner ? st->values[base + k] : st->values[triangle[k]];
                    source.uvs->push_back({uv[0], uv[1]});
                }
            }
        }
        return source;
    }
    source.positions.reserve(points.size());
    for (const GfVec3f& point : points) {
        source.positions.push_back({point[0], point[1], point[2]});
    }
    source.triangles.reserve(triangles.size());
    for (const GfVec3i& triangle : triangles) {
        source.triangles.push_back({static_cast<std::uint32_t>(triangle[0]),
                                    static_cast<std::uint32_t>(triangle[1]),
                                    static_cast<std::uint32_t>(triangle[2])});
    }
    if (normals) {
        source.normals.emplace();
        source.normals->reserve(normals->values.size());
        for (const GfVec3f& normal : normals->values) {
            source.normals->push_back({normal[0], normal[1], normal[2]});
        }
    }
    if (st) {
        source.uvs.emplace();
        source.uvs->reserve(st->values.size());
        for (const GfVec2f& uv : st->values) {
            source.uvs->push_back({uv[0], uv[1]});
        }
    }
    return source;
}

/// UsdGeomBasisCurves' vstep: how many control vertices one span of a
/// cubic curve advances by. Mirror of `CurveBasis::vstep`.
size_t _VStep(const cenote::wire::CurveBasis basis) {
    return basis == cenote::wire::CurveBasis::Bezier ? 3 : 1;
}

/// Whether a wrap actually pins — `pinned` is defined only for the
/// approximating bases, and a pinned bezier is a nonperiodic bezier.
/// Mirror of curves.rs' `pinned`.
bool _Pins(const cenote::wire::CurveWrap wrap, const cenote::wire::CurveBasis basis) {
    return wrap == cenote::wire::CurveWrap::Pinned && basis != cenote::wire::CurveBasis::Bezier;
}

/// One curve's segment count, or nullopt for a vertex count no span of
/// this basis can hold. The arithmetic half of `curves::segment_count`,
/// the server's single statement of what geometry it accepts; the
/// periodic wrap it also refuses is refused by name in _ReadCurves,
/// which is why no branch here reads for it. Duplicated at all because a
/// ChangeSet is atomic: a prim the server would reject must be withdrawn
/// rather than emitted, and only this table says which prims those are.
/// Both refusals are exercised from the fixture stage
/// (tests/stages/curves-stage.usda), one prim each.
std::optional<size_t> _SegmentCount(const size_t count, const cenote::wire::CurveType curveType,
                                    const cenote::wire::CurveBasis basis,
                                    const cenote::wire::CurveWrap wrap) {
    if (curveType == cenote::wire::CurveType::Linear || _Pins(wrap, basis)) {
        return count >= 2 ? std::optional<size_t>(count - 1) : std::nullopt;
    }
    const size_t step = _VStep(basis);
    if (count >= 4 && (count - 4) % step == 0) {
        return (count - 4) / step + 1;
    }
    return std::nullopt;
}

/// The three topology tokens, each nullopt when it spells something the
/// wire has no word for (`hermite` and `power`, retired from
/// UsdGeomBasisCurves, land here) — refused by name rather than
/// approximated by the nearest basis.
std::optional<cenote::wire::CurveType> _CurveType(const TfToken& token) {
    if (token == HdTokens->linear) {
        return cenote::wire::CurveType::Linear;
    }
    if (token == HdTokens->cubic) {
        return cenote::wire::CurveType::Cubic;
    }
    return std::nullopt;
}

std::optional<cenote::wire::CurveBasis> _CurveBasis(const TfToken& token) {
    if (token == HdTokens->bezier) {
        return cenote::wire::CurveBasis::Bezier;
    }
    if (token == HdTokens->bSpline) {
        return cenote::wire::CurveBasis::BSpline;
    }
    if (token == HdTokens->catmullRom) {
        return cenote::wire::CurveBasis::CatmullRom;
    }
    return std::nullopt;
}

std::optional<cenote::wire::CurveWrap> _CurveWrap(const TfToken& token) {
    if (token == HdTokens->nonperiodic) {
        return cenote::wire::CurveWrap::Nonperiodic;
    }
    if (token == HdTokens->pinned) {
        return cenote::wire::CurveWrap::Pinned;
    }
    if (token == HdTokens->periodic) {
        return cenote::wire::CurveWrap::Periodic;
    }
    return std::nullopt;
}

/// The width stream, or nullopt for "unauthored" — which the server
/// reads as UsdGeomCurves' own fallback of one meter everywhere, so an
/// absent `widths` is a value, not a gap. A stream whose length does not
/// match what its interpolation asks for is dropped under a warning,
/// the same rule the mesh side applies to a mis-counted primvar: the
/// curves still render, at the spec's default width, and the log says
/// why they look wrong.
std::optional<cenote::wire::Widths> _ReadWidths(const SdfPath& path,
                                                const HdPrimvarsSchema& primvars,
                                                const size_t vertices, const size_t varying,
                                                const size_t curves) {
    const HdPrimvarSchema primvar = primvars.GetPrimvar(HdTokens->widths);
    const HdSampledDataSourceHandle value = primvar.GetPrimvarValue();
    if (!value) {
        return std::nullopt;
    }
    const TfToken interpolation = _TokenOr(primvar.GetInterpolation(), TfToken());
    cenote::wire::Widths widths;
    size_t expected = 0;
    if (interpolation == HdPrimvarSchemaTokens->constant) {
        widths.interpolation = cenote::wire::WidthInterpolation::Constant;
        expected = 1;
    } else if (interpolation == HdPrimvarSchemaTokens->uniform) {
        widths.interpolation = cenote::wire::WidthInterpolation::Uniform;
        expected = curves;
    } else if (interpolation == HdPrimvarSchemaTokens->varying) {
        widths.interpolation = cenote::wire::WidthInterpolation::Varying;
        expected = varying;
    } else if (interpolation == HdPrimvarSchemaTokens->vertex) {
        widths.interpolation = cenote::wire::WidthInterpolation::Vertex;
        expected = vertices;
    } else {
        TF_WARN("<%s> interpolates widths as %s, which UsdGeomCurves does not define; dropped, so "
                "the curves render one meter wide",
                path.GetText(), interpolation.GetText());
        return std::nullopt;
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (!sampled.IsHolding<VtFloatArray>()) {
        TF_WARN("<%s> has widths of type %s, not float[]; dropped, so the curves render one meter "
                "wide",
                path.GetText(), sampled.GetTypeName().c_str());
        return std::nullopt;
    }
    const VtFloatArray& values = sampled.UncheckedGet<VtFloatArray>();
    if (values.size() != expected) {
        TF_WARN("<%s> carries %zu %s widths, but its topology asks for %zu; dropped, so the curves "
                "render one meter wide",
                path.GetText(), values.size(), interpolation.GetText(), expected);
        return std::nullopt;
    }
    widths.values.assign(values.begin(), values.end());
    return widths;
}

/// The wire Curves payload: BasisCurves cells passed through as authored.
/// Nothing is evaluated here — points, counts, and the three topology
/// tokens travel verbatim, and the server owns the curve mathematics
/// exactly once (scene/curves.rs). The whole of this function is reading
/// and refusing.
///
/// Nullopt is "nothing the server would accept": no curves, a periodic
/// wrap, a vertex count no basis can span, counts that do not partition
/// the points array, or the indexed topology the wire has no room for.
/// Each is a withdrawal rather than a silent approximation, because a
/// ChangeSet is atomic — one rejected op would take every other edit in
/// the flush down with it.
std::optional<cenote::wire::CurveCells>
_ReadCurves(const SdfPath& path, const HdSceneIndexPrim& prim, bool& warnedNormals) {
    const HdBasisCurvesTopologySchema topology =
        HdBasisCurvesSchema::GetFromParent(prim.dataSource).GetTopology();
    // UsdGeomBasisCurves' own fallbacks, for a prim that leaves one
    // unauthored: a cubic bezier that does not close.
    const TfToken typeToken = _TokenOr(topology.GetType(), HdTokens->cubic);
    const TfToken basisToken = _TokenOr(topology.GetBasis(), HdTokens->bezier);
    const TfToken wrapToken = _TokenOr(topology.GetWrap(), HdTokens->nonperiodic);
    const std::optional<cenote::wire::CurveType> curveType = _CurveType(typeToken);
    const std::optional<cenote::wire::CurveWrap> wrap = _CurveWrap(wrapToken);
    // A linear curve has no basis, and Houdini writes the token empty
    // rather than inventing one. Mapping the basis before the type
    // would therefore refuse every polyline groom a DCC exports. Bezier
    // stands in; nothing downstream of Linear consults it.
    const std::optional<cenote::wire::CurveBasis> basis =
        curveType == cenote::wire::CurveType::Linear
            ? std::optional<cenote::wire::CurveBasis>(cenote::wire::CurveBasis::Bezier)
            : _CurveBasis(basisToken);
    if (!curveType || !basis || !wrap) {
        TF_WARN("<%s> is a '%s' '%s' '%s' curve, which cenote does not read; curves dropped",
                path.GetText(), typeToken.GetText(), basisToken.GetText(), wrapToken.GetText());
        return std::nullopt;
    }
    if (*wrap == cenote::wire::CurveWrap::Periodic) {
        // The delegate's one statement of the rule, in the server's own
        // words — _SegmentCount below reads only vertex counts.
        TF_WARN("<%s> is periodic, and a closed loop has no root to sweep a strand from; curves "
                "dropped",
                path.GetText());
        return std::nullopt;
    }
    if (!_IntsOr(topology.GetCurveIndices()).empty()) {
        // Indexed curve vertices are a Hydra topology the wire has no
        // spelling for (UsdGeomBasisCurves authors none). De-indexing
        // would have to guess which primvar domains the indices apply
        // to, so the honest answer is to say no.
        TF_WARN("<%s> indexes its curve vertices, which cenote does not read; curves dropped",
                path.GetText());
        return std::nullopt;
    }
    const VtIntArray counts = _IntsOr(topology.GetCurveVertexCounts());
    if (counts.empty()) {
        return std::nullopt;
    }
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    const VtVec3fArray points = _Points(path, primvars);
    if (points.empty()) {
        return std::nullopt;
    }
    // The two totals the width stream is measured against: control
    // vertices, and segment ends. Walking the counts is also what proves
    // the topology, one curve at a time, in the server's own words.
    size_t vertices = 0;
    size_t varying = 0;
    for (size_t index = 0; index < counts.size(); ++index) {
        const int count = counts[index];
        const std::optional<size_t> segments =
            count < 0 ? std::nullopt
                      : _SegmentCount(static_cast<size_t>(count), *curveType, *basis, *wrap);
        if (!segments) {
            TF_WARN("<%s> gives curve %zu %d vertices, which its topology cannot span; curves "
                    "dropped",
                    path.GetText(), index, count);
            return std::nullopt;
        }
        vertices += static_cast<size_t>(count);
        varying += *segments + 1;
    }
    if (vertices != points.size()) {
        TF_WARN("<%s> names %zu vertices across %zu curves, but carries %zu points; curves dropped",
                path.GetText(), vertices, counts.size(), points.size());
        return std::nullopt;
    }
    if (!warnedNormals && primvars.GetPrimvar(HdTokens->normals).GetPrimvarValue()) {
        warnedNormals = true;
        TF_WARN("<%s> authors curve normals; cenote sweeps round tubes and ignores them",
                path.GetText());
    }
    cenote::wire::CurveCells cells;
    cells.curve_type = *curveType;
    cells.basis = *basis;
    cells.wrap = *wrap;
    cells.points.reserve(points.size());
    for (const GfVec3f& point : points) {
        cells.points.push_back({point[0], point[1], point[2]});
    }
    cells.curve_vertex_counts.reserve(counts.size());
    for (const int count : counts) {
        cells.curve_vertex_counts.push_back(static_cast<std::uint32_t>(count));
    }
    cells.widths = _ReadWidths(path, primvars, vertices, varying, counts.size());
    return cells;
}

/// What the companion material wears: a constant displayColor is used
/// directly, a vertex one is approximated by its first element — the
/// wire carries no per-vertex color, so the approximation is the
/// contract, not a stopgap — and anything else is the neutral default.
std::array<float, 3> _ReadDisplayColor(const HdSceneIndexPrim& prim) {
    const HdPrimvarSchema primvar =
        HdPrimvarsSchema::GetFromParent(prim.dataSource).GetPrimvar(HdTokens->displayColor);
    const HdSampledDataSourceHandle value = primvar.GetPrimvarValue();
    if (!value) {
        return kNeutralColor;
    }
    const TfToken interpolation = _TokenOr(primvar.GetInterpolation(), TfToken());
    if (interpolation != HdPrimvarSchemaTokens->constant &&
        interpolation != HdPrimvarSchemaTokens->vertex &&
        interpolation != HdPrimvarSchemaTokens->varying) {
        return kNeutralColor;
    }
    const VtValue sampled = value->GetValue(0.0f);
    if (sampled.IsHolding<GfVec3f>()) {
        const GfVec3f color = sampled.UncheckedGet<GfVec3f>();
        return _AcescgFromRec709({color[0], color[1], color[2]});
    }
    if (sampled.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& colors = sampled.UncheckedGet<VtVec3fArray>();
        if (!colors.empty()) {
            return _AcescgFromRec709({colors[0][0], colors[0][1], colors[0][2]});
        }
    }
    return kNeutralColor;
}

/// The all-purpose binding target; empty when unbound. The flattening
/// stack has already resolved binding inheritance and the registered
/// resolving filter (sceneIndexPlugins.cpp) collapsed
/// material:binding:preview onto the all-purpose slot, so this one
/// read is the final word.
SdfPath _ReadBinding(const HdSceneIndexPrim& prim) {
    const HdPathDataSourceHandle path =
        HdMaterialBindingsSchema::GetFromParent(prim.dataSource).GetMaterialBinding().GetPath();
    return path ? path->GetTypedValue(0.0f) : SdfPath();
}

/// Resolved (flattened) visibility; absent means visible.
bool _ReadVisibility(const HdSceneIndexPrim& prim) {
    const HdBoolDataSourceHandle visibility =
        HdVisibilitySchema::GetFromParent(prim.dataSource).GetVisibility();
    return visibility ? visibility->GetTypedValue(0.0f) : true;
}

/// The prim's flattened transform; identity when unauthored. For an
/// un-instanced prim this is the world matrix; for one inside a point-
/// instancer prototype it is prototype-root relative (the prototype root
/// carries resetXformStack, so the instancer's placement is never folded
/// in here — that is ComputePlacements's job).
GfMatrix4d _WorldMatrix(const HdSceneIndexPrim& prim) {
    if (const HdMatrixDataSourceHandle matrix =
            HdXformSchema::GetFromParent(prim.dataSource).GetMatrix()) {
        return matrix->GetTypedValue(0.0f);
    }
    return GfMatrix4d(1.0);
}

/// A world matrix as Transform::Matrix — never decomposed. Gf composes
/// against row vectors, the wire against column vectors, so the rows are
/// the transpose's. Nullopt means the server would reject the placement
/// (see _Invertible).
std::optional<cenote::wire::Matrix> _WireTransform(const GfMatrix4d& world) {
    cenote::wire::Matrix transform;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            transform.rows[row][column] = static_cast<float>(world[column][row]);
        }
    }
    if (!_Invertible(transform)) {
        return std::nullopt;
    }
    return transform;
}

/// Everything that reshapes the wire Mesh payload: topology plus the
/// three primvars Inline carries.
const HdDataSourceLocatorSet& _MeshLocators() {
    static const HdDataSourceLocatorSet locators{
        HdMeshSchema::GetDefaultLocator(), HdPrimvarsSchema::GetPointsLocator(),
        HdPrimvarsSchema::GetDefaultLocator().Append(HdTokens->normals),
        HdPrimvarsSchema::GetDefaultLocator().Append(_tokens->st)};
    return locators;
}

/// Everything that reshapes the wire Curves payload: the BasisCurves
/// topology, the points, and the widths. Authored normals are absent on
/// purpose — they are dropped on the way in, so an edit to them changes
/// nothing the server would see.
const HdDataSourceLocatorSet& _CurveLocators() {
    static const HdDataSourceLocatorSet locators{
        HdBasisCurvesSchema::GetDefaultLocator(), HdPrimvarsSchema::GetPointsLocator(),
        HdPrimvarsSchema::GetDefaultLocator().Append(HdTokens->widths)};
    return locators;
}

const HdDataSourceLocator& _DisplayColorLocator() {
    static const HdDataSourceLocator locator =
        HdPrimvarsSchema::GetDefaultLocator().Append(HdTokens->displayColor);
    return locator;
}

/// Everything that reshapes the placements array: the flattened transform,
/// visibility, and instancedBy (the set of instancers that copy this prim,
/// which turns a lone placement into an array and back).
const HdDataSourceLocatorSet& _PlacementLocators() {
    static const HdDataSourceLocatorSet locators{HdXformSchema::GetDefaultLocator(),
                                                 HdVisibilitySchema::GetDefaultLocator(),
                                                 HdInstancedBySchema::GetDefaultLocator()};
    return locators;
}

} // namespace

HdCenoteGeometryPrim::HdCenoteGeometryPrim(const SdfPath& path, const TfToken& primType,
                                           const HdsiPrimManagingSceneIndexObserver* observer,
                                           cenote::wire::ChangeSet* pending,
                                           std::shared_ptr<Registry> live,
                                           std::shared_ptr<const MaterialRegistry> materials,
                                           std::shared_ptr<const InstancerRegistry> instancers)
    : _path(path), _curves(primType == HdPrimTypeTokens->basisCurves), _name(path.GetString()),
      _material(path.GetString() + "/displayColor"), _pending(pending), _live(std::move(live)),
      _materials(std::move(materials)), _instancers(std::move(instancers)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: the previous translator still holds server objects.
        // Inherit its ledger — the reconcile below then updates in
        // place — and take the registry slot so its destructor, which
        // runs after this constructor, goes quietly. The inherited
        // payload may be the other kind; _Reconcile withdraws it.
        const HdCenoteGeometryPrim& previous = *it->second;
        it->second = this;
        _payload = previous._payload;
        _instanceLive = previous._instanceLive;
        _binding = previous._binding;
        _wearsBinding = previous._wearsBinding;
        _warnedElement = previous._warnedElement;
        _warnedNormals = previous._warnedNormals;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _Withdraw();
        return;
    }
    _Reconcile(prim, _Dirt{.geometry = true, .color = true, .placement = true, .binding = true});
}

HdCenoteGeometryPrim::~HdCenoteGeometryPrim() {
    const auto it = _live->find(_path);
    if (it == _live->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _live->erase(it);
    _Withdraw();
}

void HdCenoteGeometryPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                                  const HdsiPrimManagingSceneIndexObserver* observer) {
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        return;
    }
    _Reconcile(prim, _Dirt{
                         .geometry = entry.dirtyLocators.Intersects(_curves ? _CurveLocators()
                                                                            : _MeshLocators()),
                         .color = entry.dirtyLocators.Intersects(_DisplayColorLocator()),
                         .placement = entry.dirtyLocators.Intersects(_PlacementLocators()),
                         .binding = entry.dirtyLocators.Intersects(
                             HdMaterialBindingsSchema::GetDefaultLocator()),
                     });
}

void HdCenoteGeometryPrim::_Reconcile(const HdSceneIndexPrim& prim, const _Dirt dirt) {
    bool born = false;
    if (dirt.geometry) {
        if (_payload && *_payload != _Kind()) {
            // The prim changed type under this path, and the payload the
            // resync handed over shares the name but not the kind. It
            // goes now, ahead of the adds below and inside the same
            // atomic ChangeSet — the only order in which a Remove names
            // an object that exists and an add does not collide.
            _Withdraw();
        }
        // The one place the two kinds part company: what is read, and
        // which patch carries it.
        bool read = false;
        if (_curves) {
            if (std::optional<cenote::wire::CurveCells> cells =
                    _ReadCurves(_path, prim, _warnedNormals)) {
                _pending->ops.push_back(cenote::wire::CurvesPatch{
                    .name = _name, .source = cenote::wire::CurvesSource{std::move(*cells)}});
                read = true;
            }
        } else if (std::optional<cenote::wire::Inline> source = _ReadMesh(_path, prim)) {
            _pending->ops.push_back(cenote::wire::MeshPatch{
                .name = _name, .source = cenote::wire::MeshSource{std::move(*source)}});
            read = true;
        }
        if (!read) {
            _Withdraw();
            return;
        }
        if (!_payload) {
            _pending->ops.push_back(cenote::wire::MaterialPatch{
                .name = _material,
                .base_color =
                    cenote::wire::Constant<std::array<float, 3>>{_ReadDisplayColor(prim)}});
            _payload = _Kind();
            born = true;
        }
    }
    if (!_payload) {
        // Nothing on the wire yet; a future geometry dirty re-enters.
        return;
    }
    if (dirt.color && !born) {
        _pending->ops.push_back(cenote::wire::MaterialPatch{
            .name = _material,
            .base_color = cenote::wire::Constant<std::array<float, 3>>{_ReadDisplayColor(prim)}});
    }
    if (born || dirt.binding) {
        // Materials flush ahead of geometry (the batching priorities in
        // observer.cpp), so a target this sync cannot see is truly not
        // there — absent from the stage or arriving in a later wave.
        const SdfPath binding = _ReadBinding(prim);
        if (binding != _binding) {
            _binding = binding;
            if (!_binding.IsEmpty() && !_Bindable()) {
                TF_WARN("<%s> binds <%s>, which is not a material on the wire; wearing "
                        "displayColor until it arrives",
                        _path.GetText(), _binding.GetText());
            }
        }
    }
    if (born || dirt.placement) {
        // Refresh the cached placement inputs, then place from them. A
        // later instancer poke replaces the placements from these same
        // inputs without touching the scene index.
        _visible = _ReadVisibility(prim);
        _protoXform = _WorldMatrix(prim);
        _instancedBy = _ReadInstancedBy(prim);
        _Place();
    }
    // Every reconcile ends by squaring the instance's wear with the
    // registry — a binding edit repoints here, and everything else
    // no-ops.
    ResolveBinding();
}

void HdCenoteGeometryPrim::RecomposeInstancing() {
    if (_instancedBy.empty() || !_payload) {
        // A prim no instancer copies ignores instancer pokes, and one
        // whose geometry has not reached the wire has no instance to
        // place. The cached inputs are still current — only the instancer
        // moved — so _Place re-reads nothing.
        return;
    }
    _Place();
}

void HdCenoteGeometryPrim::_Place() {
    std::optional<std::vector<cenote::wire::Transform>> placements = _ComposePlacements();
    if (!placements) {
        _WithdrawInstance();
        return;
    }
    if (!_instanceLive) {
        _wearsBinding = _Bindable();
        cenote::wire::InstancePatch patch{.name = _name,
                                          .material =
                                              _wearsBinding ? _binding.GetString() : _material,
                                          .transforms = std::move(placements)};
        // Two spellings of one field on the target; naming both is
        // refused server-side, so exactly one is filled.
        (_curves ? patch.curves : patch.mesh) = _name;
        _pending->ops.push_back(std::move(patch));
        _instanceLive = true;
    } else {
        // The whole array replaces; the geometry and the binding
        // stay put.
        _pending->ops.push_back(
            cenote::wire::InstancePatch{.name = _name, .transforms = std::move(placements)});
    }
}

std::optional<std::vector<cenote::wire::Transform>> HdCenoteGeometryPrim::_ComposePlacements() {
    if (!_visible) {
        return std::nullopt;
    }
    if (_instancedBy.empty()) {
        // Un-instanced: the prim stands once at its own transform.
        const std::optional<cenote::wire::Matrix> matrix = _WireTransform(_protoXform);
        if (!matrix) {
            TF_WARN("<%s> has a non-invertible transform; instance %s", _path.GetText(),
                    _instanceLive ? "removed" : "not placed");
            return std::nullopt;
        }
        return std::vector<cenote::wire::Transform>{*matrix};
    }
    // Instanced: concatenate the placements every instancer contributes,
    // each composed as (this prim's own transform) · (the instancer chain).
    std::vector<cenote::wire::Transform> placements;
    for (const _Instancer& instancer : _instancedBy) {
        const auto it = _instancers->find(instancer.path);
        if (it == _instancers->end()) {
            // The instancer is not on the registry yet; park until its
            // birth pokes this prim. An empty placement would say the
            // opposite — resident, placed nowhere — so honesty is nullopt.
            return std::nullopt;
        }
        for (const GfMatrix4d& chain : it->second->ComputePlacements(instancer.prototypeRoot)) {
            if (const std::optional<cenote::wire::Matrix> matrix =
                    _WireTransform(_protoXform * chain)) {
                placements.push_back(*matrix);
            } else if (!_warnedElement) {
                _warnedElement = true;
                TF_WARN("<%s> composes a non-invertible instance placement; the copies it would "
                        "make are dropped",
                        _path.GetText());
            }
        }
    }
    return placements;
}

std::vector<HdCenoteGeometryPrim::_Instancer>
HdCenoteGeometryPrim::_ReadInstancedBy(const HdSceneIndexPrim& prim) const {
    const HdInstancedBySchema schema = HdInstancedBySchema::GetFromParent(prim.dataSource);
    const HdPathArrayDataSourceHandle pathsSource = schema.GetPaths();
    if (!pathsSource) {
        return {};
    }
    const VtArray<SdfPath> paths = pathsSource->GetTypedValue(0.0f);
    VtArray<SdfPath> roots;
    if (const HdPathArrayDataSourceHandle rootsSource = schema.GetPrototypeRoots()) {
        roots = rootsSource->GetTypedValue(0.0f);
    }
    std::vector<_Instancer> instancers;
    instancers.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        // prototypeRoots runs parallel to paths; a missing entry leaves the
        // root empty, and ComputePlacements then matches no prototype —
        // placing nothing, the honest reading of malformed instancedBy.
        instancers.push_back({paths[i], i < roots.size() ? roots[i] : SdfPath()});
    }
    return instancers;
}

void HdCenoteGeometryPrim::ResolveBinding() {
    if (!_instanceLive) {
        return;
    }
    const bool bindable = _Bindable();
    if (bindable == _wearsBinding) {
        return;
    }
    _pending->ops.push_back(cenote::wire::InstancePatch{
        .name = _name, .material = bindable ? _binding.GetString() : _material});
    _wearsBinding = bindable;
}

bool HdCenoteGeometryPrim::_Bindable() const {
    if (_binding.IsEmpty()) {
        return false;
    }
    const auto it = _materials->find(_binding);
    return it != _materials->end() && it->second->Published();
}

void HdCenoteGeometryPrim::_WithdrawInstance() {
    if (_instanceLive) {
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Instance, .name = _name});
        _instanceLive = false;
    }
}

void HdCenoteGeometryPrim::_Withdraw() {
    _WithdrawInstance();
    if (_payload) {
        // The kind the ledger recorded, not the kind this translator
        // publishes: after a type change under one path those differ.
        _pending->ops.push_back(cenote::wire::Remove{.kind = *_payload, .name = _name});
        _pending->ops.push_back(
            cenote::wire::Remove{.kind = cenote::wire::Kind::Material, .name = _material});
        _payload.reset();
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
