#include "materialPrim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/tf/stringUtils.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/materialConnectionSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeParameterSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/schemaTypeDefs.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens,
                         (UsdPreviewSurface)(UsdUVTexture)(UsdTransform2d)(diffuseColor)(emissiveColor)(useSpecularWorkflow)(specularColor)(metallic)(roughness)(clearcoat)(clearcoatRoughness)(opacity)(opacityThreshold)(ior)(normal)(displacement)(occlusion)(file)(st)(wrapS)(wrapT)(fallback)(scale)(bias)(varname)(in)(r)(g)(b)(a)(sRGB)(raw)(repeat)(useMetadata));

namespace {

/// The identifier prefix all eight UsdPrimvarReader flavors share.
const char* const kReaderPrefix = "UsdPrimvarReader_";

TfToken _TokenOr(const HdTokenDataSourceHandle& source, const TfToken& fallback) {
    return source ? source->GetTypedValue(0.0f) : fallback;
}

// Lenient value extraction, the light translator's habit: the stage
// path serves the schema type, and anything else reads as absent.

std::optional<float> _AsFloat(const VtValue& value) {
    if (value.IsHolding<float>()) {
        return value.UncheckedGet<float>();
    }
    if (value.IsHolding<double>()) {
        return static_cast<float>(value.UncheckedGet<double>());
    }
    return std::nullopt;
}

std::optional<int> _AsInt(const VtValue& value) {
    if (value.IsHolding<int>()) {
        return value.UncheckedGet<int>();
    }
    if (value.IsHolding<bool>()) {
        return value.UncheckedGet<bool>() ? 1 : 0;
    }
    return std::nullopt;
}

std::optional<std::array<float, 3>> _AsColor(const VtValue& value) {
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f color = value.UncheckedGet<GfVec3f>();
        return std::array<float, 3>{color[0], color[1], color[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3f color(value.UncheckedGet<GfVec3d>());
        return std::array<float, 3>{color[0], color[1], color[2]};
    }
    if (value.IsHolding<GfVec4f>()) {
        const GfVec4f color = value.UncheckedGet<GfVec4f>();
        return std::array<float, 3>{color[0], color[1], color[2]};
    }
    return std::nullopt;
}

std::optional<GfVec4f> _AsVec4(const VtValue& value) {
    if (value.IsHolding<GfVec4f>()) {
        return value.UncheckedGet<GfVec4f>();
    }
    if (value.IsHolding<GfVec4d>()) {
        return GfVec4f(value.UncheckedGet<GfVec4d>());
    }
    return std::nullopt;
}

std::optional<TfToken> _AsToken(const VtValue& value) {
    if (value.IsHolding<TfToken>()) {
        return value.UncheckedGet<TfToken>();
    }
    if (value.IsHolding<std::string>()) {
        return TfToken(value.UncheckedGet<std::string>());
    }
    return std::nullopt;
}

/// The authored value of `name` on `node`, empty when unauthored — the
/// scene index's parameter table carries authored inputs only, with
/// connection-overridden values already excluded.
VtValue _ParameterOr(const HdMaterialNodeSchema& node, const TfToken& name) {
    const HdSampledDataSourceHandle value = node.GetParameters().Get(name).GetValue();
    return value ? value->GetValue(0.0f) : VtValue();
}

/// One input of one node, resolved against the network: the upstream
/// node when connected (connections win over values in USD), otherwise
/// the authored constant, otherwise nothing.
struct _Input {
    VtValue value;                 //< authored constant, when not connected
    HdMaterialNodeSchema upstream; //< connected node; null if the table lacks it
    TfToken upstreamId;            //< its identifier, empty when unresolvable
    TfToken output;                //< the upstream output tapped
    bool connected;
};

_Input _ResolveInput(const HdMaterialNodeSchema& node, const HdMaterialNodeContainerSchema& nodes,
                     const TfToken& name) {
    _Input input{VtValue(), HdMaterialNodeSchema(nullptr), TfToken(), TfToken(), false};
    const HdMaterialConnectionVectorSchema connections = node.GetInputConnections().Get(name);
    if (connections.GetNumElements() > 0) {
        const HdMaterialConnectionSchema connection = connections.GetElement(0);
        input.upstream = nodes.Get(_TokenOr(connection.GetUpstreamNodePath(), TfToken()));
        input.upstreamId = _TokenOr(input.upstream.GetNodeIdentifier(), TfToken());
        input.output = _TokenOr(connection.GetUpstreamNodeOutputName(), TfToken());
        input.connected = true;
        return input;
    }
    input.value = _ParameterOr(node, name);
    return input;
}

bool _IsReader(const _Input& input) {
    return TfStringStartsWith(input.upstreamId.GetString(), kReaderPrefix);
}

/// The single-channel outputs a scalar slot can tap. Anything else —
/// rgb on a scalar input is malformed USD — reads as the wire's
/// absent-means-red default.
std::optional<cenote::wire::Channel> _ChannelFor(const TfToken& output) {
    if (output == _tokens->r) {
        return cenote::wire::Channel::R;
    }
    if (output == _tokens->g) {
        return cenote::wire::Channel::G;
    }
    if (output == _tokens->b) {
        return cenote::wire::Channel::B;
    }
    if (output == _tokens->a) {
        return cenote::wire::Channel::A;
    }
    return std::nullopt;
}

float _Component(const GfVec4f& value, const std::optional<cenote::wire::Channel> channel) {
    switch (channel.value_or(cenote::wire::Channel::R)) {
    case cenote::wire::Channel::G:
        return value[1];
    case cenote::wire::Channel::B:
        return value[2];
    case cenote::wire::Channel::A:
        return value[3];
    default:
        return value[0];
    }
}

/// A UsdUVTexture node's own `fallback` input — the value its author
/// chose for exactly the can't-sample case (spec default (0,0,0,1)).
GfVec4f _TextureFallback(const HdMaterialNodeSchema& node) {
    return _AsVec4(_ParameterOr(node, _tokens->fallback)).value_or(GfVec4f(0.0f, 0.0f, 0.0f, 1.0f));
}

/// A UsdUVTexture read down to what the wire can express (Q7). `ref` is
/// the reference when the file is usable and the texture samples the
/// mesh's one st channel; otherwise it is empty, a warning has named
/// why, and the node's fallback stands in. Warn policy: silent where
/// the authored value is what the core does anyway (repeat wraps, the
/// normal remap, auto color space), one warning naming the material and
/// input where fidelity is lost — and never a path the server's
/// validate_path would reject, since one rejected op takes the whole
/// atomic flush down with it (D-117).
struct _Texture {
    std::optional<cenote::wire::TextureRef> ref;
    GfVec4f fallback;
};

_Texture _ReadTexture(const SdfPath& path, const TfToken& inputName,
                      const HdMaterialNodeSchema& node,
                      const HdMaterialNodeContainerSchema& nodes) {
    _Texture texture{std::nullopt, _TextureFallback(node)};
    const HdMaterialNodeParameterSchema file = node.GetParameters().Get(_tokens->file);
    const HdSampledDataSourceHandle fileValue = file.GetValue();
    const VtValue asset = fileValue ? fileValue->GetValue(0.0f) : VtValue();
    if (!asset.IsHolding<SdfAssetPath>()) {
        TF_WARN("<%s>'s %s has no file; using the fallback constant", path.GetText(),
                inputName.GetText());
        return texture;
    }
    const SdfAssetPath& assetPath = asset.UncheckedGet<SdfAssetPath>();
    if (assetPath.GetAssetPath().find("<UDIM>") != std::string::npos) {
        TF_WARN("<%s>'s %s is a UDIM set, which cenote cannot load; using the fallback constant",
                path.GetText(), inputName.GetText());
        return texture;
    }
    // Ar has already run; the server additionally demands the resolved
    // path be absolute and an existing file, checked here where failing
    // costs one texture instead of the flush.
    const std::string& resolved = assetPath.GetResolvedPath();
    std::error_code unused;
    if (resolved.empty() || !std::filesystem::path(resolved).is_absolute() ||
        !std::filesystem::is_regular_file(resolved, unused)) {
        TF_WARN("<%s>'s %s references \"%s\", which is not a readable file; using the fallback "
                "constant",
                path.GetText(), inputName.GetText(), assetPath.GetAssetPath().c_str());
        return texture;
    }
    // The st feed (Q8): unconnected quietly means the st primvar; a
    // reader must name st, since the wire mesh ships exactly that one
    // channel and any other varname would place the texture silently
    // wrong; a UsdTransform2d is skipped under a warning — placement
    // drift is kinder than killing the texture over a UV offset.
    _Input st = _ResolveInput(node, nodes, _tokens->st);
    for (int hops = 0; st.connected && st.upstreamId == _tokens->UsdTransform2d && hops < 8;
         ++hops) {
        TF_WARN("<%s>'s %s has a UsdTransform2d, which the sampler cannot honor; the UV "
                "transform is ignored",
                path.GetText(), inputName.GetText());
        st = _ResolveInput(st.upstream, nodes, _tokens->in);
    }
    if (st.connected) {
        if (!_IsReader(st)) {
            TF_WARN("<%s>'s %s takes its UVs from something other than an st primvar reader; "
                    "using the fallback constant",
                    path.GetText(), inputName.GetText());
            return texture;
        }
        const TfToken varname =
            _AsToken(_ParameterOr(st.upstream, _tokens->varname)).value_or(TfToken());
        if (varname != _tokens->st) {
            TF_WARN("<%s>'s %s samples the %s primvar, but the wire mesh carries only st; "
                    "using the fallback constant",
                    path.GetText(), inputName.GetText(), varname.GetText());
            return texture;
        }
    }
    const auto honored = [](const TfToken& wrap) {
        return wrap == _tokens->repeat || wrap == _tokens->useMetadata;
    };
    const TfToken wrapS =
        _AsToken(_ParameterOr(node, _tokens->wrapS)).value_or(_tokens->useMetadata);
    const TfToken wrapT =
        _AsToken(_ParameterOr(node, _tokens->wrapT)).value_or(_tokens->useMetadata);
    if (!honored(wrapS) || !honored(wrapT)) {
        TF_WARN("<%s>'s %s wraps as %s/%s, but the sampler is hardwired repeat", path.GetText(),
                inputName.GetText(), wrapS.GetText(), wrapT.GetText());
    }
    // The identity scale/bias passes silently, and on the normal input
    // so does the canonical [0,1]→[-1,1] remap in rgb: the core's
    // normal-map decode already applies it, so the authored pair merely
    // spells what the sampler does natively (the fourth component never
    // gets sampled from a normal map, so its leg is moot).
    const GfVec4f one(1.0f);
    const GfVec4f zero(0.0f);
    const GfVec4f scale = _AsVec4(_ParameterOr(node, _tokens->scale)).value_or(one);
    const GfVec4f bias = _AsVec4(_ParameterOr(node, _tokens->bias)).value_or(zero);
    const bool identity = scale == one && bias == zero;
    const bool remap = inputName == _tokens->normal && scale[0] == 2.0f && scale[1] == 2.0f &&
                       scale[2] == 2.0f && bias[0] == -1.0f && bias[1] == -1.0f && bias[2] == -1.0f;
    if (!identity && !remap) {
        TF_WARN("<%s>'s %s authors a scale/bias the sampler cannot apply; ignored", path.GetText(),
                inputName.GetText());
    }
    // usdImaging consolidates sourceColorSpace onto the file parameter.
    // sRGB and raw pin the wire's two encodings; auto — and any other
    // spelling — stays absent, because the core's per-slot auto (8-bit
    // color reads sRGB, float linear, scalars and normals raw) is
    // exactly auto's spec semantics.
    const TfToken colorSpace = _TokenOr(file.GetColorSpace(), TfToken());
    std::optional<cenote::wire::ColorSpace> space;
    if (colorSpace == _tokens->sRGB) {
        space = cenote::wire::ColorSpace::Srgb;
    } else if (colorSpace == _tokens->raw) {
        space = cenote::wire::ColorSpace::Linear;
    }
    texture.ref = cenote::wire::TextureRef{resolved, space, std::nullopt};
    return texture;
}

/// A color input resolved to what the wire carries: a texture when one
/// is usable, a constant otherwise, nothing when unauthored — the
/// caller keeps its OpenPBR default. An unusable texture degrades to
/// the texture's fallback, a primvar reader to the reader's fallback
/// (Q8: per-vertex data has no wire home), and an unmappable upstream
/// node reads as unauthored, each under one warning.
std::optional<cenote::wire::Texturable<std::array<float, 3>>>
_ReadColorSlot(const SdfPath& path, const HdMaterialNodeSchema& surface,
               const HdMaterialNodeContainerSchema& nodes, const TfToken& name) {
    const _Input input = _ResolveInput(surface, nodes, name);
    if (input.connected) {
        if (input.upstreamId == _tokens->UsdUVTexture) {
            _Texture texture = _ReadTexture(path, name, input.upstream, nodes);
            if (texture.ref) {
                return std::move(*texture.ref);
            }
            return cenote::wire::Constant<std::array<float, 3>>{
                {texture.fallback[0], texture.fallback[1], texture.fallback[2]}};
        }
        if (_IsReader(input)) {
            TF_WARN("<%s>'s %s reads a primvar, which the wire cannot carry; using the "
                    "reader's fallback",
                    path.GetText(), name.GetText());
            return cenote::wire::Constant<std::array<float, 3>>{
                _AsColor(_ParameterOr(input.upstream, _tokens->fallback))
                    .value_or(std::array<float, 3>{0.0f, 0.0f, 0.0f})};
        }
        TF_WARN("<%s>'s %s is driven by %s, which has no mapping; treated as unauthored",
                path.GetText(), name.GetText(),
                input.upstreamId.IsEmpty() ? "an unresolvable node" : input.upstreamId.GetText());
        return std::nullopt;
    }
    if (const auto color = _AsColor(input.value)) {
        return cenote::wire::Constant<std::array<float, 3>>{*color};
    }
    return std::nullopt;
}

/// The scalar twin of _ReadColorSlot; the connection's output name
/// picks the texture's source channel (Q6's TextureRef growth).
std::optional<cenote::wire::Texturable<float>>
_ReadScalarSlot(const SdfPath& path, const HdMaterialNodeSchema& surface,
                const HdMaterialNodeContainerSchema& nodes, const TfToken& name) {
    const _Input input = _ResolveInput(surface, nodes, name);
    if (input.connected) {
        if (input.upstreamId == _tokens->UsdUVTexture) {
            const std::optional<cenote::wire::Channel> channel = _ChannelFor(input.output);
            _Texture texture = _ReadTexture(path, name, input.upstream, nodes);
            if (texture.ref) {
                texture.ref->channel = channel;
                return std::move(*texture.ref);
            }
            return cenote::wire::Constant<float>{_Component(texture.fallback, channel)};
        }
        if (_IsReader(input)) {
            TF_WARN("<%s>'s %s reads a primvar, which the wire cannot carry; using the "
                    "reader's fallback",
                    path.GetText(), name.GetText());
            return cenote::wire::Constant<float>{
                _AsFloat(_ParameterOr(input.upstream, _tokens->fallback)).value_or(0.0f)};
        }
        TF_WARN("<%s>'s %s is driven by %s, which has no mapping; treated as unauthored",
                path.GetText(), name.GetText(),
                input.upstreamId.IsEmpty() ? "an unresolvable node" : input.upstreamId.GetText());
        return std::nullopt;
    }
    if (const auto scalar = _AsFloat(input.value)) {
        return cenote::wire::Constant<float>{*scalar};
    }
    return std::nullopt;
}

/// An input whose wire slot takes only a constant (specular_ior and the
/// coat pair). A connection cannot cross, so it degrades to the
/// connected node's declared fallback under one warning — or to
/// unauthored when the upstream has none to give.
std::optional<float> _ReadConstantSlot(const SdfPath& path, const HdMaterialNodeSchema& surface,
                                       const HdMaterialNodeContainerSchema& nodes,
                                       const TfToken& name) {
    const _Input input = _ResolveInput(surface, nodes, name);
    if (!input.connected) {
        return _AsFloat(input.value);
    }
    std::optional<float> fallback;
    if (input.upstreamId == _tokens->UsdUVTexture) {
        fallback = _Component(_TextureFallback(input.upstream), _ChannelFor(input.output));
    } else if (_IsReader(input)) {
        fallback = _AsFloat(_ParameterOr(input.upstream, _tokens->fallback)).value_or(0.0f);
    }
    if (fallback) {
        TF_WARN("<%s>'s %s is connected, but its wire slot takes only a constant; using the "
                "fallback %g",
                path.GetText(), name.GetText(), *fallback);
    } else {
        TF_WARN("<%s>'s %s is connected, but its wire slot takes only a constant; treated as "
                "unauthored",
                path.GetText(), name.GetText());
    }
    return fallback;
}

/// specularColor pinned to a constant for the workflow collapse: the
/// authored value, a connection's declared fallback, or the spec
/// default black. The caller's one warning covers the whole
/// approximation, hue and texture alike.
std::array<float, 3> _ReadSpecularColor(const HdMaterialNodeSchema& surface,
                                        const HdMaterialNodeContainerSchema& nodes) {
    const _Input input = _ResolveInput(surface, nodes, _tokens->specularColor);
    const std::array<float, 3> black{0.0f, 0.0f, 0.0f};
    if (!input.connected) {
        return _AsColor(input.value).value_or(black);
    }
    if (input.upstreamId == _tokens->UsdUVTexture) {
        const GfVec4f fallback = _TextureFallback(input.upstream);
        return {fallback[0], fallback[1], fallback[2]};
    }
    return _AsColor(_ParameterOr(input.upstream, _tokens->fallback)).value_or(black);
}

/// The total patch (Q9): every mapped field explicitly set — the wire's
/// own defaults (description.rs) where the USD input is unauthored — so
/// the wire material is a pure function of the network. Sparse fields
/// mean "don't touch" on the wire, and a sparse patch would strand
/// stale state when an input is de-authored.
cenote::wire::MaterialPatch _ReadMaterial(const SdfPath& path, const HdSceneIndexPrim& prim) {
    cenote::wire::MaterialPatch patch{
        .name = path.GetString(),
        .base_color = cenote::wire::Constant<std::array<float, 3>>{{0.8f, 0.8f, 0.8f}},
        .base_metalness = cenote::wire::Constant<float>{0.0f},
        .specular_roughness = cenote::wire::Constant<float>{0.3f},
        .specular_ior = 1.5f,
        .coat_weight = 0.0f,
        .coat_roughness = 0.0f,
        .emission_luminance = 0.0f,
        .emission_color = cenote::wire::Constant<std::array<float, 3>>{{1.0f, 1.0f, 1.0f}},
        .geometry_opacity = cenote::wire::Constant<float>{1.0f},
        .geometry_normal = cenote::wire::Clear{},
    };
    // The network at the explicit universal render context — 26.03
    // removed cross-context fallback, so the empty token is the one
    // correct read (Q4, D-102).
    const HdMaterialNetworkSchema network =
        HdMaterialSchema::GetFromParent(prim.dataSource)
            .GetMaterialNetwork(HdMaterialSchemaTokens->universalRenderContext);
    const HdMaterialNodeContainerSchema nodes = network.GetNodes();
    const HdMaterialConnectionSchema terminal =
        network.GetTerminals().Get(HdMaterialSchemaTokens->surface);
    const HdMaterialNodeSchema surface =
        nodes.Get(_TokenOr(terminal.GetUpstreamNodePath(), TfToken()));
    const TfToken identifier = _TokenOr(surface.GetNodeIdentifier(), TfToken());
    if (identifier != _tokens->UsdPreviewSurface) {
        // Published anyway: with defaults upserted server-side, a later
        // binding to this material stays legal and the look degrades to
        // gray instead of a rejected wave (Q4).
        TF_WARN("<%s> has no UsdPreviewSurface surface (%s); it wears OpenPBR defaults",
                path.GetText(),
                identifier.IsEmpty() ? "no surface terminal" : identifier.GetText());
        return patch;
    }
    // The Q5 table, row by row.
    if (auto diffuse = _ReadColorSlot(path, surface, nodes, _tokens->diffuseColor)) {
        patch.base_color = std::move(*diffuse);
    }
    if (auto roughness = _ReadScalarSlot(path, surface, nodes, _tokens->roughness)) {
        patch.specular_roughness = std::move(*roughness);
    }
    if (_AsInt(_ParameterOr(surface, _tokens->useSpecularWorkflow)).value_or(0) == 1) {
        // D-117 amends D-102: no specular tint crosses the wire, so the
        // legacy direct-F0 workflow collapses to specularColor's
        // luminance, re-expressed as the IOR with that reflectivity —
        // F0 = ((ior−1)/(ior+1))², inverted — and clamped to a plausible
        // dielectric range. metallic and ior belong to the other
        // workflow, so their defaults stand.
        const std::array<float, 3> color = _ReadSpecularColor(surface, nodes);
        const float f0 =
            std::clamp(0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2], 0.0f, 0.99f);
        const float root = std::sqrt(f0);
        patch.specular_ior = std::clamp((1.0f + root) / (1.0f - root), 1.0f, 5.0f);
        TF_WARN("<%s> uses the specular workflow; specularColor collapses to its luminance as "
                "IOR %.3g, and the hue is dropped",
                path.GetText(), static_cast<double>(*patch.specular_ior));
    } else {
        if (auto metallic = _ReadScalarSlot(path, surface, nodes, _tokens->metallic)) {
            patch.base_metalness = std::move(*metallic);
        }
        if (const auto ior = _ReadConstantSlot(path, surface, nodes, _tokens->ior)) {
            patch.specular_ior = ior;
        }
    }
    if (const auto clearcoat = _ReadConstantSlot(path, surface, nodes, _tokens->clearcoat)) {
        patch.coat_weight = clearcoat;
    }
    if (const auto coatRoughness =
            _ReadConstantSlot(path, surface, nodes, _tokens->clearcoatRoughness)) {
        patch.coat_roughness = coatRoughness;
    }
    if (auto emissive = _ReadColorSlot(path, surface, nodes, _tokens->emissiveColor)) {
        // The nit convention (Q5): the color or map is the tint at
        // luminance 1, so radiance equals emissiveColor exactly, and
        // absent emission keeps the luminance 0 that marks non-lights.
        patch.emission_color = std::move(*emissive);
        patch.emission_luminance = 1.0f;
    }
    if (auto opacity = _ReadScalarSlot(path, surface, nodes, _tokens->opacity)) {
        const float threshold =
            _AsFloat(_ParameterOr(surface, _tokens->opacityThreshold)).value_or(0.0f);
        if (threshold > 0.0f) {
            // The cutout: a constant binarizes here, geometry_opacity
            // being plain coverage; a texture cannot be thresholded per
            // texel on this side, so it ships soft under a warning (Q5).
            if (auto* constant = std::get_if<cenote::wire::Constant<float>>(&*opacity)) {
                constant->value = constant->value < threshold ? 0.0f : 1.0f;
            } else {
                TF_WARN("<%s> thresholds a textured opacity, which the wire cannot binarize; "
                        "rendered as soft coverage",
                        path.GetText());
            }
        }
        patch.geometry_opacity = std::move(*opacity);
    }
    const _Input normal = _ResolveInput(surface, nodes, _tokens->normal);
    if (normal.connected && normal.upstreamId == _tokens->UsdUVTexture) {
        _Texture texture = _ReadTexture(path, _tokens->normal, normal.upstream, nodes);
        if (texture.ref) {
            patch.geometry_normal =
                cenote::wire::Set<cenote::wire::TextureRef>{std::move(*texture.ref)};
        }
        // A broken normal map has already warned; Clear — the geometric
        // normal — is its only honest stand-in.
    } else if (normal.connected) {
        TF_WARN("<%s>'s normal is driven by %s, not a UsdUVTexture; ignored", path.GetText(),
                normal.upstreamId.IsEmpty() ? "an unresolvable node" : normal.upstreamId.GetText());
    } else if (const auto constant = _AsColor(normal.value)) {
        if (*constant != std::array<float, 3>{0.0f, 0.0f, 1.0f}) {
            TF_WARN("<%s> authors a constant normal, but only a normal map crosses the wire; "
                    "ignored",
                    path.GetText());
        }
    }
    // displacement and occlusion have no wire story at all; they warn
    // only when they would have changed the picture (Q5).
    const _Input displacement = _ResolveInput(surface, nodes, _tokens->displacement);
    if (displacement.connected || _AsFloat(displacement.value).value_or(0.0f) != 0.0f) {
        TF_WARN("<%s> authors displacement, which cenote does not read; ignored", path.GetText());
    }
    const _Input occlusion = _ResolveInput(surface, nodes, _tokens->occlusion);
    if (occlusion.connected || _AsFloat(occlusion.value).value_or(1.0f) != 1.0f) {
        TF_WARN("<%s> authors occlusion, which cenote does not read; ignored", path.GetText());
    }
    return patch;
}

} // namespace

HdCenoteMaterialPrim::HdCenoteMaterialPrim(const SdfPath& path,
                                           const HdsiPrimManagingSceneIndexObserver* observer,
                                           cenote::wire::ChangeSet* pending,
                                           std::shared_ptr<Registry> live,
                                           std::shared_ptr<const HdCenoteMeshPrim::Registry> meshes)
    : _path(path), _name(path.GetString()), _pending(pending), _live(std::move(live)),
      _meshes(std::move(meshes)) {
    const auto [it, inserted] = _live->try_emplace(_path, this);
    if (!inserted) {
        // A resync: inherit the previous translator's ledger and take
        // the registry slot so its destructor, which runs after this
        // constructor, goes quietly.
        _sent = it->second->_sent;
        it->second = this;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        _Withdraw();
        return;
    }
    _Reconcile(prim);
}

HdCenoteMaterialPrim::~HdCenoteMaterialPrim() {
    const auto it = _live->find(_path);
    if (it == _live->end() || it->second != this) {
        // Superseded by a resync; the successor answers for the path.
        return;
    }
    _live->erase(it);
    _Withdraw();
}

void HdCenoteMaterialPrim::_Dirty(const HdSceneIndexObserver::DirtiedPrimEntry& entry,
                                  const HdsiPrimManagingSceneIndexObserver* observer) {
    if (!entry.dirtyLocators.Intersects(HdMaterialSchema::GetDefaultLocator())) {
        return;
    }
    const HdSceneIndexPrim prim = observer->GetSceneIndex()->GetPrim(_path);
    if (!prim.dataSource) {
        return;
    }
    _Reconcile(prim);
}

void HdCenoteMaterialPrim::_Reconcile(const HdSceneIndexPrim& prim) {
    const bool born = !_sent;
    // No identical-patch suppression: like a mesh's no-op edit, a
    // byte-identical patch still travels and bumps the epoch (Q9).
    _pending->ops.push_back(_ReadMaterial(_path, prim));
    _sent = true;
    if (born) {
        // The birth hook (D-115): a material arriving after the meshes
        // that bind it generates no binding dirt, so the waiting meshes
        // repoint themselves now.
        _RepointMeshes();
    }
}

void HdCenoteMaterialPrim::_Withdraw() {
    if (!_sent) {
        return;
    }
    // The death hook, the birth walk's mirror: unpublish first so the
    // walk sees it, and every bound instance lets go before the Remove
    // is appended — one flush, validated post-set, nothing dangles.
    _sent = false;
    _RepointMeshes();
    _pending->ops.push_back(
        cenote::wire::Remove{.kind = cenote::wire::Kind::Material, .name = _name});
}

void HdCenoteMaterialPrim::_RepointMeshes() {
    for (const auto& [path, mesh] : *_meshes) {
        mesh->ResolveBinding();
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
