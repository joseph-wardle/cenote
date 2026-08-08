// The scene half of the wire mirror — cenote-wire's scene.rs, type for
// type and field for field, keeping Rust's exact names. std::optional
// stands in for Option, std::variant for the data-carrying enums, and
// designated-initializer construction gives patches the same shape as
// Rust's struct-update syntax: name the fields you mean, and every
// omitted optional defaults to the wire's absent None. The `{}` member
// defaults on the patches mirror the Rust side's `Default` derive — and
// keep -Wextra's missing-field-initializers check quiet for the patch
// idiom while it still guards every always-required field.
//
// The semantics ride along unchanged: a patch targets an object by name
// (get-or-create), only present fields overwrite, removal errors if the
// target is missing or a reference would strand. Per-field meaning lives
// on the Rust originals; duplicating those docs here would only let the
// two drift.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "msgpack.hpp"

namespace cenote::wire {

/// The seven object kinds — mirror of `Kind`.
enum class Kind {
    Mesh,
    Instance,
    Material,
    Light,
    Camera,
    Environment,
    Settings,
};

/// How an image's stored values map to linear light — mirror of
/// `ColorSpace`: sRGB-encoded (linearized on sampling) or already linear.
enum class ColorSpace {
    Srgb,
    Linear,
};

/// One channel of a source image — mirror of `Channel`: the component a
/// scalar slot samples.
enum class Channel {
    R,
    G,
    B,
    A,
};

/// A reference to an image file — mirror of `TextureRef`. Send absolute
/// paths; an empty `color_space` derives it from the slot; an empty
/// `channel` means red (color and normal slots ignore it).
struct TextureRef {
    std::string path;
    std::optional<ColorSpace> color_space;
    std::optional<Channel> channel;
};

/// `Reset`'s Clear alternative — the renderer's `Some(None)`, restoring
/// the field's default.
struct Clear {};

/// `Reset`'s Set alternative — the renderer's `Some(Some(value))`.
template <typename T> struct Set {
    T value;
};

/// The wire spelling of a doubly-optional patch field — mirror of
/// `Reset<T>`. A `std::optional<Reset<T>>` keeps all three states
/// distinct on the wire: absent leaves the field alone, `Clear` restores
/// its default, `Set` overwrites it.
template <typename T> using Reset = std::variant<Clear, Set<T>>;

/// `Texturable`'s Constant alternative: the same value across the whole
/// surface.
template <typename T> struct Constant {
    T value;
};

/// A constant-or-textured material parameter — mirror of `Texturable<T>`;
/// the `TextureRef` alternative is Rust's `Texture` variant, sampled from
/// an image at the hit's UV.
template <typename T> using Texturable = std::variant<Constant<T>, TextureRef>;

/// `Transform`'s Trs alternative: translate · rotate · scale — scale
/// first, rotations about world X, then Y, then Z, in degrees, then
/// translation in meters.
struct Trs {
    std::array<float, 3> translate;
    std::array<float, 3> rotate_degrees;
    std::array<float, 3> scale;
};

/// `Transform`'s Matrix alternative: the top three rows of an affine
/// matrix — translation in the last column, the implied bottom row
/// `0 0 0 1`.
struct Matrix {
    std::array<std::array<float, 4>, 3> rows;
};

/// An object-to-world placement — mirror of `Transform`.
using Transform = std::variant<Trs, Matrix>;

/// `MeshSource`'s Inline alternative: geometry spelled out in the
/// message. Positions in meters, object space; unit normals one per
/// position (absent means the server derives them); UVs one per position;
/// triangles counter-clockwise-outward index triples into `positions`.
struct Inline {
    std::vector<std::array<float, 3>> positions;
    std::optional<std::vector<std::array<float, 3>>> normals;
    std::optional<std::vector<std::array<float, 2>>> uvs;
    std::vector<std::array<std::uint32_t, 3>> triangles;
};

/// `MeshSource`'s Ply alternative: geometry loaded server-side from an
/// absolute `.ply` path.
struct Ply {
    std::string path;
};

/// A mesh's geometry payload — mirror of `MeshSource`.
using MeshSource = std::variant<Inline, Ply>;

/// `Light`'s Distant alternative: parallel light from infinitely far
/// away — the direction it travels, and the irradiance on a facing
/// surface (W/m², linear Rec.709).
struct Distant {
    std::array<float, 3> direction;
    std::array<float, 3> irradiance;
};

/// `Light`'s Point alternative: an isotropic point — position in meters,
/// world space, and radiant intensity (W/sr, linear Rec.709).
struct Point {
    std::array<float, 3> position;
    std::array<float, 3> intensity;
};

/// A delta light's definition — mirror of `Light`.
using Light = std::variant<Distant, Point>;

/// Mirror of `MeshPatch`.
struct MeshPatch {
    std::string name;
    std::optional<MeshSource> source{};
};

/// Mirror of `InstancePatch`. `transforms` is the placements array —
/// one element per copy, the whole array replaces, and empty is legal
/// (resident, places nothing).
struct InstancePatch {
    std::string name;
    std::optional<std::string> mesh{};
    std::optional<std::string> material{};
    std::optional<std::vector<Transform>> transforms{};
    std::optional<bool> camera_visible{};
};

/// Mirror of `MaterialPatch` — the wide surface the drift guard exists
/// for. Field meanings and defaults are OpenPBR's.
struct MaterialPatch {
    std::string name;
    std::optional<Texturable<std::array<float, 3>>> base_color{};
    std::optional<float> base_diffuse_roughness{};
    std::optional<Texturable<float>> base_metalness{};
    std::optional<float> specular_weight{};
    std::optional<Texturable<float>> specular_roughness{};
    std::optional<float> specular_ior{};
    std::optional<float> transmission_weight{};
    std::optional<std::array<float, 3>> transmission_color{};
    std::optional<float> transmission_depth{};
    std::optional<std::array<float, 3>> transmission_scatter{};
    std::optional<float> transmission_scatter_anisotropy{};
    std::optional<float> coat_weight{};
    std::optional<std::array<float, 3>> coat_color{};
    std::optional<float> coat_roughness{};
    std::optional<float> coat_ior{};
    std::optional<float> coat_darkening{};
    std::optional<float> fuzz_weight{};
    std::optional<std::array<float, 3>> fuzz_color{};
    std::optional<float> fuzz_roughness{};
    std::optional<float> emission_luminance{};
    std::optional<Texturable<std::array<float, 3>>> emission_color{};
    std::optional<Texturable<float>> geometry_opacity{};
    std::optional<bool> geometry_thin_walled{};
    std::optional<Reset<TextureRef>> geometry_normal{};
};

/// Mirror of `LightPatch`.
struct LightPatch {
    std::string name;
    std::optional<Light> light{};
};

/// Mirror of `CameraPatch`. As on the Rust side, this cannot move the
/// view — the session's inputs-lane camera overwrites the scene camera
/// every wave; use `SetCamera`. It exists because the mirror is total.
struct CameraPatch {
    std::string name;
    std::optional<std::array<float, 3>> position{};
    std::optional<std::array<float, 3>> look_at{};
    std::optional<std::array<float, 3>> up{};
    std::optional<float> vfov_degrees{};
    std::optional<Reset<float>> focus_distance{};
    std::optional<float> aperture_radius{};
};

/// Mirror of `EnvironmentPatch`. The path is doubly optional like the
/// material's normal map: absent leaves the image alone, `Clear` restores
/// the constant white sky.
struct EnvironmentPatch {
    std::string name;
    std::optional<Reset<std::string>> path{};
    std::optional<std::array<float, 3>> tint{};
    std::optional<Transform> transform{};
};

/// Mirror of `SettingsPatch`.
struct SettingsPatch {
    std::string name;
    std::optional<std::array<std::uint32_t, 2>> resolution{};
    std::optional<std::uint32_t> spp{};
    std::optional<std::uint32_t> max_bounces{};
    std::optional<std::uint32_t> seed{};
};

/// `Op`'s Remove alternative — mirror of the `Remove(Kind, String)` tuple
/// variant: delete an object outright (renames arrive as remove +
/// re-insert). The tuple's unnamed fields get the obvious names.
struct Remove {
    Kind kind;
    std::string name;
};

/// One edit — mirror of `Op`. Each alternative keeps its Rust variant
/// name through `encode`: the patches drop their `Patch` suffix, and
/// `Remove` is itself. Rust boxes the material for size; here every `Op`
/// is simply material-sized, which a change-set list can afford.
using Op = std::variant<MeshPatch, InstancePatch, MaterialPatch, LightPatch, CameraPatch,
                        EnvironmentPatch, SettingsPatch, Remove>;

/// An ordered list of edits, applied atomically server-side — mirror of
/// `ChangeSet`, the payload of `Replace` and `Apply`.
struct ChangeSet {
    std::vector<Op> ops;
};

/// The classic lambda-overload set, for `std::visit`ing the wire's
/// variants.
template <typename... Fns> struct Overloaded : Fns... {
    using Fns::operator()...;
};

// Appends a value's wire encoding to `writer` — one overload per wire
// type, fields in Rust declaration order, variant names spelled the way
// `rmp-serde` pins them in the golden corpus. Together these reproduce
// the Rust encoder byte for byte.

inline void encode(Writer& writer, bool value) { writer.boolean(value); }
inline void encode(Writer& writer, float value) { writer.f32(value); }
inline void encode(Writer& writer, std::uint32_t value) { writer.uint(value); }
inline void encode(Writer& writer, std::uint64_t value) { writer.uint(value); }
inline void encode(Writer& writer, const std::string& value) { writer.str(value); }

void encode(Writer& writer, Kind value);
void encode(Writer& writer, ColorSpace value);
void encode(Writer& writer, Channel value);
void encode(Writer& writer, const TextureRef& value);
void encode(Writer& writer, const Transform& value);
void encode(Writer& writer, const MeshSource& value);
void encode(Writer& writer, const Light& value);
void encode(Writer& writer, const MeshPatch& value);
void encode(Writer& writer, const InstancePatch& value);
void encode(Writer& writer, const MaterialPatch& value);
void encode(Writer& writer, const LightPatch& value);
void encode(Writer& writer, const CameraPatch& value);
void encode(Writer& writer, const EnvironmentPatch& value);
void encode(Writer& writer, const SettingsPatch& value);
void encode(Writer& writer, const Op& value);
void encode(Writer& writer, const ChangeSet& value);

// The composite shapes: an absent optional is nil; std::array and
// std::vector are MessagePack arrays; Reset and Texturable follow the
// enum rule — a bare variant-name string when the variant carries
// nothing, a one-entry map from name to value when it does.

template <typename T> void encode(Writer& writer, const std::optional<T>& value) {
    if (value) {
        encode(writer, *value);
    } else {
        writer.nil();
    }
}

template <typename T, std::size_t N> void encode(Writer& writer, const std::array<T, N>& value) {
    writer.array_header(N);
    for (const T& element : value) {
        encode(writer, element);
    }
}

template <typename T> void encode(Writer& writer, const std::vector<T>& value) {
    writer.array_header(value.size());
    for (const T& element : value) {
        encode(writer, element);
    }
}

template <typename T> void encode(Writer& writer, const Reset<T>& value) {
    if (const Set<T>* set = std::get_if<Set<T>>(&value)) {
        writer.map_header(1);
        writer.str("Set");
        encode(writer, set->value);
    } else {
        writer.str("Clear");
    }
}

template <typename T> void encode(Writer& writer, const Texturable<T>& value) {
    writer.map_header(1);
    if (const Constant<T>* constant = std::get_if<Constant<T>>(&value)) {
        writer.str("Constant");
        encode(writer, constant->value);
    } else {
        writer.str("Texture");
        encode(writer, std::get<TextureRef>(value));
    }
}

} // namespace cenote::wire
