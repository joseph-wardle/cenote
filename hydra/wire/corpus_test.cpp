// The C++ half of the wire drift guard: the same 12 cases
// crates/cenote-wire/tests/corpus.rs pins, constructed here with
// designated initializers, encoded through the production encode()
// overloads, and byte-compared against the checked-in goldens (the
// directory arrives as argv[1]). Coverage is symmetric — a golden file no
// case claims means the Rust corpus grew without this mirror, and a case
// without a golden means the reverse; both fail. The four responses also
// travel the other direction, through decode_response() and back to the
// same bytes, and three refusal probes hold the decoder to its documented
// strictness.
#include "protocol.hpp"
#include "scene.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace cenote::wire;

using Bytes = std::vector<std::uint8_t>;

int failures = 0;

void fail(std::string_view message) {
    ++failures;
    std::println(stderr, "FAIL: {}", message);
}

Bytes read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::size_t first_divergence(const Bytes& got, const Bytes& want) {
    const std::size_t limit = std::min(got.size(), want.size());
    for (std::size_t offset = 0; offset < limit; ++offset) {
        if (got[offset] != want[offset]) {
            return offset;
        }
    }
    return limit;
}

/// A hex window around `offset`: up to eight bytes of run-up, the byte at
/// the divergence bracketed — or `[end]` when that buffer stops short.
std::string window(const Bytes& bytes, std::size_t offset) {
    const std::size_t begin = offset < 8 ? 0 : offset - 8;
    const std::size_t end = std::min(bytes.size(), offset + 8);
    std::string out = begin > 0 ? ".." : "";
    for (std::size_t index = begin; index < end; ++index) {
        const char* open = index == offset ? "[" : "";
        const char* close = index == offset ? "]" : "";
        out += std::format("{}{}{:02x}{}", out.empty() ? "" : " ", open, bytes[index], close);
    }
    if (offset >= bytes.size()) {
        out += out.empty() ? "[end]" : " [end]";
    } else if (end < bytes.size()) {
        out += " ..";
    }
    return out;
}

void check_bytes(std::string_view name, const Bytes& got, const Bytes& want) {
    if (got == want) {
        return;
    }
    const std::size_t offset = first_divergence(got, want);
    fail(std::format("{} diverges at offset {} (want {} bytes, got {})\n  want: {}\n  got:  {}",
                     name, offset, want.size(), got.size(), window(want, offset),
                     window(got, offset)));
}

template <typename T> Bytes encoded(const T& value) {
    Writer writer;
    encode(writer, value);
    return writer.take();
}

/// corpus.rs's `genesis()`: one ChangeSet holding every Op variant, every
/// patch field present at least once, both spellings of every
/// two-spelling type, and each doubly-optional field in all three states.
ChangeSet genesis() {
    ChangeSet set;
    set.ops = {
        MeshPatch{
            .name = "tri",
            .source =
                Inline{
                    .positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
                    .normals = std::vector<std::array<float, 3>>{{0.0f, 0.0f, 1.0f},
                                                                 {0.0f, 0.0f, 1.0f},
                                                                 {0.0f, 0.0f, 1.0f}},
                    .uvs =
                        std::vector<std::array<float, 2>>{{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}},
                    .triangles = {{0, 1, 2}},
                },
        },
        MeshPatch{
            .name = "статуя",
            .source = Ply{.path = "/scènes/geo/статуя.ply"},
        },
        InstancePatch{
            .name = "thing",
            .mesh = "tri",
            .material = "m-set",
            .transforms = std::vector<Transform>{Trs{
                .translate = {1.0f, 2.0f, 3.0f},
                .rotate_degrees = {0.0f, 90.0f, 0.0f},
                .scale = {2.0f, 2.0f, 2.0f},
            }},
            .camera_visible = false,
        },
        // A multi-element transforms array, both spellings in one — the
        // array-instancer form the field's vector exists for.
        InstancePatch{
            .name = "matrix-thing",
            .mesh = "статуя",
            .material = "m-clear",
            .transforms = std::vector<Transform>{Matrix{.rows = {{{1.0f, 0.0f, 0.0f, 4.0f},
                                                                  {0.0f, 1.0f, 0.0f, 5.0f},
                                                                  {0.0f, 0.0f, 1.0f, 6.0f}}}},
                                                 Trs{
                                                     .translate = {-4.0f, 0.0f, 4.0f},
                                                     .rotate_degrees = {0.0f, 0.0f, 45.0f},
                                                     .scale = {1.0f, 1.0f, 1.0f},
                                                 }},
            .camera_visible = true,
        },
        // The empty array is legal and distinct from an absent field:
        // resident, places nothing (a fully-masked instancer).
        InstancePatch{
            .name = "masked",
            .mesh = "tri",
            .material = "m-leave",
            .transforms = std::vector<Transform>{},
            .camera_visible = {},
        },
        MaterialPatch{
            .name = "m-set",
            .base_color = TextureRef{.path = "/textures/дерево.png",
                                     .color_space = ColorSpace::Srgb,
                                     .channel = {}},
            .base_diffuse_roughness = 0.25f,
            .base_metalness = Constant{1.0f},
            .specular_weight = 0.9f,
            .specular_roughness = TextureRef{.path = "/textures/rough.png",
                                             .color_space = ColorSpace::Linear,
                                             .channel = Channel::G},
            .specular_ior = 1.45f,
            .transmission_weight = 0.5f,
            .transmission_color = std::array{0.9f, 0.95f, 1.0f},
            .transmission_depth = 0.1f,
            .transmission_scatter = std::array{0.05f, 0.4f, 0.7f},
            .transmission_scatter_anisotropy = 0.6f,
            .coat_weight = 1.0f,
            .coat_color = std::array{1.0f, 0.9f, 0.8f},
            .coat_roughness = 0.05f,
            .coat_ior = 1.6f,
            .coat_darkening = 0.75f,
            .fuzz_weight = 0.2f,
            .fuzz_color = std::array{1.0f, 1.0f, 0.9f},
            .fuzz_roughness = 0.6f,
            .emission_luminance = 1000.0f,
            .emission_color = Constant{std::array{1.0f, 0.5f, 0.25f}},
            .geometry_opacity =
                TextureRef{.path = "/textures/mask.png", .color_space = {}, .channel = Channel::A},
            .geometry_thin_walled = true,
            // A channel on a normal slot is inert server-side; the mirror
            // is total, so its bytes are pinned regardless.
            .geometry_normal = Set{TextureRef{
                .path = "/textures/normal.png", .color_space = {}, .channel = Channel::B}},
        },
        MaterialPatch{
            .name = "m-clear",
            .geometry_normal = Clear{},
        },
        MaterialPatch{
            .name = "m-leave",
            .base_metalness = Constant{0.0f},
            .specular_roughness =
                TextureRef{.path = "/textures/orm.png", .color_space = {}, .channel = Channel::R},
            .geometry_opacity = Constant{1.0f},
        },
        LightPatch{
            .name = "sun",
            .light = Distant{.direction = {-0.3f, -1.0f, -0.2f}, .irradiance = {3.0f, 2.9f, 2.7f}},
        },
        LightPatch{
            .name = "bulb",
            .light = Point{.position = {0.0f, 2.0f, 0.0f}, .intensity = {10.0f, 9.0f, 8.0f}},
        },
        CameraPatch{
            .name = "cam-set",
            .position = std::array{0.0f, 1.0f, 5.0f},
            .look_at = std::array{0.0f, 1.0f, 0.0f},
            .up = std::array{0.0f, 1.0f, 0.0f},
            .vfov_degrees = 45.0f,
            .focus_distance = Set{5.0f},
            .aperture_radius = 0.02f,
        },
        CameraPatch{
            .name = "cam-clear",
            .focus_distance = Clear{},
        },
        CameraPatch{
            .name = "cam-leave",
            .vfov_degrees = 30.0f,
        },
        // The environment patch three times over, like the material: the
        // image set, cleared (back to the constant white sky), and left
        // alone — the path's three doubly-optional states.
        EnvironmentPatch{
            .name = "ciel-d'été",
            .path = Set{std::string{"/scènes/небо.exr"}},
            .tint = std::array{1.0f, 0.9f, 0.8f},
            .transform =
                Trs{
                    .translate = {0.0f, 0.0f, 0.0f},
                    .rotate_degrees = {0.0f, 45.0f, 0.0f},
                    .scale = {1.0f, 1.0f, 1.0f},
                },
        },
        EnvironmentPatch{
            .name = "env-clear",
            .path = Clear{},
        },
        EnvironmentPatch{
            .name = "env-leave",
            .tint = std::array{0.5f, 0.5f, 0.5f},
        },
        SettingsPatch{
            .name = "main",
            .resolution = std::array<std::uint32_t, 2>{1920, 1080},
            .spp = 256U,
            .max_bounces = 8U,
            .seed = 7U,
        },
        Remove{.kind = Kind::Mesh, .name = "outgrown"},
    };
    return set;
}

} // namespace

// A failed check reports by escaping: terminate prints the what() and the
// exit is nonzero — that is the test contract, not an oversight.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    if (argc != 2) {
        std::println(stderr, "usage: corpus-test <golden-dir>");
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    if (!std::filesystem::is_directory(dir)) {
        std::println(stderr, "corpus-test: {} is not a directory", dir.string());
        return 2;
    }

    const std::vector<std::pair<std::string, Request>>
        requests =
            {
                {"request-hello",
                 Hello{.protocol = PROTOCOL, .token = "0123456789abcdef0123456789abcdef"}},
                {"request-ping", Ping{}},
                {"request-resize", Resize{.width = 1920, .height = 1080}},
                {"request-set-camera-focused", SetCamera{.camera =
                                                             Camera{
                                                                 .position = {1.0f, 2.5f, -3.0f},
                                                                 .look_at = {0.0f, 1.0f, 0.0f},
                                                                 .up = {0.0f, 1.0f, 0.0f},
                                                                 .vfov_degrees = 36.5f,
                                                                 .focus_distance = 4.25f,
                                                                 .aperture_radius = 0.05f,
                                                             }}},
                {"request-set-camera-auto-focus", SetCamera{.camera =
                                                                Camera{
                                                                    .position = {0.0f, 0.0f, 5.0f},
                                                                    .look_at = {0.0f, 0.0f, 0.0f},
                                                                    .up = {0.0f, 1.0f, 0.0f},
                                                                    .vfov_degrees = 40.0f,
                                                                    .focus_distance = {},
                                                                    .aperture_radius = 0.0f,
                                                                }}},
                {"request-replace-empty", Replace{.changes = {}}},
                {"request-replace-genesis", Replace{.changes = genesis()}},
                {"request-apply-removes",
                 Apply{
                     .changes =
                         ChangeSet{.ops =
                                       {
                                           Remove{.kind = Kind::Mesh, .name = "mesh"},
                                           Remove{.kind = Kind::Instance, .name = "instance"},
                                           Remove{.kind = Kind::Material, .name = "材料/µaterial"},
                                           Remove{.kind = Kind::Light, .name = "light"},
                                           Remove{.kind = Kind::Camera, .name = "camera"},
                                           Remove{.kind = Kind::Environment, .name = "environment"},
                                           Remove{.kind = Kind::Settings, .name = "settings"},
                                       }}}},
            };

    const std::vector<std::pair<std::string, Response>> responses = {
        {"response-welcome", Welcome{.protocol = PROTOCOL,
                                     .fb =
                                         FbDesc{
                                             .shm_name = "/cenote-12345-1",
                                             .bytes = 4096 + 2 * (1280ULL * 720 * 20),
                                             .width = 1280,
                                             .height = 720,
                                         }}},
        {"response-ack-clean", Ack{.rejected = {}, .epoch = 7}},
        {"response-ack-rejected",
         Ack{.rejected =
                 {
                     "instance \"chair\" references a mesh \"seat\" that does not exist",
                     "environment \"ciel-d'été\" references \"/scènes/небо.exr\", which does "
                     "not exist",
                 },
             // Past u32, deliberately: the epoch is a u64 on the wire,
             // and this pins the width against the Rust corpus.
             .epoch = 4'294'967'296}},
        {"response-resized", Resized{.fb =
                                         FbDesc{
                                             .shm_name = "/cenote-12345-2",
                                             .bytes = 4096 + 2 * (640ULL * 480 * 20),
                                             .width = 640,
                                             .height = 480,
                                         },
                                     .epoch = 9}},
    };

    // The symmetric set check: every file in the golden directory belongs
    // to exactly one case here, and every case has its file.
    std::set<std::string> claimed;
    for (const auto& [name, request] : requests) {
        claimed.insert(name + ".bin");
    }
    for (const auto& [name, response] : responses) {
        claimed.insert(name + ".bin");
    }
    std::set<std::string> present;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        present.insert(entry.path().filename().string());
    }
    for (const auto& file : present) {
        if (!claimed.contains(file)) {
            fail(std::format("{} has no case here — the Rust corpus grew without this mirror",
                             file));
        }
    }
    for (const auto& file : claimed) {
        if (!present.contains(file)) {
            fail(std::format("{} is missing from the golden directory", file));
        }
    }

    for (const auto& [name, request] : requests) {
        if (present.contains(name + ".bin")) {
            check_bytes(name, encoded(request), read_file(dir / (name + ".bin")));
        }
    }
    for (const auto& [name, response] : responses) {
        if (!present.contains(name + ".bin")) {
            continue;
        }
        const Bytes golden = read_file(dir / (name + ".bin"));
        check_bytes(name + " (encode)", encoded(response), golden);
        const auto decoded = decode_response(golden);
        if (!decoded) {
            fail(std::format("{} does not decode (expected {} at offset {})", name,
                             decoded.error().expected, decoded.error().offset));
            continue;
        }
        check_bytes(name + " (decode round-trip)", encoded(*decoded), golden);
    }

    // The decoder's documented strictness, one probe per axis: a request
    // payload is not a Response, an unknown variant name is refused, and
    // bytes after a complete message are refused.
    if (present.contains("request-ping.bin") &&
        decode_response(read_file(dir / "request-ping.bin"))) {
        fail("a Ping request decoded as a Response");
    }
    const Bytes unknown = {0x81, 0xa2, 'N', 'o', 0xc0};
    if (decode_response(unknown)) {
        fail("an unknown variant name decoded");
    }
    if (present.contains("response-ack-clean.bin")) {
        Bytes trailing = read_file(dir / "response-ack-clean.bin");
        trailing.push_back(0x00);
        if (decode_response(trailing)) {
            fail("trailing bytes after a complete message were accepted");
        }
    }

    if (failures != 0) {
        std::println(stderr, "{} check(s) failed", failures);
        return 1;
    }
    std::println("corpus-test: {} goldens byte-exact, both directions", claimed.size());
    return 0;
}
