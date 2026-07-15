// Unit tests for the msgpack codec: every format-selection boundary the
// Writer can take, the Reader's strictness and truncation reporting, and
// two frames from the golden corpus (crates/cenote-wire/tests/golden/)
// embedded whole — one encoded, one decoded — so the codec is checked
// against real `rmp-serde` output, not just against itself. The corpus
// conformance runner covers the full corpus; these embedded copies rot
// only if the wire changes, which is exactly when they should fail.
#include "msgpack.hpp"

#include <cstdint>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using cenote::wire::Reader;
using cenote::wire::Writer;

using Bytes = std::vector<std::uint8_t>;

int failures = 0;

void fail(std::string_view message) {
    ++failures;
    std::println(stderr, "FAIL: {}", message);
}

void check(bool condition, std::string_view label) {
    if (!condition) {
        fail(label);
    }
}

std::string hex(std::span<const std::uint8_t> bytes) {
    std::string out;
    for (auto byte : bytes) {
        out += std::format("{}{:02x}", out.empty() ? "" : " ", byte);
    }
    return out;
}

void check_bytes(std::string_view label, const Bytes& got, const Bytes& want) {
    if (got != want) {
        fail(std::format("{}\n  want: {}\n  got:  {}", label, hex(want), hex(got)));
    }
}

/// Every unsigned boundary: the largest value of each format and the
/// smallest of the next.
void writer_unsigned_integers() {
    struct Case {
        std::uint64_t value;
        Bytes want;
    };
    const Case cases[] = {
        {0, {0x00}},
        {127, {0x7f}},
        {128, {0xcc, 0x80}},
        {255, {0xcc, 0xff}},
        {256, {0xcd, 0x01, 0x00}},
        {65535, {0xcd, 0xff, 0xff}},
        {65536, {0xce, 0x00, 0x01, 0x00, 0x00}},
        {0xffff'ffff, {0xce, 0xff, 0xff, 0xff, 0xff}},
        {0x1'0000'0000, {0xcf, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}},
        {0xffff'ffff'ffff'ffff, {0xcf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
    };
    for (const auto& [value, want] : cases) {
        Writer writer;
        writer.uint(value);
        check_bytes(std::format("uint({})", value), writer.bytes(), want);
    }
}

/// Signed boundaries: non-negative values must take the unsigned formats,
/// negative ones the smallest signed format that fits.
void writer_signed_integers() {
    struct Case {
        std::int64_t value;
        Bytes want;
    };
    const Case cases[] = {
        {0, {0x00}},
        {300, {0xcd, 0x01, 0x2c}},
        {-1, {0xff}},
        {-32, {0xe0}},
        {-33, {0xd0, 0xdf}},
        {-128, {0xd0, 0x80}},
        {-129, {0xd1, 0xff, 0x7f}},
        {-32768, {0xd1, 0x80, 0x00}},
        {-32769, {0xd2, 0xff, 0xff, 0x7f, 0xff}},
        {-2147483648, {0xd2, 0x80, 0x00, 0x00, 0x00}},
        {-2147483649, {0xd3, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff}},
    };
    for (const auto& [value, want] : cases) {
        Writer writer;
        writer.sint(value);
        check_bytes(std::format("sint({})", value), writer.bytes(), want);
    }
}

void writer_atoms() {
    Writer writer;
    writer.nil();
    writer.boolean(false);
    writer.boolean(true);
    writer.f32(0.0f);
    writer.f32(1.0f);
    writer.f32(-2.5f);
    check_bytes("nil, bools, f32s", writer.bytes(),
                {0xc0, 0xc2, 0xc3, 0xca, 0x00, 0x00, 0x00, 0x00, 0xca, 0x3f, 0x80, 0x00, 0x00, 0xca,
                 0xc0, 0x20, 0x00, 0x00});
}

void writer_strings() {
    struct Case {
        std::string value;
        Bytes prefix;
    };
    const Case cases[] = {
        {"", {0xa0}},
        {"tri", {0xa3}},
        {"µ", {0xa2}}, // two UTF-8 bytes — the length counts bytes, not code points
        {std::string(31, 'a'), {0xbf}},
        {std::string(32, 'a'), {0xd9, 0x20}},
        {std::string(255, 'a'), {0xd9, 0xff}},
        {std::string(256, 'a'), {0xda, 0x01, 0x00}},
        {std::string(65535, 'a'), {0xda, 0xff, 0xff}},
        {std::string(65536, 'a'), {0xdb, 0x00, 0x01, 0x00, 0x00}},
    };
    for (const auto& [value, prefix] : cases) {
        Writer writer;
        writer.str(value);
        Bytes want = prefix;
        want.insert(want.end(), value.begin(), value.end());
        check_bytes(std::format("str of {} bytes", value.size()), writer.bytes(), want);
    }
}

void writer_headers() {
    struct Case {
        std::uint8_t fix_base;
        std::uint8_t format16;
        std::uint8_t format32;
        void (Writer::*method)(std::size_t);
        const char* label;
    };
    const Case cases[] = {
        {0x90, 0xdc, 0xdd, &Writer::array_header, "array"},
        {0x80, 0xde, 0xdf, &Writer::map_header, "map"},
    };
    for (const auto& c : cases) {
        {
            Writer writer;
            (writer.*c.method)(0);
            check_bytes(std::format("{}_header(0)", c.label), writer.bytes(), {c.fix_base});
        }
        {
            Writer writer;
            (writer.*c.method)(15);
            check_bytes(std::format("{}_header(15)", c.label), writer.bytes(),
                        {static_cast<std::uint8_t>(c.fix_base | 15)});
        }
        {
            Writer writer;
            (writer.*c.method)(16);
            check_bytes(std::format("{}_header(16)", c.label), writer.bytes(),
                        {c.format16, 0x00, 0x10});
        }
        {
            Writer writer;
            (writer.*c.method)(65535);
            check_bytes(std::format("{}_header(65535)", c.label), writer.bytes(),
                        {c.format16, 0xff, 0xff});
        }
        {
            Writer writer;
            (writer.*c.method)(65536);
            check_bytes(std::format("{}_header(65536)", c.label), writer.bytes(),
                        {c.format32, 0x00, 0x01, 0x00, 0x00});
        }
    }
}

void writer_take_resets() {
    Writer writer;
    writer.uint(7);
    const Bytes first = writer.take();
    check(first == Bytes{0x07}, "take() yields the payload");
    check(writer.bytes().empty(), "take() leaves the writer empty");
    writer.nil();
    check(writer.bytes() == Bytes{0xc0}, "the writer is reusable after take()");
}

/// Everything the Writer encodes, the Reader hands back — across every
/// format each accepts.
void reader_round_trips() {
    for (std::uint64_t value :
         {std::uint64_t{0}, std::uint64_t{127}, std::uint64_t{128}, std::uint64_t{65535},
          std::uint64_t{65536}, std::uint64_t{0xffff'ffff}, std::uint64_t{0x1'0000'0000}}) {
        Writer writer;
        writer.uint(value);
        Reader reader(writer.bytes());
        const auto back = reader.uint();
        check(back && *back == value && reader.remaining() == 0,
              std::format("uint({}) round-trips", value));
    }
    for (std::size_t length :
         {std::size_t{0}, std::size_t{31}, std::size_t{32}, std::size_t{256}, std::size_t{65536}}) {
        const std::string value(length, 'x');
        Writer writer;
        writer.str(value);
        Reader reader(writer.bytes());
        const auto back = reader.str();
        check(back && *back == value && reader.remaining() == 0,
              std::format("str of {} bytes round-trips", length));
    }
    for (std::size_t count :
         {std::size_t{0}, std::size_t{15}, std::size_t{16}, std::size_t{65536}}) {
        Writer arrays;
        arrays.array_header(count);
        Reader array_reader(arrays.bytes());
        const auto array_back = array_reader.array_header();
        check(array_back && *array_back == count && array_reader.remaining() == 0,
              std::format("array_header({}) round-trips", count));

        Writer maps;
        maps.map_header(count);
        Reader map_reader(maps.bytes());
        const auto map_back = map_reader.map_header();
        check(map_back && *map_back == count && map_reader.remaining() == 0,
              std::format("map_header({}) round-trips", count));
    }
}

/// The Reader accepts exactly what `rmp-serde` emits for the matching
/// Rust type — a signed format where an unsigned belongs is drift, and
/// the error carries the byte and offset that prove it.
void reader_strictness() {
    {
        const Bytes payload = {0xd0, 0x05}; // int8, valid msgpack, never emitted for a uint
        Reader reader(payload);
        const auto result = reader.uint();
        check(!result && result.error().offset == 0 && result.error().found == std::uint8_t{0xd0},
              "uint() refuses a signed format");
    }
    {
        const Bytes payload = {0xce, 0x00, 0x00, 0x00, 0x2a}; // uint32 where a string belongs
        Reader reader(payload);
        const auto result = reader.str();
        check(!result && result.error().offset == 0 && result.error().found == std::uint8_t{0xce},
              "str() refuses a non-string format");
    }
    {
        const Bytes payload = {0x81}; // fixmap where an array belongs
        Reader reader(payload);
        const auto result = reader.array_header();
        check(!result && result.error().found == std::uint8_t{0x81},
              "array_header() refuses a map");
    }
    {
        const Bytes payload = {0x91}; // fixarray where a map belongs
        Reader reader(payload);
        const auto result = reader.map_header();
        check(!result && result.error().found == std::uint8_t{0x91},
              "map_header() refuses an array");
    }
}

void reader_truncation() {
    {
        Reader reader(std::span<const std::uint8_t>{});
        const auto result = reader.uint();
        check(!result && result.error().offset == 0 && !result.error().found,
              "an empty payload reports truncation at offset 0");
    }
    {
        const Bytes payload = {0xcd, 0x01}; // uint16 missing its second byte
        Reader reader(payload);
        const auto result = reader.uint();
        check(!result && result.error().offset == 2 && !result.error().found,
              "a cut-short uint16 reports truncation");
    }
    {
        const Bytes payload = {0xa5, 'h', 'i'}; // fixstr promising 5 bytes, carrying 2
        Reader reader(payload);
        const auto result = reader.str();
        check(!result && result.error().offset == 3 && !result.error().found,
              "a cut-short string reports truncation");
    }
}

/// request-resize.bin from the golden corpus: `Resize { width: 1920,
/// height: 1080 }` — a struct variant is a one-entry map from variant name
/// to a positional array of its fields.
void golden_resize_encodes() {
    Writer writer;
    writer.map_header(1);
    writer.str("Resize");
    writer.array_header(2);
    writer.uint(1920);
    writer.uint(1080);
    check_bytes(
        "request-resize golden", writer.bytes(),
        {0x81, 0xa6, 'R', 'e', 's', 'i', 'z', 'e', 0x92, 0xcd, 0x07, 0x80, 0xcd, 0x04, 0x38});
}

/// response-welcome.bin from the golden corpus: `Welcome { protocol: 1,
/// fb: FbDesc { shm_name: "/cenote-12345-1", bytes: 36868096, width: 1280,
/// height: 720 } }` — decoded field by field, the way the response decoder
/// will walk it.
void golden_welcome_decodes() {
    const Bytes golden = {0x81, 0xa7, 'W',  'e',  'l',  'c',  'o',  'm',  'e',  0x92,
                          0x01, 0x94, 0xaf, '/',  'c',  'e',  'n',  'o',  't',  'e',
                          '-',  '1',  '2',  '3',  '4',  '5',  '-',  '1',  0xce, 0x02,
                          0x32, 0x90, 0x00, 0xcd, 0x05, 0x00, 0xcd, 0x02, 0xd0};
    Reader reader(golden);
    const auto entries = reader.map_header();
    check(entries && *entries == 1, "welcome: the envelope is a one-entry map");
    const auto variant = reader.str();
    check(variant && *variant == "Welcome", "welcome: the variant name");
    const auto fields = reader.array_header();
    check(fields && *fields == 2, "welcome: two fields");
    const auto protocol = reader.uint();
    check(protocol && *protocol == 1, "welcome: protocol 1");
    const auto fb_fields = reader.array_header();
    check(fb_fields && *fb_fields == 4, "welcome: FbDesc has four fields");
    const auto shm_name = reader.str();
    check(shm_name && *shm_name == "/cenote-12345-1", "welcome: shm name");
    const auto bytes = reader.uint();
    check(bytes && *bytes == 36868096, "welcome: segment bytes");
    const auto width = reader.uint();
    check(width && *width == 1280, "welcome: width");
    const auto height = reader.uint();
    check(height && *height == 720, "welcome: height");
    check(reader.remaining() == 0, "welcome: fully consumed");
}

void framing() {
    check(cenote::wire::frame_prefix(0x01020304) ==
              (std::array<std::uint8_t, 4>{0x04, 0x03, 0x02, 0x01}),
          "the frame prefix is little-endian");
    for (std::uint32_t length : {std::uint32_t{0}, std::uint32_t{15}, std::uint32_t{0x0102'0304},
                                 std::uint32_t{0xffff'ffff}}) {
        check(cenote::wire::framed_length(cenote::wire::frame_prefix(length)) == length,
              std::format("frame length {} round-trips", length));
    }
}

} // namespace

// A failed check reports by escaping: terminate prints the what() and the
// exit is nonzero — that is the test contract, not an oversight.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    writer_unsigned_integers();
    writer_signed_integers();
    writer_atoms();
    writer_strings();
    writer_headers();
    writer_take_resets();
    reader_round_trips();
    reader_strictness();
    reader_truncation();
    golden_resize_encodes();
    golden_welcome_decodes();
    framing();
    if (failures != 0) {
        std::println(stderr, "{} check(s) failed", failures);
        return 1;
    }
    std::println("msgpack-test: all checks passed");
    return 0;
}
