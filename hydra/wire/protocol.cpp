#include "protocol.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cenote::wire {

namespace {

/// A value-level mismatch: the bytes decoded, but not to the shape the
/// envelope requires. There is no format byte to blame, so `found` stays
/// empty and the offset points just past the offending value.
DecodeError mismatch(const Reader& reader, const char* expected) {
    return {.expected = expected, .offset = reader.consumed(), .found = std::nullopt};
}

/// An unsigned field that must fit the Rust side's u32.
std::expected<std::uint32_t, DecodeError> decode_u32(Reader& reader, const char* expected) {
    const auto value = reader.uint();
    if (!value) {
        return std::unexpected(value.error());
    }
    if (*value > 0xffff'ffff) {
        return std::unexpected(mismatch(reader, expected));
    }
    return static_cast<std::uint32_t>(*value);
}

std::expected<FbDesc, DecodeError> decode_fb_desc(Reader& reader) {
    const auto fields = reader.array_header();
    if (!fields) {
        return std::unexpected(fields.error());
    }
    if (*fields != 4) {
        return std::unexpected(mismatch(reader, "the four FbDesc fields"));
    }
    const auto shm_name = reader.str();
    if (!shm_name) {
        return std::unexpected(shm_name.error());
    }
    const auto bytes = reader.uint();
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    const auto width = decode_u32(reader, "a width that fits u32");
    if (!width) {
        return std::unexpected(width.error());
    }
    const auto height = decode_u32(reader, "a height that fits u32");
    if (!height) {
        return std::unexpected(height.error());
    }
    return FbDesc{
        .shm_name = std::string(*shm_name),
        .bytes = *bytes,
        .width = *width,
        .height = *height,
    };
}

std::expected<Response, DecodeError> decode_welcome(Reader& reader) {
    const auto fields = reader.array_header();
    if (!fields) {
        return std::unexpected(fields.error());
    }
    if (*fields != 2) {
        return std::unexpected(mismatch(reader, "the two Welcome fields"));
    }
    const auto protocol = decode_u32(reader, "a protocol revision that fits u32");
    if (!protocol) {
        return std::unexpected(protocol.error());
    }
    auto fb = decode_fb_desc(reader);
    if (!fb) {
        return std::unexpected(fb.error());
    }
    return Welcome{.protocol = *protocol, .fb = std::move(*fb)};
}

std::expected<Response, DecodeError> decode_ack(Reader& reader) {
    const auto fields = reader.array_header();
    if (!fields) {
        return std::unexpected(fields.error());
    }
    if (*fields != 1) {
        return std::unexpected(mismatch(reader, "the one Ack field"));
    }
    const auto count = reader.array_header();
    if (!count) {
        return std::unexpected(count.error());
    }
    std::vector<std::string> rejected;
    for (std::size_t index = 0; index < *count; ++index) {
        const auto message = reader.str();
        if (!message) {
            return std::unexpected(message.error());
        }
        rejected.emplace_back(*message);
    }
    return Ack{.rejected = std::move(rejected)};
}

std::expected<Response, DecodeError> decode_resized(Reader& reader) {
    auto fb = decode_fb_desc(reader);
    if (!fb) {
        return std::unexpected(fb.error());
    }
    return Resized{.fb = std::move(*fb)};
}

} // namespace

void encode(Writer& writer, const Camera& value) {
    writer.array_header(6);
    encode(writer, value.position);
    encode(writer, value.look_at);
    encode(writer, value.up);
    encode(writer, value.vfov_degrees);
    encode(writer, value.focus_distance);
    encode(writer, value.aperture_radius);
}

void encode(Writer& writer, const FbDesc& value) {
    writer.array_header(4);
    encode(writer, value.shm_name);
    encode(writer, value.bytes);
    encode(writer, value.width);
    encode(writer, value.height);
}

void encode(Writer& writer, const Request& value) {
    std::visit(Overloaded{
                   [&](const Hello& hello) {
                       writer.map_header(1);
                       writer.str("Hello");
                       writer.array_header(2);
                       encode(writer, hello.protocol);
                       encode(writer, hello.token);
                   },
                   [&](const Replace& replace) {
                       writer.map_header(1);
                       writer.str("Replace");
                       encode(writer, replace.changes);
                   },
                   [&](const Apply& apply) {
                       writer.map_header(1);
                       writer.str("Apply");
                       encode(writer, apply.changes);
                   },
                   [&](const SetCamera& set_camera) {
                       writer.map_header(1);
                       writer.str("SetCamera");
                       encode(writer, set_camera.camera);
                   },
                   [&](const Resize& resize) {
                       writer.map_header(1);
                       writer.str("Resize");
                       writer.array_header(2);
                       encode(writer, resize.width);
                       encode(writer, resize.height);
                   },
                   [&](const Ping&) { writer.str("Ping"); },
               },
               value);
}

void encode(Writer& writer, const Response& value) {
    writer.map_header(1);
    std::visit(Overloaded{
                   [&](const Welcome& welcome) {
                       writer.str("Welcome");
                       writer.array_header(2);
                       encode(writer, welcome.protocol);
                       encode(writer, welcome.fb);
                   },
                   [&](const Ack& ack) {
                       writer.str("Ack");
                       writer.array_header(1);
                       encode(writer, ack.rejected);
                   },
                   [&](const Resized& resized) {
                       writer.str("Resized");
                       encode(writer, resized.fb);
                   },
               },
               value);
}

std::expected<Response, DecodeError> decode_response(std::span<const std::uint8_t> payload) {
    Reader reader(payload);
    // No Response variant is a bare unit, so every reply arrives as a
    // one-entry map from variant name to value.
    const auto entries = reader.map_header();
    if (!entries) {
        return std::unexpected(entries.error());
    }
    if (*entries != 1) {
        return std::unexpected(mismatch(reader, "a one-entry variant map"));
    }
    const auto name = reader.str();
    if (!name) {
        return std::unexpected(name.error());
    }
    std::expected<Response, DecodeError> response =
        std::unexpected(mismatch(reader, "a Response variant name"));
    if (*name == "Welcome") {
        response = decode_welcome(reader);
    } else if (*name == "Ack") {
        response = decode_ack(reader);
    } else if (*name == "Resized") {
        response = decode_resized(reader);
    }
    if (response && reader.remaining() != 0) {
        return std::unexpected(mismatch(reader, "no bytes after the message"));
    }
    return response;
}

} // namespace cenote::wire
