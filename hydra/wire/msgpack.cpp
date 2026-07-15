#include "msgpack.hpp"

#include <bit>

namespace cenote::wire {

namespace {

// MessagePack format bytes — the subset the wire uses.
constexpr std::uint8_t format_nil = 0xc0;
constexpr std::uint8_t format_false = 0xc2;
constexpr std::uint8_t format_true = 0xc3;
constexpr std::uint8_t format_f32 = 0xca;
constexpr std::uint8_t format_uint8 = 0xcc;
constexpr std::uint8_t format_uint16 = 0xcd;
constexpr std::uint8_t format_uint32 = 0xce;
constexpr std::uint8_t format_uint64 = 0xcf;
constexpr std::uint8_t format_int8 = 0xd0;
constexpr std::uint8_t format_int16 = 0xd1;
constexpr std::uint8_t format_int32 = 0xd2;
constexpr std::uint8_t format_int64 = 0xd3;
constexpr std::uint8_t format_str8 = 0xd9;
constexpr std::uint8_t format_str16 = 0xda;
constexpr std::uint8_t format_str32 = 0xdb;
constexpr std::uint8_t format_array16 = 0xdc;
constexpr std::uint8_t format_array32 = 0xdd;
constexpr std::uint8_t format_map16 = 0xde;
constexpr std::uint8_t format_map32 = 0xdf;

} // namespace

void Writer::nil() { buffer_.push_back(format_nil); }

void Writer::boolean(bool value) { buffer_.push_back(value ? format_true : format_false); }

void Writer::uint(std::uint64_t value) {
    if (value <= 0x7f) {
        buffer_.push_back(static_cast<std::uint8_t>(value));
    } else if (value <= 0xff) {
        big_endian(format_uint8, value, 1);
    } else if (value <= 0xffff) {
        big_endian(format_uint16, value, 2);
    } else if (value <= 0xffff'ffff) {
        big_endian(format_uint32, value, 4);
    } else {
        big_endian(format_uint64, value, 8);
    }
}

void Writer::sint(std::int64_t value) {
    // Two's complement survives the unsigned cast, so big_endian's low
    // bytes are exactly the encoding each signed format wants.
    if (value >= 0) {
        uint(static_cast<std::uint64_t>(value));
    } else if (value >= -32) {
        buffer_.push_back(static_cast<std::uint8_t>(value));
    } else if (value >= -128) {
        big_endian(format_int8, static_cast<std::uint64_t>(value), 1);
    } else if (value >= -32768) {
        big_endian(format_int16, static_cast<std::uint64_t>(value), 2);
    } else if (value >= -2147483648) {
        big_endian(format_int32, static_cast<std::uint64_t>(value), 4);
    } else {
        big_endian(format_int64, static_cast<std::uint64_t>(value), 8);
    }
}

void Writer::f32(float value) { big_endian(format_f32, std::bit_cast<std::uint32_t>(value), 4); }

void Writer::str(std::string_view value) {
    const std::size_t length = value.size();
    if (length <= 31) {
        buffer_.push_back(static_cast<std::uint8_t>(0xa0 | length));
    } else if (length <= 0xff) {
        big_endian(format_str8, length, 1);
    } else if (length <= 0xffff) {
        big_endian(format_str16, length, 2);
    } else {
        big_endian(format_str32, length, 4);
    }
    buffer_.insert(buffer_.end(), value.begin(), value.end());
}

void Writer::array_header(std::size_t count) {
    if (count <= 15) {
        buffer_.push_back(static_cast<std::uint8_t>(0x90 | count));
    } else if (count <= 0xffff) {
        big_endian(format_array16, count, 2);
    } else {
        big_endian(format_array32, count, 4);
    }
}

void Writer::map_header(std::size_t count) {
    if (count <= 15) {
        buffer_.push_back(static_cast<std::uint8_t>(0x80 | count));
    } else if (count <= 0xffff) {
        big_endian(format_map16, count, 2);
    } else {
        big_endian(format_map32, count, 4);
    }
}

void Writer::big_endian(std::uint8_t format, std::uint64_t value, int count) {
    buffer_.push_back(format);
    for (int shift = 8 * (count - 1); shift >= 0; shift -= 8) {
        buffer_.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

std::expected<std::uint64_t, DecodeError> Reader::uint() {
    constexpr auto expected = "an unsigned integer";
    const std::size_t start = cursor_;
    const auto format = next_byte(expected);
    if (!format) {
        return std::unexpected(format.error());
    }
    if (*format <= 0x7f) {
        return *format;
    }
    switch (*format) {
    case format_uint8:
        return big_endian(expected, 1);
    case format_uint16:
        return big_endian(expected, 2);
    case format_uint32:
        return big_endian(expected, 4);
    case format_uint64:
        return big_endian(expected, 8);
    default:
        return std::unexpected(
            DecodeError{.expected = expected, .offset = start, .found = *format});
    }
}

std::expected<std::string_view, DecodeError> Reader::str() {
    constexpr auto expected = "a string";
    const std::size_t start = cursor_;
    const auto format = next_byte(expected);
    if (!format) {
        return std::unexpected(format.error());
    }
    std::expected<std::uint64_t, DecodeError> length;
    if ((*format & 0xe0) == 0xa0) {
        length = *format & 0x1f;
    } else if (*format == format_str8) {
        length = big_endian(expected, 1);
    } else if (*format == format_str16) {
        length = big_endian(expected, 2);
    } else if (*format == format_str32) {
        length = big_endian(expected, 4);
    } else {
        return std::unexpected(
            DecodeError{.expected = expected, .offset = start, .found = *format});
    }
    if (!length) {
        return std::unexpected(length.error());
    }
    if (remaining() < *length) {
        return std::unexpected(
            DecodeError{.expected = expected, .offset = payload_.size(), .found = std::nullopt});
    }
    const auto* data = reinterpret_cast<const char*>(payload_.data() + cursor_);
    cursor_ += static_cast<std::size_t>(*length);
    return std::string_view(data, static_cast<std::size_t>(*length));
}

std::expected<std::size_t, DecodeError> Reader::array_header() {
    constexpr auto expected = "an array";
    const std::size_t start = cursor_;
    const auto format = next_byte(expected);
    if (!format) {
        return std::unexpected(format.error());
    }
    if ((*format & 0xf0) == 0x90) {
        return *format & 0x0f;
    }
    const auto wide = [&](std::size_t count) {
        return big_endian(expected, count).transform([](std::uint64_t value) {
            return static_cast<std::size_t>(value);
        });
    };
    if (*format == format_array16) {
        return wide(2);
    }
    if (*format == format_array32) {
        return wide(4);
    }
    return std::unexpected(DecodeError{.expected = expected, .offset = start, .found = *format});
}

std::expected<std::size_t, DecodeError> Reader::map_header() {
    constexpr auto expected = "a map";
    const std::size_t start = cursor_;
    const auto format = next_byte(expected);
    if (!format) {
        return std::unexpected(format.error());
    }
    if ((*format & 0xf0) == 0x80) {
        return *format & 0x0f;
    }
    const auto wide = [&](std::size_t count) {
        return big_endian(expected, count).transform([](std::uint64_t value) {
            return static_cast<std::size_t>(value);
        });
    };
    if (*format == format_map16) {
        return wide(2);
    }
    if (*format == format_map32) {
        return wide(4);
    }
    return std::unexpected(DecodeError{.expected = expected, .offset = start, .found = *format});
}

std::expected<std::uint8_t, DecodeError> Reader::next_byte(const char* expected) {
    if (cursor_ == payload_.size()) {
        return std::unexpected(
            DecodeError{.expected = expected, .offset = cursor_, .found = std::nullopt});
    }
    return payload_[cursor_++];
}

std::expected<std::uint64_t, DecodeError> Reader::big_endian(const char* expected,
                                                             std::size_t count) {
    if (remaining() < count) {
        return std::unexpected(
            DecodeError{.expected = expected, .offset = payload_.size(), .found = std::nullopt});
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < count; ++index) {
        value = (value << 8) | payload_[cursor_++];
    }
    return value;
}

std::array<std::uint8_t, 4> frame_prefix(std::uint32_t payload_bytes) {
    return {
        static_cast<std::uint8_t>(payload_bytes),
        static_cast<std::uint8_t>(payload_bytes >> 8),
        static_cast<std::uint8_t>(payload_bytes >> 16),
        static_cast<std::uint8_t>(payload_bytes >> 24),
    };
}

std::uint32_t framed_length(std::span<const std::uint8_t, 4> prefix) {
    return static_cast<std::uint32_t>(prefix[0]) | (static_cast<std::uint32_t>(prefix[1]) << 8) |
           (static_cast<std::uint32_t>(prefix[2]) << 16) |
           (static_cast<std::uint32_t>(prefix[3]) << 24);
}

} // namespace cenote::wire
