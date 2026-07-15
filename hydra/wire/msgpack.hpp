// The MessagePack layer of the C++ wire mirror: a Writer covering exactly
// the encodings the wire uses, a Reader sufficient to decode the response
// envelope, and the u32-LE frame prefix. Hand-rolled because every C++
// dependency in this tree is a host-ABI liability (docs/m4-plan.md §2).
//
// The byte shapes are not this file's to choose: `rmp-serde` defaults on
// the Rust side are the authority, pinned by the golden corpus in
// crates/cenote-wire/tests/golden/. That means the smallest encoding that
// fits, always — positive fixint before uint8 before uint16…, fixstr
// before str8, fixarray before array16 — f32 never widened, `None` as nil,
// unit enum variants as a bare string of the variant name, and every other
// variant as a one-entry map from variant name to value.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace cenote::wire {

/// Appends MessagePack values to a growing payload buffer, making the same
/// format choices as `rmp-serde`, so the per-type encode functions built
/// on top reproduce the Rust bytes exactly.
class Writer {
public:
    /// nil — the wire's absent `Option` field.
    void nil();

    /// true / false.
    void boolean(bool value);

    /// Unsigned integer: the smallest of positive fixint, uint8, uint16,
    /// uint32, uint64.
    void uint(std::uint64_t value);

    /// Signed integer: non-negative values take the unsigned formats
    /// (exactly as `rmp` does), negative the smallest of negative fixint,
    /// int8, int16, int32, int64.
    void sint(std::int64_t value);

    /// IEEE-754 single, always the five-byte f32 format — a Rust `f32`
    /// never widens.
    void f32(float value);

    /// UTF-8 string: the smallest of fixstr, str8, str16, str32. The
    /// caller guarantees UTF-8 (Rust strings always are) — this layer
    /// counts bytes, not code points.
    void str(std::string_view value);

    /// Array header only; the caller writes `count` elements after it.
    void array_header(std::size_t count);

    /// Map header only; the caller writes `count` key/value pairs after it.
    void map_header(std::size_t count);

    /// The payload so far.
    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const { return buffer_; }

    /// Moves the payload out, leaving the writer empty for reuse.
    [[nodiscard]] std::vector<std::uint8_t> take() { return std::exchange(buffer_, {}); }

private:
    /// A format byte, then the low `count` bytes of `value`, big-endian —
    /// MessagePack numbers are network order.
    void big_endian(std::uint8_t format, std::uint64_t value, int count);

    std::vector<std::uint8_t> buffer_;
};

/// Why a decode failed: what the caller asked for, where, and the format
/// byte actually found there (absent when the payload ended early).
struct DecodeError {
    /// The kind of value the caller wanted, e.g. "a string".
    const char* expected;
    /// Byte offset of the failure within the payload.
    std::size_t offset;
    /// The format byte found; `nullopt` means the payload was truncated —
    /// or, from the envelope layer above, that the bytes decoded fine but
    /// not to the shape the message requires (a wrong field count, an
    /// unknown variant name, an out-of-range value).
    std::optional<std::uint8_t> found;
};

/// Decodes the handful of formats the response envelope contains: unsigned
/// integers, strings, and array/map headers. Strict — each method accepts
/// exactly the formats `rmp-serde` emits for the corresponding Rust type,
/// so a drifted server fails loudly instead of close-enough.
class Reader {
public:
    /// Borrows the payload; string views returned by [`str`] point into it.
    explicit Reader(std::span<const std::uint8_t> payload) : payload_(payload) {}

    /// An unsigned integer: positive fixint or uint8/16/32/64.
    [[nodiscard]] std::expected<std::uint64_t, DecodeError> uint();

    /// A string: fixstr or str8/16/32. The view borrows the payload.
    [[nodiscard]] std::expected<std::string_view, DecodeError> str();

    /// An array header; the caller reads that many elements after it.
    [[nodiscard]] std::expected<std::size_t, DecodeError> array_header();

    /// A map header; the caller reads that many key/value pairs after it.
    [[nodiscard]] std::expected<std::size_t, DecodeError> map_header();

    /// Bytes not yet consumed — zero after a complete decode.
    [[nodiscard]] std::size_t remaining() const { return payload_.size() - cursor_; }

    /// Bytes consumed so far — where the next read begins, and the offset
    /// the envelope layer's semantic errors report.
    [[nodiscard]] std::size_t consumed() const { return cursor_; }

private:
    /// Consumes the next byte, or reports truncation.
    [[nodiscard]] std::expected<std::uint8_t, DecodeError> next_byte(const char* expected);

    /// Consumes `count` bytes as one big-endian value.
    [[nodiscard]] std::expected<std::uint64_t, DecodeError> big_endian(const char* expected,
                                                                       std::size_t count);

    std::span<const std::uint8_t> payload_;
    std::size_t cursor_ = 0;
};

/// The framed protocol's length prefix (`protocol.rs`): four bytes, little
/// endian, giving the payload byte count that follows.
[[nodiscard]] std::array<std::uint8_t, 4> frame_prefix(std::uint32_t payload_bytes);

/// Reads the prefix back. Checking the result against the protocol's
/// frame cap is the transport's job — it owns that policy.
[[nodiscard]] std::uint32_t framed_length(std::span<const std::uint8_t, 4> prefix);

} // namespace cenote::wire
