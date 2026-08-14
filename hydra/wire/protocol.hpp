// The control half of the wire mirror — cenote-wire's protocol.rs: the
// request/response envelope, its encoders, and a hand-written decoder for
// the three server replies. Strict request/response over loopback TCP;
// the u32-LE length prefix from msgpack.hpp frames each payload.
//
// This process is the client, so the decoder covers Response only. The
// Request encoders still earn their keep twice over: the conformance test
// proves both directions of the golden corpus, and the server never has
// to trust that a shape it received was ever produced.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "msgpack.hpp"
#include "scene.hpp"

namespace cenote::wire {

/// The protocol revision carried in Hello/Welcome — mirror of `PROTOCOL`.
/// The Rust side bumps it on any change to any encoded shape; the two
/// sides move in lockstep with the golden corpus.
inline constexpr std::uint32_t PROTOCOL = 11;

/// The most bytes one frame may claim — mirror of `MAX_MESSAGE_BYTES`,
/// the transport's plausibility guard so a garbage or hostile length
/// prefix cannot demand an unbounded allocation.
inline constexpr std::uint32_t MAX_MESSAGE_BYTES = 1U << 30;

/// The active camera as `SetCamera` carries it — mirror of `Camera`.
/// `focus_distance` absent focuses at `look_at`; `aperture_radius` 0 is
/// a pinhole.
struct Camera {
    std::array<float, 3> position;
    std::array<float, 3> look_at;
    std::array<float, 3> up;
    float vfov_degrees;
    std::optional<float> focus_distance;
    float aperture_radius;
};

/// Where the pixels live — mirror of `FbDesc`: the `shm_open` name
/// (`/cenote-<pid>-<generation>`) and enough to map the segment.
struct FbDesc {
    std::string shm_name;
    std::uint64_t bytes;
    std::uint32_t width;
    std::uint32_t height;
};

/// `Request`'s Hello alternative: the handshake, first on the socket —
/// this client's `PROTOCOL` and the spawn-time token proving it is the
/// process that spawned the server.
struct Hello {
    std::uint32_t protocol;
    std::string token;
};

/// `Request`'s Replace alternative: the whole scene from empty — genesis
/// and stage-reload. (The Rust newtype's unnamed field gets a name.)
struct Replace {
    ChangeSet changes;
};

/// `Request`'s Apply alternative: an overlay edit onto the current scene
/// — the steady-state delta.
struct Apply {
    ChangeSet changes;
};

/// `Request`'s SetCamera alternative: the inputs-lane fast path for the
/// active camera — a camera inside a ChangeSet would be silently dead.
struct SetCamera {
    Camera camera;
};

/// `Request`'s Resize alternative: a new framebuffer size. The reply's
/// descriptor is immediately mappable.
struct Resize {
    std::uint32_t width;
    std::uint32_t height;
};

/// `Request`'s Ping alternative: a liveness probe, and how an idle client
/// collects rejection messages once the shm header's counter moves.
struct Ping {};

/// A client message — mirror of `Request`. Every variant is answered by
/// exactly one Response, in order.
using Request = std::variant<Hello, Replace, Apply, SetCamera, Resize, Ping>;

/// `Response`'s Welcome alternative: answers Hello — the server's
/// `PROTOCOL` and the initial framebuffer segment.
struct Welcome {
    std::uint32_t protocol;
    FbDesc fb;
};

/// `Response`'s Ack alternative: answers Replace, Apply, SetCamera, and
/// Ping. A receipt, not a validation — edits apply at the next wave
/// boundary, and `rejected` carries every rejection message accumulated
/// since the last response. `epoch` is the session epoch after this
/// request: the first shm frame whose header stamp reaches it
/// incorporates everything sent so far, applied or rejected.
struct Ack {
    std::vector<std::string> rejected;
    std::uint64_t epoch;
};

/// `Response`'s Resized alternative: answers Resize — the new framebuffer
/// segment, ready to map, and Ack's epoch on the one reply that is not
/// an Ack.
struct Resized {
    FbDesc fb;
    std::uint64_t epoch;
};

/// The server's reply — mirror of `Response`.
using Response = std::variant<Welcome, Ack, Resized>;

// The envelope encoders — same contract as scene.hpp's set.

void encode(Writer& writer, const Camera& value);
void encode(Writer& writer, const FbDesc& value);
void encode(Writer& writer, const Request& value);
void encode(Writer& writer, const Response& value);

/// Decodes one reply payload (no length prefix). Strict on every axis a
/// drifted server could move: the variant name must be known, field
/// counts must match, integers must fit their Rust widths, and the
/// message's last byte must be the payload's last byte.
[[nodiscard]] std::expected<Response, DecodeError>
decode_response(std::span<const std::uint8_t> payload);

} // namespace cenote::wire
