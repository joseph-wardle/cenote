// The transport client: the delegate's end of the wire. Constructing one
// spawns cenote-server, reads the one stdout line naming its port,
// connects over loopback TCP, and proves this process did the spawning
// with the token the Hello carries. After that it speaks strict
// request/response — one frame out, exactly one frame back — and keeps
// the Welcome's framebuffer segment mapped: resize() swaps segment and
// mapping together, so view() always matches the descriptor the server
// last named.
//
// Every failure, at birth or later, lands in degraded mode: the client
// stays alive, renders nothing, and says why on the warning stream. The
// host outlives any render server it spawns — isolation applies to
// birth, not just death.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sys/types.h>

#include "transport/view.hpp"
#include "wire/protocol.hpp"

namespace cenote::transport {

/// One spawned server and the socket to it. Not thread-safe: one caller
/// at a time, which is the delegate's own discipline.
class Client {
public:
    /// Spawns the server and completes the Hello/Welcome handshake. Any
    /// failure — no binary, no port line, a refused handshake — posts a
    /// diagnostic and leaves the client degraded, never the host dead.
    Client();

    /// Socket EOF is the shutdown message (the wire has no other); a
    /// grace period lets the server exit 0 and unlink its shm segments,
    /// then SIGKILL ends whatever remains.
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    /// False is degraded mode — birth failed, or a later transport
    /// failure tore the connection down. Degradation is one-way.
    [[nodiscard]] bool alive() const { return socket_ != -1; }

    /// The framebuffer segment the server last named — the Welcome's at
    /// birth, the newest Resized's after. Meaningful only while alive().
    [[nodiscard]] const wire::FbDesc& fb() const { return fb_; }

    /// The mapped framebuffer, or null when degraded. Replaced by
    /// resize(); nothing taken from it survives one.
    [[nodiscard]] const View* view() const { return view_.get(); }

    /// Whether the server's accumulation has settled at a picture that
    /// is also *current* (D-113): the shm header's converged flag counts
    /// only once the front frame's epoch has reached the one the last
    /// Ack or Resized carried — a settled stale picture is not
    /// converged. True when degraded: the picture will never improve,
    /// and a host looping render-until-converged (usdrecord) must not
    /// spin forever against a dead server.
    [[nodiscard]] bool converged() const {
        return view_ == nullptr || (view_->epoch() >= last_acked_epoch_ && view_->converged());
    }

    /// Brings the framebuffer to width×height: a no-op when it already
    /// matches, otherwise one Resize round-trip and a remap of the
    /// reply's segment. False when degraded (silently — the diagnostic
    /// already posted), when a dimension is zero, or when the exchange
    /// or the remap fails, which degrades.
    bool resize(std::uint32_t width, std::uint32_t height);

    /// The whole scene from empty, atomically — genesis and stage
    /// reload. True once the server acknowledged it; any rejection
    /// messages riding the Ack surface as warnings. False when degraded
    /// (silently — the diagnostic already posted) or on transport
    /// failure, which degrades.
    bool replace(const wire::ChangeSet& changes);

    /// An overlay edit onto the current scene — the steady-state delta.
    /// Same contract as replace().
    bool apply(const wire::ChangeSet& changes);

    /// The active camera down the SetCamera lane. Same contract as
    /// replace().
    bool set_camera(const wire::Camera& camera);

    /// The between-requests health probe, for once per host frame.
    /// Strict request/response (D-100) means the socket is silent
    /// between calls, so anything readable outside one — bytes or a
    /// hangup — is server death or a protocol violation: degrade, with
    /// the one warning that names the recovery gesture (D-099).
    void check_liveness();

    /// Collects rejection messages an otherwise-idle client would never
    /// see: when the header's rejected-edit counter moves off what was
    /// last seen, one Ping fetches the strings riding its Ack (D-100).
    void collect_rejections();

    /// One request out, one response back. Any transport failure — a
    /// short write, a dead socket, a frame that will not decode —
    /// degrades the client and returns nullopt.
    [[nodiscard]] std::optional<wire::Response> call(const wire::Request& request);

private:
    // The birth stages, in calling order; each posts its own diagnostic
    // and returns false, and the constructor degrades on the first lie.
    bool spawn(const std::string& token);
    bool read_port_line();
    bool connect_socket();
    bool handshake(const std::string& token);
    bool map_framebuffer();

    /// The shared shape of every Ack-answered request: send, demand an
    /// Ack (naming the request `what` if the reply is anything else),
    /// surface the rejection messages the Ack carries.
    bool acked(const wire::Request& request, const char* what);

    /// One framed request onto the socket, warning as `what` on failure.
    bool send_frame(const std::vector<std::uint8_t>& payload, const char* what);

    /// One framed reply off the socket, warning as `what` on failure.
    std::optional<std::vector<std::uint8_t>> read_frame(const char* what);

    /// Enter degraded mode: close everything and reap the child, so a
    /// failed transport never leaves an orphan behind. Idempotent — also
    /// the destructor's whole body.
    void shut_down();

    pid_t pid_ = -1;
    int stdout_pipe_ = -1;
    int socket_ = -1;
    std::uint16_t port_ = 0;
    wire::FbDesc fb_{};
    std::unique_ptr<View> view_;
    /// The epoch the last Ack or Resized carried — the bar the front
    /// frame must reach before its converged flag is believed.
    std::uint64_t last_acked_epoch_ = 0;
    /// Where the rejected-edit counter stood when the strings were last
    /// collected.
    std::uint32_t seen_rejections_ = 0;
};

} // namespace cenote::transport
