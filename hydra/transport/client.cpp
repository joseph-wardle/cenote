// One of transport/'s two deliberately-POSIX files (view.cpp, the shm
// reader, is the other): pipes, posix_spawn, poll, waitpid, and BSD
// sockets live here and nowhere else in the tree. posix_spawn rather
// than fork() because the host is a large threaded process — forking
// one is undefined-behaviour roulette.
#include "client.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pxr/base/tf/diagnostic.h"

extern char** environ;

PXR_NAMESPACE_USING_DIRECTIVE

namespace cenote::transport {

namespace {

/// How long birth may wait for the port line. Really a GPU-init budget:
/// the server prints the port only after its GPU context and Session
/// exist, precisely so that slowness surfaces here, legibly, instead of
/// as a dead socket later.
constexpr std::chrono::seconds PORT_LINE_DEADLINE{30};

/// How long one reply may take before the server counts as gone. Every
/// request is answered at socket speed — edits land at wave boundaries,
/// so no reply ever waits on the renderer.
constexpr std::chrono::seconds RESPONSE_TIMEOUT{30};

/// How long teardown waits after the EOF for a clean exit — which is
/// what unlinks the shm segments — before concluding the server is
/// stuck and sending SIGKILL.
constexpr std::chrono::seconds SHUTDOWN_GRACE{5};

/// The one stdout line the server prints, up to the port digits. The
/// token suffix the server can append never appears here: this client
/// always supplies CENOTE_SERVER_TOKEN, so the server never generates.
constexpr std::string_view PORT_PREFIX = "cenote-server port=";

/// 128 bits from std::random_device, hex — the spawn-time secret the
/// Hello must echo back. Loopback TCP is reachable by any local
/// process; this is what closes that gap.
std::string generate_token() {
    std::random_device device;
    std::string hex;
    hex.reserve(32);
    for (int word = 0; word < 4; ++word) {
        char chunk[9];
        std::snprintf(chunk, sizeof chunk, "%08x", static_cast<unsigned int>(device()));
        hex += chunk;
    }
    return hex;
}

/// The directory holding the shared object this code is loaded from —
/// where a packaged distribution drops cenote-server beside the plugin.
/// Empty when the loader cannot say.
std::string plugin_directory() {
    Dl_info info = {};
    if (::dladdr(reinterpret_cast<void*>(&plugin_directory), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    const std::string_view path = info.dli_fname;
    const std::size_t slash = path.rfind('/');
    if (slash == std::string_view::npos) {
        return {};
    }
    return std::string(path.substr(0, slash));
}

/// The binary lookup: $CENOTE_SERVER → beside the plugin .so → $PATH.
/// Returns the path to spawn, or empty after a runtime error naming all
/// three lanes.
std::string locate_server() {
    // An explicit override that does not work is an error to surface,
    // never a thing to silently fall past — the user would be rendering
    // with a different binary than the one they named.
    const char* overridden = std::getenv("CENOTE_SERVER");
    if (overridden != nullptr && *overridden != '\0') {
        if (::access(overridden, X_OK) == 0) {
            return overridden;
        }
        TF_RUNTIME_ERROR("no usable cenote-server: $CENOTE_SERVER is set to \"%s\" but nothing "
                         "executable is there (%s); beside-the-plugin and $PATH defer to the "
                         "explicit setting",
                         overridden, std::strerror(errno));
        return {};
    }

    const std::string beside = plugin_directory();
    if (!beside.empty()) {
        std::string candidate = beside + "/cenote-server";
        if (::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
    }

    const char* path = std::getenv("PATH");
    if (path != nullptr) {
        std::string_view remaining = path;
        while (true) {
            const std::size_t colon = remaining.find(':');
            const std::string_view directory = remaining.substr(0, colon);
            // POSIX reads an empty $PATH entry as the current directory.
            std::string candidate =
                std::string(directory.empty() ? "." : directory) + "/cenote-server";
            if (::access(candidate.c_str(), X_OK) == 0) {
                return candidate;
            }
            if (colon == std::string_view::npos) {
                break;
            }
            remaining.remove_prefix(colon + 1);
        }
    }

    const std::string beside_lane =
        beside.empty() ? std::string("the plugin's own location could not be determined")
                       : "none beside the plugin (" + beside + "/cenote-server)";
    TF_RUNTIME_ERROR("no usable cenote-server: $CENOTE_SERVER is unset, %s, and no executable "
                     "cenote-server appears on $PATH",
                     beside_lane.c_str());
    return {};
}

/// send(2) until every byte is out. MSG_NOSIGNAL because a dead peer
/// must come back as an error to degrade on, never a SIGPIPE delivered
/// to the host.
bool send_all(int fd, const std::uint8_t* bytes, std::size_t count) {
    while (count > 0) {
        const ssize_t sent = ::send(fd, bytes, count, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        bytes += sent;
        count -= static_cast<std::size_t>(sent);
    }
    return true;
}

/// How a full-buffer read ended: everything arrived, the peer hung up,
/// or recv(2) failed (a receive-timeout lands here as EAGAIN).
enum class Recv { done, eof, failed };

Recv recv_all(int fd, std::uint8_t* bytes, std::size_t count) {
    while (count > 0) {
        const ssize_t got = ::recv(fd, bytes, count, 0);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Recv::failed;
        }
        if (got == 0) {
            return Recv::eof;
        }
        bytes += got;
        count -= static_cast<std::size_t>(got);
    }
    return Recv::done;
}

/// A child's wait status as prose, for the abnormal-exit warning.
std::string describe_status(int status) {
    if (WIFEXITED(status)) {
        return "exit status " + std::to_string(WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        return "signal " + std::to_string(WTERMSIG(status));
    }
    return "wait status " + std::to_string(status);
}

} // namespace

Client::Client() {
    const std::string token = generate_token();
    if (!(spawn(token) && read_port_line() && connect_socket() && handshake(token) &&
          map_framebuffer())) {
        shut_down();
    }
}

Client::~Client() { shut_down(); }

bool Client::resize(std::uint32_t width, std::uint32_t height) {
    if (socket_ == -1 || width == 0 || height == 0) {
        return false;
    }
    if (fb_.width == width && fb_.height == height) {
        return true;
    }
    const auto response = call(wire::Request{wire::Resize{.width = width, .height = height}});
    if (!response) {
        return false; // call() degraded and said why.
    }
    const auto* resized = std::get_if<wire::Resized>(&*response);
    if (resized == nullptr) {
        TF_WARN("the reply to Resize was not Resized");
        shut_down();
        return false;
    }
    fb_ = resized->fb;
    last_acked_epoch_ = resized->epoch;
    // The old mapping goes first; the server keeps the segment it
    // replaced alive until our *next* request, so the new name is still
    // linked when map_framebuffer opens it.
    view_.reset();
    if (!map_framebuffer()) {
        shut_down();
        return false;
    }
    return true;
}

bool Client::replace(const wire::ChangeSet& changes) {
    return acked(wire::Request{wire::Replace{.changes = changes}}, "Replace");
}

bool Client::apply(const wire::ChangeSet& changes) {
    return acked(wire::Request{wire::Apply{.changes = changes}}, "Apply");
}

bool Client::set_camera(const wire::Camera& camera) {
    return acked(wire::Request{wire::SetCamera{.camera = camera}}, "SetCamera");
}

void Client::check_liveness() {
    if (socket_ == -1) {
        return;
    }
    struct pollfd probe = {.fd = socket_, .events = POLLIN, .revents = 0};
    if (::poll(&probe, 1, 0) <= 0) {
        return; // A silent socket is the healthy steady state.
    }
    // Strict request/response: between calls the server never
    // speaks, so a readable or hung-up socket is death or a protocol
    // violation — degraded either way. Recovery is a fresh spawn
    //: destroying and recreating the delegate replays the whole
    // stage.
    TF_WARN("cenote-server hung up outside a request; the picture is frozen — toggle the "
            "renderer or reload the stage to spawn a fresh server");
    shut_down();
}

void Client::collect_rejections() {
    if (view_ == nullptr) {
        return;
    }
    // Inequality, not order: the counter restarts at zero with each
    // resize's fresh segment.
    const std::uint32_t rejected = view_->rejected_edits();
    if (rejected == seen_rejections_) {
        return;
    }
    seen_rejections_ = rejected;
    // acked() warns the strings riding the Ack — and refreshes the
    // acked epoch for free.
    acked(wire::Request{wire::Ping{}}, "Ping");
}

bool Client::acked(const wire::Request& request, const char* what) {
    const auto response = call(request);
    if (!response) {
        return false;
    }
    const auto* ack = std::get_if<wire::Ack>(&*response);
    if (ack == nullptr) {
        TF_WARN("the reply to %s was not Ack", what);
        shut_down();
        return false;
    }
    // The reply's epoch is the new bar for converged(): the front frame
    // must reach it before a settled flag is believed again.
    last_acked_epoch_ = ack->epoch;
    // A receipt, not a validation: the rejections riding it are from
    // earlier edits, surfaced here so they are never silently dropped.
    for (const std::string& message : ack->rejected) {
        TF_WARN("cenote-server rejected an edit: %s", message.c_str());
    }
    return true;
}

std::optional<wire::Response> Client::call(const wire::Request& request) {
    if (socket_ == -1) {
        // Degraded: the diagnostic was posted when it happened.
        return std::nullopt;
    }
    wire::Writer writer;
    wire::encode(writer, request);
    if (!send_frame(writer.take(), "sending a request")) {
        shut_down();
        return std::nullopt;
    }
    const auto reply = read_frame("reading a response");
    if (!reply) {
        shut_down();
        return std::nullopt;
    }
    auto response = wire::decode_response(*reply);
    if (!response) {
        TF_WARN("a response that would not decode: expected %s at offset %zu",
                response.error().expected, response.error().offset);
        shut_down();
        return std::nullopt;
    }
    return std::move(*response);
}

bool Client::spawn(const std::string& token) {
    std::string binary = locate_server();
    if (binary.empty()) {
        return false;
    }

    int port_pipe[2];
    if (::pipe2(port_pipe, O_CLOEXEC) != 0) {
        TF_WARN("creating the port pipe: %s", std::strerror(errno));
        return false;
    }

    // The child environment is the host's with the token put in place of
    // any stale one. The token travels in the environment, never argv —
    // /proc/*/cmdline is world-readable.
    std::vector<std::string> variables;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        if (!std::string_view(*entry).starts_with("CENOTE_SERVER_TOKEN=")) {
            variables.emplace_back(*entry);
        }
    }
    variables.push_back("CENOTE_SERVER_TOKEN=" + token);
    std::vector<char*> envp;
    envp.reserve(variables.size() + 1);
    for (std::string& variable : variables) {
        envp.push_back(variable.data());
    }
    envp.push_back(nullptr);
    std::array<char*, 2> argv = {binary.data(), nullptr};

    // stdin from /dev/null (the server never reads it), stdout to the
    // pipe the port line arrives on, stderr inherited — server logs
    // interleave with the host's, by design.
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, port_pipe[1], STDOUT_FILENO);

    // The host is a large threaded process that may mask or ignore
    // signals on the spawning thread; the server starts from POSIX
    // defaults instead of inheriting that.
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    sigset_t defaulted;
    sigfillset(&defaulted);
    posix_spawnattr_setsigdefault(&attributes, &defaulted);
    sigset_t unmasked;
    sigemptyset(&unmasked);
    posix_spawnattr_setsigmask(&attributes, &unmasked);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);

    const int spawned =
        ::posix_spawn(&pid_, binary.c_str(), &actions, &attributes, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    ::close(port_pipe[1]);
    if (spawned != 0) {
        ::close(port_pipe[0]);
        pid_ = -1;
        TF_WARN("spawning %s: %s", binary.c_str(), std::strerror(spawned));
        return false;
    }
    stdout_pipe_ = port_pipe[0];
    return true;
}

bool Client::read_port_line() {
    const auto deadline = std::chrono::steady_clock::now() + PORT_LINE_DEADLINE;
    std::string line;
    while (line.find('\n') == std::string::npos) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining <= std::chrono::milliseconds::zero()) {
            TF_WARN("cenote-server did not print its port line within %lld s",
                    static_cast<long long>(PORT_LINE_DEADLINE.count()));
            return false;
        }
        struct pollfd waiting = {.fd = stdout_pipe_, .events = POLLIN, .revents = 0};
        const int ready = ::poll(&waiting, 1, static_cast<int>(remaining.count()));
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            TF_WARN("polling for the port line: %s", std::strerror(errno));
            return false;
        }
        if (ready == 0) {
            continue; // The loop head turns this into the deadline warning.
        }
        char chunk[128];
        const ssize_t got = ::read(stdout_pipe_, chunk, sizeof chunk);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            TF_WARN("reading the port line: %s", std::strerror(errno));
            return false;
        }
        if (got == 0) {
            // Startup death: its stderr (inherited) already says why, and
            // shut_down's reap will name the exit status.
            TF_WARN("cenote-server closed its stdout before printing a port line");
            return false;
        }
        line.append(chunk, static_cast<std::size_t>(got));
        if (line.size() > 256) {
            TF_WARN("cenote-server printed %zu bytes of stdout with no port line in them",
                    line.size());
            return false;
        }
    }
    ::close(stdout_pipe_);
    stdout_pipe_ = -1;

    const std::string_view content = std::string_view(line).substr(0, line.find('\n'));
    unsigned int port = 0;
    bool recognized = content.starts_with(PORT_PREFIX);
    if (recognized) {
        const std::string_view digits = content.substr(PORT_PREFIX.size());
        const char* end = digits.data() + digits.size();
        // from_chars reads [first, last): the size travels as `end`.
        // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
        const auto parsed = std::from_chars(digits.data(), end, port);
        recognized = parsed.ec == std::errc() && parsed.ptr == end && port > 0 && port <= 65535;
    }
    if (!recognized) {
        TF_WARN("unrecognized stdout line from cenote-server: \"%s\"",
                std::string(content).c_str());
        return false;
    }
    port_ = static_cast<std::uint16_t>(port);
    return true;
}

bool Client::connect_socket() {
    socket_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket_ == -1) {
        TF_WARN("creating the socket: %s", std::strerror(errno));
        return false;
    }
    // Small frames answered promptly is the whole traffic pattern.
    const int nodelay = 1;
    ::setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof nodelay);
    // A reply slower than the response window means the server is gone
    // or wedged; recv fails and the client degrades 
    // instead of freezing the host.
    const struct timeval window = {.tv_sec = static_cast<time_t>(RESPONSE_TIMEOUT.count()),
                                   .tv_usec = 0};
    ::setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &window, sizeof window);

    struct sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port_);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    while (::connect(socket_, reinterpret_cast<const struct sockaddr*>(&address), sizeof address) !=
           0) {
        // An interrupted connect keeps connecting; retrying reports
        // EISCONN once it lands.
        if (errno == EINTR || errno == EALREADY) {
            continue;
        }
        if (errno == EISCONN) {
            break;
        }
        TF_WARN("connecting to 127.0.0.1:%u: %s", port_, std::strerror(errno));
        return false;
    }
    return true;
}

bool Client::handshake(const std::string& token) {
    wire::Writer writer;
    wire::encode(writer, wire::Request{wire::Hello{.protocol = wire::PROTOCOL, .token = token}});
    if (!send_frame(writer.take(), "sending Hello")) {
        return false;
    }
    const auto reply = read_frame("reading Welcome");
    if (!reply) {
        return false;
    }
    const auto response = wire::decode_response(*reply);
    if (!response) {
        TF_WARN("a Welcome that would not decode: expected %s at offset %zu",
                response.error().expected, response.error().offset);
        return false;
    }
    const auto* welcome = std::get_if<wire::Welcome>(&*response);
    if (welcome == nullptr) {
        TF_WARN("the handshake reply was not Welcome");
        return false;
    }
    if (welcome->protocol != wire::PROTOCOL) {
        TF_WARN("protocol mismatch: cenote-server speaks revision %u, this client speaks %u",
                welcome->protocol, wire::PROTOCOL);
        return false;
    }
    fb_ = welcome->fb;
    return true;
}

bool Client::map_framebuffer() {
    view_ = View::open(fb_);
    if (view_ == nullptr) {
        // View::open already named the specific disagreement.
        TF_WARN("the framebuffer segment cenote-server offered is unusable");
        return false;
    }
    return true;
}

bool Client::send_frame(const std::vector<std::uint8_t>& payload, const char* what) {
    if (payload.size() > wire::MAX_MESSAGE_BYTES) {
        TF_WARN("%s: %zu bytes exceeds the frame cap", what, payload.size());
        return false;
    }
    const auto prefix = wire::frame_prefix(static_cast<std::uint32_t>(payload.size()));
    if (!send_all(socket_, prefix.data(), prefix.size()) ||
        !send_all(socket_, payload.data(), payload.size())) {
        TF_WARN("%s: %s", what, std::strerror(errno));
        return false;
    }
    return true;
}

std::optional<std::vector<std::uint8_t>> Client::read_frame(const char* what) {
    const auto receive = [&](std::uint8_t* bytes, std::size_t count) {
        switch (recv_all(socket_, bytes, count)) {
        case Recv::done:
            return true;
        case Recv::eof:
            TF_WARN("%s: the server hung up", what);
            return false;
        case Recv::failed:
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                TF_WARN("%s: no reply within %lld s", what,
                        static_cast<long long>(RESPONSE_TIMEOUT.count()));
            } else {
                TF_WARN("%s: %s", what, std::strerror(errno));
            }
            return false;
        }
        return false;
    };
    std::array<std::uint8_t, 4> prefix;
    if (!receive(prefix.data(), prefix.size())) {
        return std::nullopt;
    }
    const std::uint32_t length = wire::framed_length(prefix);
    if (length > wire::MAX_MESSAGE_BYTES) {
        TF_WARN("%s: a %u-byte frame exceeds the cap", what, length);
        return std::nullopt;
    }
    std::vector<std::uint8_t> payload(length);
    if (!receive(payload.data(), payload.size())) {
        return std::nullopt;
    }
    return payload;
}

void Client::shut_down() {
    // Unmapping first means a degraded client's render buffers freeze on
    // whatever frame they last copied, rather than reading a segment no
    // writer will ever advance again.
    view_.reset();
    if (stdout_pipe_ != -1) {
        ::close(stdout_pipe_);
        stdout_pipe_ = -1;
    }
    if (socket_ != -1) {
        // The EOF — for a healthy server, the whole shutdown message.
        ::close(socket_);
        socket_ = -1;
    }
    if (pid_ == -1) {
        return;
    }
    // The grace period is for the clean exit that unlinks the shm
    // segments; a server that outlives it is not shutting down.
    const auto deadline = std::chrono::steady_clock::now() + SHUTDOWN_GRACE;
    int status = 0;
    pid_t reaped = ::waitpid(pid_, &status, WNOHANG);
    while (reaped == 0 && std::chrono::steady_clock::now() < deadline) {
        ::poll(nullptr, 0, 50);
        reaped = ::waitpid(pid_, &status, WNOHANG);
    }
    if (reaped == 0) {
        TF_WARN("cenote-server (pid %d) survived the shutdown grace period; sending SIGKILL",
                static_cast<int>(pid_));
        ::kill(pid_, SIGKILL);
        do {
            reaped = ::waitpid(pid_, &status, 0);
        } while (reaped == -1 && errno == EINTR);
    } else if (reaped == pid_ && WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        // Nonzero is the server's fault lane; the message is on stderr.
        TF_WARN("cenote-server exited abnormally: %s", describe_status(status).c_str());
    } else if (reaped == pid_ && WIFSIGNALED(status)) {
        TF_WARN("cenote-server died on %s", describe_status(status).c_str());
    }
    pid_ = -1;
}

} // namespace cenote::transport
