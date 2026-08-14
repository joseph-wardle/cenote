//! Spawn the real `cenote-server`
//! binary, drive it over TCP exactly as the C++ delegate will, and read a
//! correct frame out of shared memory — the whole transport proven with
//! no Hydra in sight. The saturated primary is the anti-regression tooth:
//! an authored `Rec.709` red must come back as `Rec.709` red, so dropping
//! the server-side `ACEScg` → `Rec.709` conversion fails the color
//! assertions (an unconverted red leaks ~11% into green).
//!
//! Needs a GPU (the server refuses to spawn without one); skips cleanly
//! — with a note on stderr — where there isn't one. Run serially, like
//! every GPU test: `--test-threads=1`.

use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use cenote_server::shm::{Snapshot, View};
use cenote_wire::protocol::{self, Camera, PROTOCOL, Request, Response};
use cenote_wire::scene as wire;

/// The spawn-time secret handed to the server; also what the wrong-token
/// test must *not* present.
const TOKEN: &str = "spine-test-token";

/// A low accumulation cap so the converged flag flips inside the test.
const CAP: u32 = 32;

const DEADLINE: Duration = Duration::from_mins(1);

/// The spawned server, killed on drop so a failing assertion never leaks
/// a GPU process (or its shm segments' *writer* — the names unlink with
/// it on the clean path only; a killed server is the leak the pid-scoped
/// naming tolerates).
struct Server {
    child: Child,
    port: u16,
}

impl Server {
    /// Spawn the binary under test with the test's token and cap, and
    /// parse the one stdout line. `None` when the server exited instead —
    /// no GPU here, the skip case (any real server-side bug reproduces
    /// under the GPU gate, so the skip cannot hide one).
    fn spawn() -> Option<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cenote-server"))
            .env("CENOTE_SERVER_TOKEN", TOKEN)
            .env("CENOTE_SERVER_MAX_SAMPLES", CAP.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawning cenote-server");
        let stdout = child.stdout.take().expect("stdout is piped");
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("reading the port line");
        if line.is_empty() {
            // EOF before the port line: the server failed to stand up —
            // on this machine, that means no capable GPU.
            child.wait().expect("reaping the failed spawn");
            eprintln!("skipping: cenote-server could not start (no GPU?)");
            return None;
        }
        let port = line
            .trim()
            .strip_prefix("cenote-server port=")
            .unwrap_or_else(|| panic!("unexpected port line: {line:?}"))
            .parse()
            .unwrap_or_else(|_| {
                panic!("the token must not print when supplied via env: {line:?}")
            });
        Some(Self { child, port })
    }

    fn connect(&self) -> TcpStream {
        let stream =
            TcpStream::connect(("127.0.0.1", self.port)).expect("connecting to the server");
        stream.set_nodelay(true).ok();
        stream
    }

    /// Wait for exit, bounded — a hung server fails the test, not the
    /// suite.
    fn wait(mut self, expect_success: bool) {
        let deadline = Instant::now() + DEADLINE;
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("polling the server") {
                break status;
            }
            assert!(Instant::now() < deadline, "the server never exited");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(
            status.success(),
            expect_success,
            "server exit status: {status:?}"
        );
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// One strict request/response exchange.
fn call(stream: &mut TcpStream, request: &Request) -> Response {
    protocol::write_message(stream, request).expect("writing a request");
    protocol::read_message(stream).expect("reading the response")
}

/// Poll the view until `accept` blesses an untorn snapshot. Every
/// snapshot seen on the way is held to the header's honesty invariant:
/// the converged flag is exactly "samples reached the cap", never a
/// leftover from a previous convergence run.
fn wait_for(view: &View, what: &str, accept: impl Fn(&Snapshot) -> bool) -> Snapshot {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(snapshot) = view.snapshot() {
            assert_eq!(
                snapshot.converged,
                snapshot.samples >= CAP,
                "the converged flag must match the frame it rides"
            );
            if accept(&snapshot) {
                return snapshot;
            }
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The genesis the test drives: a red emissive quad at x = 10, sized to
/// fill the test camera's whole frame. Deliberately *not* on the default
/// camera's axis, so a red frame proves the `SetCamera` lane moved the
/// view. No camera or settings op — the server injects its singletons,
/// exactly as it must for a Hydra genesis.
fn genesis() -> wire::ChangeSet {
    wire::ChangeSet {
        ops: vec![
            wire::Op::Mesh(wire::MeshPatch {
                name: "quad".into(),
                source: Some(wire::MeshSource::Inline {
                    positions: vec![
                        [8.0, -2.0, 0.0],
                        [12.0, -2.0, 0.0],
                        [12.0, 2.0, 0.0],
                        [8.0, 2.0, 0.0],
                    ],
                    normals: Some(vec![[0.0, 0.0, 1.0]; 4]),
                    uvs: None,
                    triangles: vec![[0, 1, 2], [0, 2, 3]],
                }),
            }),
            wire::Op::Material(Box::new(wire::MaterialPatch {
                name: "lamp".into(),
                // The saturated primary: authored Rec.709 pure red. If the
                // server's 3×3 is ever dropped, the ACEScg encoding leaks
                // ~11% into green and the color assertions fail.
                emission_color: Some(wire::Texturable::Constant([1.0, 0.0, 0.0])),
                emission_luminance: Some(5.0),
                ..wire::MaterialPatch::default()
            })),
            wire::Op::Instance(wire::InstancePatch {
                name: "panel".into(),
                mesh: Some("quad".into()),
                curves: None,
                material: Some("lamp".into()),
                transforms: None,
                camera_visible: None,
            }),
        ],
    }
}

/// The camera that frames the quad: 5 m back along +Z from its center.
fn quad_camera() -> Camera {
    Camera {
        position: [10.0, 0.0, 5.0],
        look_at: [10.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        vfov_degrees: 40.0,
        focus_distance: None,
        aperture_radius: 0.0,
    }
}

/// Assert a texel is saturated in `hot` and dark in the other two
/// channels — tight enough that an unconverted `ACEScg` primary (which
/// leaks 11–37% into its neighbours) fails.
fn assert_primary(snapshot: &Snapshot, texel: usize, hot: usize, what: &str) {
    let rgba = &snapshot.beauty[texel * 4..texel * 4 + 4];
    assert!(rgba[hot] > 0.1, "{what}: too dark, {rgba:?}");
    for channel in 0..3 {
        if channel != hot {
            assert!(
                rgba[channel].abs() < 0.05 * rgba[hot],
                "{what}: channel {channel} is not dark — the Rec.709 conversion dropped? {rgba:?}"
            );
        }
    }
}

/// The whole spine, end to end, in the order a real delegate would drive
/// it: handshake, resize, camera, genesis, convergence, a live edit, a
/// visual no-op, a rejected edit surfacing through the header counter
/// and `Ping`, and EOF as shutdown. Threaded through it all, the epoch
/// contract: every picture-changing reply carries the session
/// epoch, the shm header's stamp reaches it — even when nothing restarts
/// — and convergence is honest only under a stamp at or past the ack.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one linear conversation — the order of exchanges is the thing under test"
)]
fn the_transport_spine_renders_over_the_wire() {
    let Some(server) = Server::spawn() else {
        return;
    };
    let mut stream = server.connect();

    // Handshake: Welcome carries the protocol and the default-sized fb.
    let Response::Welcome { protocol, fb } = call(
        &mut stream,
        &Request::Hello {
            protocol: PROTOCOL,
            token: TOKEN.into(),
        },
    ) else {
        panic!("Hello must be answered by Welcome");
    };
    assert_eq!(protocol, PROTOCOL);
    assert_eq!((fb.width, fb.height), (1280, 720), "the default size");
    View::open(&fb).expect("the initial segment maps");

    // Resize: a new segment, immediately mappable; the old one survives
    // until the next request proves this reply was processed. The reply
    // carries the post-request epoch — resize is a picture-changing verb
    // so the counter has moved off its starting zero.
    let Response::Resized {
        fb: resized,
        epoch: resize_epoch,
    } = call(
        &mut stream,
        &Request::Resize {
            width: 128,
            height: 128,
        },
    ) else {
        panic!("Resize must be answered by Resized");
    };
    assert_eq!((resized.width, resized.height), (128, 128));
    assert_ne!(resized.shm_name, fb.shm_name);
    assert!(resize_epoch >= 1, "a resize must advance the epoch");
    let view = View::open(&resized).expect("the resized segment maps");
    View::open(&fb).expect("the old segment lives until the next request");

    // The camera lane, then genesis. After SetCamera (the request that
    // proves Resized was processed) the old segment must be gone. Every
    // picture-changing request returns a strictly larger epoch.
    let Response::Ack {
        rejected,
        epoch: camera_epoch,
    } = call(&mut stream, &Request::SetCamera(quad_camera())) else {
        panic!("SetCamera must be answered by Ack");
    };
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(camera_epoch > resize_epoch);
    let Response::Ack {
        epoch: genesis_epoch,
        ..
    } = call(&mut stream, &Request::Replace(genesis())) else {
        panic!("Replace must be answered by Ack");
    };
    assert!(genesis_epoch > camera_epoch);
    assert!(
        View::open(&fb).is_err(),
        "the replaced segment must be unlinked after the next request"
    );

    // Accumulate to the cap: the converged flag flips, samples land on
    // it exactly, the frame counter is the tear protocol's monotonic
    // clock — and the header's epoch reaches the acked one, so the
    // settled picture provably incorporates the genesis.
    let converged = wait_for(&view, "the capped frame", |snapshot| {
        assert!(snapshot.samples <= CAP, "overshot the cap");
        snapshot.samples == CAP && snapshot.converged && snapshot.epoch >= genesis_epoch
    });
    assert!(view.converged());
    assert!(view.epoch() >= genesis_epoch);

    // The saturated primary, everywhere the quad is — which is the whole
    // frame: center and all four corners, red in Rec.709.
    let center = (64 * 128 + 64) as usize;
    assert_primary(&converged, center, 0, "center");
    for corner in [0, 127, 128 * 127, 128 * 128 - 1] {
        assert_primary(&converged, corner, 0, "corner");
    }
    // Depth crosses unconverted: the quad stands 5 m from the camera.
    let depth = converged.depth[center];
    assert!((depth - 5.0).abs() < 0.05, "center depth: {depth}");
    assert!(view.snapshot().expect("still readable").counter >= converged.counter);

    // A live edit: the lamp turns green, accumulation restarts, and the
    // new color arrives — the stop → apply → re-prep → restart loop over
    // the wire. (An unconverted ACEScg green would leak 37% into red.)
    // Convergence is claimed honestly again only under a stamp at or past
    // the Ack's epoch: the red frame's stale flag can never pass for the
    // green picture's.
    let greening = wire::ChangeSet {
        ops: vec![wire::Op::Material(Box::new(wire::MaterialPatch {
            name: "lamp".into(),
            emission_color: Some(wire::Texturable::Constant([0.0, 1.0, 0.0])),
            ..wire::MaterialPatch::default()
        }))],
    };
    let Response::Ack {
        epoch: green_epoch, ..
    } = call(&mut stream, &Request::Apply(greening.clone())) else {
        panic!("Apply must be answered by Ack");
    };
    assert!(green_epoch > genesis_epoch);
    let green = wait_for(&view, "the settled green frame", |snapshot| {
        snapshot.epoch >= green_epoch && snapshot.converged
    });
    assert_eq!(green.samples, CAP);
    assert_primary(&green, center, 1, "center after the edit");

    // The same patch again — a visual no-op: the equality gate dirties
    // nothing, nothing restarts, yet the epoch still advances and the
    // parked render thread republishes the settled image under the fresh
    // stamp (the epoch's delivery guarantee). Without the republish, honest
    // convergence would wedge here.
    let Response::Ack {
        rejected,
        epoch: noop_epoch,
    } = call(&mut stream, &Request::Apply(greening)) else {
        panic!("Apply must be answered by Ack");
    };
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(noop_epoch > green_epoch);
    assert!(view.converged(), "a no-op must not unsettle the picture");
    let republished = wait_for(&view, "the republished settled frame", |snapshot| {
        snapshot.epoch >= noop_epoch
    });
    assert!(republished.converged, "the republish carries the settled state");
    assert_eq!(republished.samples, CAP);
    assert_primary(&republished, center, 1, "center after the no-op");

    // A rejected edit: the Ack is a receipt, the rejection surfaces
    // through the header's counter, and Ping collects the message. The
    // epoch advances all the same — a rejected edit is *incorporated*
    // (as nothing), so waiting on its stamp can never wedge — and the
    // parked republish delivers it without a restart.
    let rejected_before = view.rejected_edits();
    let Response::Ack {
        epoch: broken_epoch,
        ..
    } = call(
        &mut stream,
        &Request::Apply(wire::ChangeSet {
            ops: vec![wire::Op::Instance(wire::InstancePatch {
                name: "broken".into(),
                mesh: Some("no-such-mesh".into()),
                curves: None,
                material: Some("lamp".into()),
                transforms: None,
                camera_visible: None,
            })],
        }),
    ) else {
        panic!("Apply must be answered by Ack");
    };
    assert!(broken_epoch > noop_epoch);
    let deadline = Instant::now() + DEADLINE;
    while view.rejected_edits() == rejected_before {
        assert!(Instant::now() < deadline, "the rejection never surfaced");
        std::thread::sleep(Duration::from_millis(10));
    }
    let after_rejection = wait_for(&view, "the frame past the rejected edit", |snapshot| {
        snapshot.epoch >= broken_epoch
    });
    assert!(after_rejection.converged);
    assert_primary(&after_rejection, center, 1, "center after the rejection");
    let Response::Ack {
        rejected,
        epoch: ping_epoch,
    } = call(&mut stream, &Request::Ping) else {
        panic!("Ping must be answered by Ack");
    };
    assert!(
        rejected.iter().any(|message| message.contains("no-such-mesh")),
        "the rejection message must ride the next response: {rejected:?}"
    );
    assert_eq!(ping_epoch, broken_epoch, "a Ping changes no picture");

    // EOF is shutdown: exit 0, and the segment's name unlinks with it.
    drop(stream);
    server.wait(true);
    assert!(
        View::open(&resized).is_err(),
        "a clean exit must unlink its segment"
    );
}

/// The token is load-bearing: a client that presents the wrong secret is
/// refused — the conversation dies and the server exits nonzero, exactly
/// like any other handshake violation.
#[test]
fn a_wrong_token_is_refused() {
    let Some(server) = Server::spawn() else {
        return;
    };
    let mut stream = server.connect();
    protocol::write_message(
        &mut stream,
        &Request::Hello {
            protocol: PROTOCOL,
            token: "not-the-token".into(),
        },
    )
    .expect("writing the bad Hello");
    let refused = protocol::read_message::<Response>(&mut stream);
    assert!(refused.is_err(), "a bad token must not be welcomed");
    server.wait(false);
}
