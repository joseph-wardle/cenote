//! The out-of-process render server : a loopback-TCP
//! request/response loop and a shared-memory framebuffer wrapped around
//! `render::Session`. Spawned per client by a scene-graph host (the
//! Hydra delegate); driven by the wire vocabulary in `cenote-wire`.
//!
//! The process contract, which the spawner and the integration test both
//! parse:
//!
//! - binds `127.0.0.1:0` and prints exactly **one stdout line** —
//!   `cenote-server port=<N>`, with ` token=<hex>` appended when
//!   `CENOTE_SERVER_TOKEN` was unset and the token self-generated. All
//!   logging goes to stderr.
//! - the GPU context and the `Session` (an empty scene — camera and
//!   settings singletons only — at the default 1280×720) are created
//!   *before* the port line prints, so a GPU failure fails the spawn
//!   legibly instead of surfacing as a dead socket later.
//! - one client. EOF on the socket is shutdown: exit 0. A render-thread
//!   fault (or any server-side failure) exits nonzero with the fault on
//!   stderr, so the client's dead-socket path handles both deaths
//!   identically. A handshake violation — wrong protocol, wrong token,
//!   anything before `Hello` — is a failure, not a negotiation.
//! - `CENOTE_SERVER_MAX_SAMPLES` (default 4096) caps accumulation; the
//!   shm header's `converged` flag reports reaching it. A finer

//!
//! Camera ownership: the session's inputs-lane camera overwrites the
//! scene camera at every wave, so the *only* way to move the view is the
//! `SetCamera` request, and the server re-asserts the last `SetCamera`
//! after any change-set that touches a camera object (a `Replace`
//! creating the singleton would otherwise clobber the view for a wave).
//! Symmetrically, a `Replace` that carries no camera or settings op gets
//! the defaults injected, since a Hydra genesis never sends either —
//! prep's singleton rule is the server's to satisfy, not the wire's.

use std::fmt::Write as _;
use std::io::{Read as _, Write as _};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use cenote::render::{AutoStop, Renderer, Session};
use cenote::scene::Scene;
use cenote::scene::changeset::{CameraPatch, ChangeSet, Kind, Op, SettingsPatch};
use cenote::scene::description::{SceneDescription, Settings};
use cenote_server::{shm, translate};
use cenote_wire::protocol::{self, PROTOCOL, Request, Response};
use glam::Vec3;

/// The frame pump's poll cadence — the refresh throttle (~30 FPS). The
/// session publishes on its own schedule; this only bounds how often the
/// server looks.
const FRAME_INTERVAL: Duration = Duration::from_millis(33);

/// The default accumulation cap when `CENOTE_SERVER_MAX_SAMPLES` is
/// unset: deep enough for a clean lookdev still, finite so a settled
/// view releases the GPU.
const DEFAULT_MAX_SAMPLES: u32 = 4096;

/// Everything the socket loop and the frame pump share, one lock.
struct Shared {
    session: Session,
    /// The last `SetCamera`, re-asserted after camera-touching edits.
    camera: cenote::scene::Camera,
    /// The live framebuffer segment.
    fb: shm::Segment,
    /// The segment a resize replaced, kept mapped until the client's next
    /// request proves the `Resized` reply was processed; dropping unlinks.
    doomed: Option<shm::Segment>,
    /// Segment-name generation, bumped per resize.
    generation: u64,
    /// Rejection messages awaiting the next response.
    rejected: Vec<String>,
    /// A pump-side fatal fault, surfaced by main as the exit reason.
    fault: Option<String>,
    max_samples: u32,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let max_samples = match std::env::var("CENOTE_SERVER_MAX_SAMPLES") {
        Ok(raw) => raw
            .parse::<u32>()
            .ok()
            .filter(|&cap| cap > 0)
            .with_context(|| {
                format!("CENOTE_SERVER_MAX_SAMPLES must be a positive integer, got \"{raw}\"")
            })?,
        Err(_) => DEFAULT_MAX_SAMPLES,
    };
    let (token, generated) = match std::env::var("CENOTE_SERVER_TOKEN") {
        Ok(token) if !token.is_empty() => (token, false),
        _ => (generate_token()?, true),
    };

    let listener = TcpListener::bind(("127.0.0.1", 0)).context("binding the loopback listener")?;
    let port = listener.local_addr()?.port();

    // Everything that can fail expensively fails *before* the port line:
    // a spawner that reads the line knows the GPU and the framebuffer are
    // both up.
    let gpu = Arc::new(cenote::gpu::Context::new().context("creating the GPU context")?);
    let mut description = SceneDescription::new();
    description
        .apply(&ChangeSet {
            ops: vec![
                Op::Camera(CameraPatch::new("camera")),
                Op::Settings(SettingsPatch::new("settings")),
            ],
        })
        .expect("the empty scene's singletons are valid");
    let scene = Scene::prep(&gpu, &mut description).context("preparing the empty scene")?;
    let camera = *scene.camera();
    let renderer = Renderer::new(&gpu).context("creating the renderer")?;
    let [width, height] = Settings::default().resolution;
    let session = Session::new(
        Arc::clone(&gpu),
        description,
        scene,
        renderer,
        camera,
        width,
        height,
        AutoStop {
            max_samples,
            noise_threshold: None,
        },
    );
    let fb = shm::Segment::create(1, width, height).context("allocating the framebuffer")?;
    let state = Arc::new(Mutex::new(Shared {
        session,
        camera,
        fb,
        doomed: None,
        generation: 1,
        rejected: Vec::new(),
        fault: None,
        max_samples,
    }));

    // The one stdout line the spawner parses.
    if generated {
        println!("cenote-server port={port} token={token}");
    } else {
        println!("cenote-server port={port}");
    }
    std::io::stdout().flush()?;

    let (mut stream, peer) = listener.accept().context("accepting the client")?;
    log::info!("client connected from {peer}");
    stream.set_nodelay(true).ok();

    let stop = Arc::new(AtomicBool::new(false));
    let pump = {
        let state = Arc::clone(&state);
        let gpu = Arc::clone(&gpu);
        let stop = Arc::clone(&stop);
        let socket = stream
            .try_clone()
            .context("cloning the socket for the pump")?;
        std::thread::Builder::new()
            .name("cenote-frame-pump".into())
            .spawn(move || pump(&state, &gpu, &stop, &socket))
            .context("spawning the frame pump")?
    };

    let served = serve(&mut stream, &state, &token);
    stop.store(true, Ordering::Relaxed);
    pump.join().expect("the frame pump does not panic");

    // A render fault outranks whatever the socket said — the shutdown
    // that unblocked `serve` *was* the fault surfacing.
    if let Some(fault) = state.lock().expect("state lock").fault.take() {
        anyhow::bail!(fault);
    }
    served
    // `state` (and with it the Session and both segments) drops here:
    // the render thread joins, the shm names unlink.
}

/// 128 bits from the OS, hex — the spawn-time secret that closes the
/// any-local-process gap loopback TCP leaves open. `/dev/urandom` because
/// the framebuffer already makes this binary POSIX-only.
fn generate_token() -> anyhow::Result<String> {
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .context("reading /dev/urandom for the token")?;
    Ok(bytes.iter().fold(String::new(), |mut hex, byte| {
        write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
        hex
    }))
}

/// The socket loop: handshake, then strictly one response per request.
/// Returns `Ok(())` on EOF — the client hanging up is the shutdown
/// signal — and `Err` on anything that violates the protocol.
fn serve(stream: &mut TcpStream, state: &Mutex<Shared>, token: &str) -> anyhow::Result<()> {
    let hello: Request = protocol::read_message(stream).context("reading Hello")?;
    let Request::Hello {
        protocol: client,
        token: presented,
    } = hello
    else {
        anyhow::bail!("the first message must be Hello");
    };
    anyhow::ensure!(
        client == PROTOCOL,
        "protocol mismatch: client speaks {client}, server speaks {PROTOCOL}"
    );
    anyhow::ensure!(presented == token, "the client's token does not match");
    let welcome = Response::Welcome {
        protocol: PROTOCOL,
        fb: state.lock().expect("state lock").fb.desc(),
    };
    protocol::write_message(stream, &welcome).context("writing Welcome")?;

    loop {
        let request: Request = match protocol::read_message(stream) {
            Ok(request) => request,
            // EOF is shutdown, mid-frame or between frames — the client
            // is gone either way, and it owes us nothing further.
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error).context("reading a request"),
        };
        let mut shared = state.lock().expect("state lock");
        // Any request past the handshake proves the previous reply was
        // processed — the segment that reply replaced can unlink now.
        shared.doomed = None;
        let response = match request {
            Request::Hello { .. } => anyhow::bail!("Hello arrived twice"),
            Request::Replace(set) => {
                let mut set = translate::change_set(set);
                ensure_singletons(&mut set);
                let touches_camera = touches(&set, Kind::Camera);
                shared.session.replace(set);
                if touches_camera {
                    let camera = shared.camera;
                    shared.session.set_camera(camera);
                }
                ack(&mut shared)
            }
            Request::Apply(set) => {
                let set = translate::change_set(set);
                let touches_camera = touches(&set, Kind::Camera);
                shared.session.apply(set);
                if touches_camera {
                    let camera = shared.camera;
                    shared.session.set_camera(camera);
                }
                ack(&mut shared)
            }
            Request::SetCamera(camera) => {
                let camera = translate::camera(camera);
                shared.camera = camera;
                shared.session.set_camera(camera);
                ack(&mut shared)
            }
            Request::Resize { width, height } => {
                anyhow::ensure!(width > 0 && height > 0, "a zero-sized resize");
                shared.generation += 1;
                let fresh = shm::Segment::create(shared.generation, width, height)
                    .context("allocating the resized framebuffer")?;
                shared.session.resize(width, height);
                shared.doomed = Some(std::mem::replace(&mut shared.fb, fresh));
                Response::Resized {
                    fb: shared.fb.desc(),
                    epoch: shared.session.epoch(),
                }
            }
            Request::Ping => ack(&mut shared),
        };
        drop(shared);
        protocol::write_message(stream, &response).context("writing a response")?;
    }
}

/// The `Ack`, carrying every rejection accumulated since the last
/// response — a receipt, not a validation (edits land at wave boundaries)
/// — and the session epoch after the request. The epoch is read
/// *last*, after every session call the request caused (including the
/// internal camera re-assert, which bumps harmlessly), so the first frame
/// stamped at or past it provably incorporates this request.
fn ack(shared: &mut Shared) -> Response {
    Response::Ack {
        rejected: std::mem::take(&mut shared.rejected),
        epoch: shared.session.epoch(),
    }
}

/// Whether any op in the set targets `kind`.
fn touches(set: &ChangeSet, kind: Kind) -> bool {
    set.ops.iter().any(|op| op.target().0 == kind)
}

/// A Hydra genesis carries no camera or settings op — the active camera
/// travels on the `SetCamera` lane and render settings are host state —
/// but prep requires the singletons, so a `Replace` lacking them gets the
/// defaults appended. Sets that do carry their own (a scene file driven
/// over the wire) pass through untouched.
fn ensure_singletons(set: &mut ChangeSet) {
    if !touches(set, Kind::Camera) {
        set.ops.push(Op::Camera(CameraPatch::new("camera")));
    }
    if !touches(set, Kind::Settings) {
        set.ops.push(Op::Settings(SettingsPatch::new("settings")));
    }
}

/// The frame pump: poll the session at the throttle, download the newest
/// frame, convert the beauty to linear `Rec.709` (the one 3×3, hoisted —
/// the silent-gamut defence), and publish into shm. Also the faults
/// lane: a dead render thread or a failed download records the fault and
/// shuts the socket down, which unblocks `serve` and ends the process
/// nonzero.
fn pump(state: &Mutex<Shared>, gpu: &cenote::gpu::Context, stop: &AtomicBool, socket: &TcpStream) {
    let matrix = cenote::color::rec709_from_acescg();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(FRAME_INTERVAL);
        let frame = {
            let mut shared = state.lock().expect("state lock");
            if let Err(fault) = shared.session.check() {
                return fail(&mut shared, socket, format!("render thread fault: {fault}"));
            }
            if let Some(error) = shared.session.take_edit_error() {
                log::warn!("edit rejected: {error}");
                shared.rejected.push(error.to_string());
                shared.fb.bump_rejected();
            }
            shared.session.take_frame()
        };
        let Some(frame) = frame else { continue };
        // The downloads run unlocked: the frame owns its buffers, and the
        // socket loop must never wait on a PCIe copy.
        let beauty = match gpu.download_buffer(frame.beauty()) {
            Ok(bytes) => bytes,
            Err(error) => {
                let mut shared = state.lock().expect("state lock");
                return fail(
                    &mut shared,
                    socket,
                    format!("downloading the beauty: {error}"),
                );
            }
        };
        let depth = match gpu.download_buffer(frame.depth()) {
            Ok(bytes) => bytes,
            Err(error) => {
                let mut shared = state.lock().expect("state lock");
                return fail(
                    &mut shared,
                    socket,
                    format!("downloading the depth: {error}"),
                );
            }
        };
        let beauty = rec709_texels(&matrix, &beauty);
        let mut shared = state.lock().expect("state lock");
        // A frame sized for a segment a resize already replaced is simply
        // skipped; the session is rebuilding its film at the new size.
        if (frame.width(), frame.height()) == (shared.fb.width(), shared.fb.height()) {
            let converged = frame.samples() >= shared.max_samples;
            shared
                .fb
                .publish(&beauty, &depth, frame.samples(), converged, frame.epoch());
        }
    }
}

/// Record a pump-side fatal fault and shut the socket, unblocking the
/// request loop so main can exit nonzero with the message.
fn fail(shared: &mut Shared, socket: &TcpStream, message: String) {
    log::error!("{message}");
    shared.fault = Some(message);
    socket.shutdown(Shutdown::Both).ok();
}

/// Downloaded `ACEScg` RGBA texels → linear `Rec.709`, alpha untouched.
///
/// The negative lobe of the 3×3 is clamped away: an `ACEScg` colour more
/// saturated than `Rec.709` can hold maps to a component below zero (the AP1
/// gamut is wider than 709), and this is the display-referred buffer usdview,
/// usdrecord, and husk read — a consumer's transfer curve raises each
/// component to a power, and `powf` of a negative is `NaN`. A display cannot
/// emit negative light, so clamping to zero is the honest gamut clip and the
/// completion of the silent-gamut defence: without it a saturated primary
/// riddles the frame with `NaN` (the failure the end-to-end golden
/// surfaced). In-gamut colour is untouched, and highlights above one stay —
/// only the below-zero out-of-gamut lobe is clipped.
fn rec709_texels(matrix: &glam::Mat3, acescg: &[u8]) -> Vec<f32> {
    let mut rec709 = Vec::with_capacity(acescg.len() / 4);
    for texel in acescg.chunks_exact(16) {
        let channel = |index: usize| {
            f32::from_le_bytes(
                texel[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("a 4-byte channel"),
            )
        };
        let rgb = (*matrix * Vec3::new(channel(0), channel(1), channel(2))).max(Vec3::ZERO);
        rec709.extend_from_slice(&[rgb.x, rgb.y, rgb.z, channel(3)]);
    }
    rec709
}

#[cfg(test)]
mod tests {
    use super::rec709_texels;

    /// One `ACEScg` RGBA texel as the 16 little-endian bytes a download hands us.
    fn texel(rgba: [f32; 4]) -> Vec<u8> {
        rgba.iter().flat_map(|c| c.to_le_bytes()).collect()
    }

    /// A saturated `ACEScg` primary sits outside the `Rec.709` gamut, so the
    /// `3x3` sends two of its components below zero; the conversion must clamp
    /// them away rather than publish negative light a consumer's transfer curve
    /// would turn into `NaN` (the step-6 golden's failure mode). The clamp is
    /// the completion of the silent-gamut defence.
    #[test]
    fn saturated_primaries_clamp_to_nonnegative() {
        let matrix = cenote::color::rec709_from_acescg();
        for primary in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]] {
            let out = rec709_texels(&matrix, &texel(primary));
            assert!(
                out[..3].iter().all(|c| *c >= 0.0),
                "ACEScg {primary:?} left a negative Rec.709 component: {:?}",
                &out[..3]
            );
            assert!((out[3] - 1.0).abs() < 1e-6, "alpha must pass through untouched");
        }
    }

    /// In-gamut colour is untouched: an `ACEScg` grey is grey in `Rec.709` (the
    /// shared white point), so the clamp is a no-op and the value round-trips.
    #[test]
    fn in_gamut_colour_is_unchanged() {
        let matrix = cenote::color::rec709_from_acescg();
        let out = rec709_texels(&matrix, &texel([0.5, 0.5, 0.5, 1.0]));
        for c in &out[..3] {
            assert!((c - 0.5).abs() < 1e-5, "grey should survive the conversion: {out:?}");
        }
    }
}
