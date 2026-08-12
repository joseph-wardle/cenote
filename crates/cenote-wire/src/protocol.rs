//! The control channel's envelope and framing.
//!
//! Strict request/response over loopback TCP: the client sends one
//! [`Request`], the server sends exactly one [`Response`], and the server
//! never speaks unprompted — continuous state (frame counter, samples,
//! converged, rejected-edit count) rides the shared-memory header
//! ([`crate::fb`]), never the socket, so `IsConverged()` is a header read
//! and the C++ client needs no reader thread.
//!
//! Framing: a `u32` little-endian byte length, then that many bytes of
//! `MessagePack` — `rmp-serde` defaults, structs as positional arrays. The
//! payload bytes are the cross-language contract the golden corpus pins.
//!
//! There is no shutdown message: the server is spawned per-client, so
//! closing the socket (EOF) *is* shutdown.

use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::scene::ChangeSet;

/// The protocol revision carried in `Hello`/`Welcome`. Bump on any change
/// to the encoded shape of any type in this crate — and regenerate the
/// golden corpus in the same commit.
pub const PROTOCOL: u32 = 9;

/// The most bytes one frame may claim, a plausibility guard so a garbage
/// or hostile length prefix cannot ask the reader to allocate without
/// bound. Real change-sets stay well under it — bulk geometry crosses by
/// PLY reference, not by value.
pub const MAX_MESSAGE_BYTES: u32 = 1 << 30;

/// The active camera, as `SetCamera` carries it — the same fields as the
/// renderer's `scene::Camera`, because the request feeds the session's
/// inputs lane directly (which camera is active is host view state, not
/// scene data; see `scene::CameraPatch`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    /// Eye position, meters.
    pub position: [f32; 3],
    /// The point the view axis passes through.
    pub look_at: [f32; 3],
    /// Screen-up direction (carries roll).
    pub up: [f32; 3],
    /// Vertical field of view, degrees, in (0, 180).
    pub vfov_degrees: f32,
    /// Distance to the focal plane, meters; `None` focuses at `look_at`.
    pub focus_distance: Option<f32>,
    /// Lens radius, meters; 0 is a pinhole.
    pub aperture_radius: f32,
}

/// Where the pixels live: one POSIX shared-memory segment per size, named
/// in-band so nothing about the framebuffer is out-of-band configuration.
/// The segment's own header ([`crate::fb`]) carries the full layout; this
/// descriptor is what a client needs to find and map it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FbDesc {
    /// The `shm_open` name, `/cenote-<pid>-<generation>` — pid keeps
    /// concurrent servers apart, generation keeps a resize's new segment
    /// apart from the old one it replaces.
    pub shm_name: String,
    /// Total segment length in bytes (header page + both buffers).
    pub bytes: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// A client message. Every variant is answered by exactly one [`Response`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Request {
    /// The handshake, first on the socket: the protocol revision this
    /// client speaks and the spawn-time token that proves it is the
    /// process that spawned the server (loopback TCP is reachable by any
    /// local process; the token closes that gap). A mismatch on either is
    /// fatal — the server logs and exits nonzero.
    Hello {
        /// The client's [`PROTOCOL`].
        protocol: u32,
        /// The spawn-time secret, hex.
        token: String,
    },
    /// The whole scene from empty — genesis and stage-reload. Objects the
    /// set no longer contains are removed server-side (the file-reload
    /// diff), so replay needs no delete bookkeeping.
    Replace(ChangeSet),
    /// An overlay edit onto the current scene — the steady-state delta.
    Apply(ChangeSet),
    /// The inputs-lane fast path for the active camera. Mandatory rather
    /// than an optimization: the session's inputs-lane camera overwrites
    /// the scene camera at every wave, so a camera travelling inside a
    /// `ChangeSet` would be silently dead.
    SetCamera(Camera),
    /// A new framebuffer size (the host viewport resized). The server
    /// allocates the new segment and resizes the session *before*
    /// replying, so the reply's descriptor is immediately mappable. The
    /// previous segment is unlinked when the client's next request proves
    /// this reply was processed.
    Resize {
        /// New width in pixels.
        width: u32,
        /// New height in pixels.
        height: u32,
    },
    /// A liveness probe — and the way an idle client collects rejection
    /// messages after the shm header's `rejected_edits` counter moves.
    Ping,
}

/// The server's reply — one per [`Request`], in order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Response {
    /// Answers `Hello`: the server's protocol revision and the initial
    /// framebuffer.
    Welcome {
        /// The server's [`PROTOCOL`].
        protocol: u32,
        /// The initial framebuffer segment.
        fb: FbDesc,
    },
    /// Answers `Replace`, `Apply`, `SetCamera`, and `Ping`. A receipt, not
    /// a validation: edits apply asynchronously at the render loop's next
    /// wave boundary, so rejections surface *later* — `rejected` carries
    /// every rejection message accumulated since the last response, and
    /// the shm header's monotonic `rejected_edits` counter tells an idle
    /// client to `Ping` for them.
    Ack {
        /// Rejection messages accumulated since the last response.
        rejected: Vec<String>,
        /// The session epoch after this request was placed, read
        /// server-side after every session call the request caused —
        /// including internal ones like the camera re-assert. The first
        /// shm frame whose header epoch reaches this value incorporates
        /// everything sent so far, applied or rejected.
        epoch: u64,
    },
    /// Answers `Resize`: the new framebuffer segment, ready to map.
    Resized {
        /// The new framebuffer segment, ready to map.
        fb: FbDesc,
        /// The session epoch after the resize — [`Response::Ack::epoch`]'s
        /// twin, on the one reply that is not an `Ack`.
        epoch: u64,
    },
}

/// Serialize one message to its wire payload (no length prefix) —
/// `rmp-serde` defaults, the byte shape the golden corpus pins.
///
/// # Errors
///
/// `io::ErrorKind::InvalidData` if the value cannot encode — with these
/// types, only a string that is not valid UTF-8 deep inside, so in
/// practice never.
pub fn encode<T: Serialize>(message: &T) -> io::Result<Vec<u8>> {
    rmp_serde::to_vec(message).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Deserialize one wire payload (no length prefix).
pub fn decode<T: DeserializeOwned>(payload: &[u8]) -> io::Result<T> {
    rmp_serde::from_slice(payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Write one framed message: u32-LE length, then the payload.
pub fn write_message<T: Serialize>(writer: &mut impl Write, message: &T) -> io::Result<()> {
    let payload = encode(message)?;
    let length = u32::try_from(payload.len())
        .ok()
        .filter(|&length| length <= MAX_MESSAGE_BYTES)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("message of {} bytes exceeds the frame cap", payload.len()),
            )
        })?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

/// Read one framed message. `Err(UnexpectedEof)` before the first length
/// byte is the orderly end of the conversation — the caller decides
/// whether that is shutdown (server side) or a dead server (client side).
///
/// # Errors
///
/// EOF and transport errors from the reader; `InvalidData` for a length
/// prefix past [`MAX_MESSAGE_BYTES`] or a payload that does not decode.
pub fn read_message<T: DeserializeOwned>(reader: &mut impl Read) -> io::Result<T> {
    let mut length = [0u8; 4];
    reader.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length);
    if length > MAX_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {length} exceeds the cap"),
        ));
    }
    let mut payload = vec![0u8; length as usize];
    reader.read_exact(&mut payload)?;
    decode(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{CameraPatch, MaterialPatch, Op, Reset, TextureRef};

    /// A frame round-trips through the length-prefixed encoding.
    #[test]
    fn framing_round_trips() {
        let request = Request::Apply(ChangeSet {
            ops: vec![Op::Camera(CameraPatch {
                name: "main".into(),
                position: Some([1.0, 2.0, 3.0]),
                ..CameraPatch::default()
            })],
        });
        let mut wire = Vec::new();
        write_message(&mut wire, &request).expect("write");
        assert_eq!(
            u32::from_le_bytes(wire[..4].try_into().expect("prefix")) as usize,
            wire.len() - 4
        );
        let back: Request = read_message(&mut wire.as_slice()).expect("read");
        assert_eq!(back, request);
    }

    /// The reason [`Reset`] exists: all three states of a doubly-optional
    /// field must survive the codec distinctly. A bare `Option<Option<T>>`
    /// fails this — `MessagePack` nil collapses `Some(None)` into `None`.
    #[test]
    fn reset_keeps_all_three_states_distinct() {
        let states = [
            None,
            Some(Reset::Clear),
            Some(Reset::Set(TextureRef {
                path: "/n.png".into(),
                color_space: None,
                channel: None,
            })),
        ];
        let mut encodings = Vec::new();
        for state in states {
            let patch = MaterialPatch {
                name: "m".into(),
                geometry_normal: state.clone(),
                ..MaterialPatch::default()
            };
            let bytes = encode(&patch).expect("encode");
            let back: MaterialPatch = decode(&bytes).expect("decode");
            assert_eq!(back.geometry_normal, state);
            encodings.push(bytes);
        }
        encodings.dedup();
        assert_eq!(encodings.len(), 3, "the three states must encode apart");
    }

    /// A hostile or garbage length prefix is refused before allocation.
    #[test]
    fn the_frame_cap_rejects_absurd_lengths() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&u32::MAX.to_le_bytes());
        wire.extend_from_slice(b"junk");
        let error = read_message::<Request>(&mut wire.as_slice()).expect_err("must refuse");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
