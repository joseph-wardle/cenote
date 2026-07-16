//! The shared-memory framebuffer — the POSIX half of the pixel transport
//! (D-101). The *layout* (header fields, plane offsets, tear protocol)
//! lives in [`cenote_wire::fb`] where both languages can mirror it; this
//! module is the ~twenty deliberately platform-specific lines that
//! create, map, and unlink a segment, plus the writer and reader that
//! follow that layout. [`Segment`] is the server's writer; [`View`] is
//! the reader — used by the integration test today, and the executable
//! specification of what the C++ delegate's `HdRenderBuffer::Map` does
//! in step 2.
//!
//! This file is the crate's unsafe quarantine (the `gpu` module's rule,
//! applied here): raw `libc` calls and raw-pointer stores stay inside,
//! and everything exported is safe.
//!
//! One pragmatic note on the tear protocol: the reader's copy races the
//! writer's stores by design, and the counter check *discards* any copy
//! that could have been torn (see `fb`'s module doc). Copying the planes
//! with plain loads is the same pragmatism every shared-memory seqlock
//! ships — the double buffer means a copy that validates was never
//! written during the read.

use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use cenote_wire::fb::{
    self, HEADER_BYTES, LAYOUT_VERSION, MAGIC, beauty_offset, depth_offset, header, segment_bytes,
};
use cenote_wire::protocol::FbDesc;

/// The raw mapping: name, base pointer, length, and whether this side
/// created (and therefore unlinks) the segment. Shared by writer and
/// reader; all field access goes through the offset helpers below.
struct Mapping {
    name: std::ffi::CString,
    base: *mut u8,
    bytes: usize,
    owns: bool,
}

// The mapping is plain memory; the pointer is not thread-affine. Sharing
// requires the same care as any &AtomicU32 — which is how it is accessed.
unsafe impl Send for Mapping {}

impl Mapping {
    /// `shm_open` + `ftruncate` + `mmap` (create), or `shm_open` + `mmap`
    /// (open). Fresh segments are zero-filled by the kernel, so a created
    /// header starts with every counter at 0.
    fn new(name: &str, bytes: u64, create: bool) -> io::Result<Self> {
        let cname = std::ffi::CString::new(name)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "shm name holds a NUL"))?;
        let (flags, prot) = if create {
            (
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        } else {
            (libc::O_RDONLY, libc::PROT_READ)
        };
        // SAFETY: plain syscalls on an owned CString; the fd is closed on
        // every path below (the mapping outlives it).
        unsafe {
            let fd = libc::shm_open(cname.as_ptr(), flags, 0o600);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            if create && libc::ftruncate(fd, bytes.cast_signed()) != 0 {
                let error = io::Error::last_os_error();
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
                return Err(error);
            }
            let base = libc::mmap(
                std::ptr::null_mut(),
                bytes as libc::size_t,
                prot,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if base == libc::MAP_FAILED {
                let error = io::Error::last_os_error();
                if create {
                    libc::shm_unlink(cname.as_ptr());
                }
                return Err(error);
            }
            Ok(Self {
                name: cname,
                base: base.cast(),
                bytes: bytes as usize,
                owns: create,
            })
        }
    }

    /// A header field as an atomic — the only way the two processes'
    /// loads and stores are ordered. Offsets come from
    /// [`cenote_wire::fb::header`], all 4-aligned inside the first page.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the fb layout aligns every header field; pinned by fb's tests"
    )]
    fn atomic_u32(&self, offset: u64) -> &AtomicU32 {
        debug_assert!(offset + 4 <= HEADER_BYTES && offset.is_multiple_of(4));
        // SAFETY: in-bounds (one page, mapped), aligned, and only ever
        // accessed atomically from both processes.
        unsafe { &*self.base.add(offset as usize).cast::<AtomicU32>() }
    }

    /// The u64 twin of [`Self::atomic_u32`], for the frame counter.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the fb layout aligns every header field; pinned by fb's tests"
    )]
    fn atomic_u64(&self, offset: u64) -> &AtomicU64 {
        debug_assert!(offset + 8 <= HEADER_BYTES && offset.is_multiple_of(8));
        // SAFETY: as `atomic_u32`, 8-aligned by the layout.
        unsafe { &*self.base.add(offset as usize).cast::<AtomicU64>() }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping the exact region mapped above; unlink is the
        // creator's job and is idempotent (a name can only be unlinked
        // once — later calls fail harmlessly).
        unsafe {
            libc::munmap(self.base.cast(), self.bytes);
            if self.owns {
                libc::shm_unlink(self.name.as_ptr());
            }
        }
    }
}

/// The writer: creates its segment, publishes frames under the tear
/// protocol, and unlinks the name when dropped (a resize drops the old
/// segment once the client's next request proves the new descriptor was
/// seen).
pub struct Segment {
    map: Mapping,
    width: u32,
    height: u32,
    name: String,
}

impl Segment {
    /// Create and map `/cenote-<pid>-<generation>` sized for
    /// `width`×`height`, header initialized, both buffers zero.
    ///
    /// # Errors
    ///
    /// The OS error from `shm_open`/`ftruncate`/`mmap` — typically
    /// `AlreadyExists` if a previous server with this pid+generation
    /// leaked its name, or ENOSPC when `/dev/shm` is full.
    pub fn create(generation: u64, width: u32, height: u32) -> io::Result<Self> {
        let name = format!("/cenote-{}-{generation}", std::process::id());
        let map = Mapping::new(&name, segment_bytes(width, height), true)?;
        let segment = Self {
            map,
            width,
            height,
            name,
        };
        let store = |offset, value: u32| {
            segment
                .map
                .atomic_u32(offset)
                .store(value, Ordering::Relaxed);
        };
        store(header::MAGIC, MAGIC);
        store(header::LAYOUT_VERSION, LAYOUT_VERSION);
        store(header::WIDTH, width);
        store(header::HEIGHT, height);
        for (offset, value) in [
            (header::BEAUTY_OFFSET_0, beauty_offset(width, height, 0)),
            (header::DEPTH_OFFSET_0, depth_offset(width, height, 0)),
            (header::BEAUTY_OFFSET_1, beauty_offset(width, height, 1)),
            (header::DEPTH_OFFSET_1, depth_offset(width, height, 1)),
        ] {
            segment
                .map
                .atomic_u64(offset)
                .store(value, Ordering::Relaxed);
        }
        Ok(segment)
    }

    /// The in-band descriptor `Welcome`/`Resized` carry.
    #[must_use]
    pub fn desc(&self) -> FbDesc {
        FbDesc {
            shm_name: self.name.clone(),
            bytes: segment_bytes(self.width, self.height),
            width: self.width,
            height: self.height,
        }
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Publish one frame: fill the back buffer, flip it front, advance
    /// the counter (the release that lets a reader trust the pixels —
    /// [`cenote_wire::fb`]'s writer steps, verbatim). `beauty` is RGBA
    /// f32 — already linear `Rec.709`, the conversion is the caller's —
    /// and `depth` crosses as the bytes the download produced. `epoch` is
    /// the frame's session-epoch stamp (D-113), stored with the status
    /// fields so a reader sees it released by the same counter advance.
    ///
    /// # Panics
    ///
    /// If a plane's length doesn't match this segment's dimensions — the
    /// caller gates on matching sizes, so a mismatch is a bug, not data.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "plane offsets are the page plus multiples of 4 — f32-aligned by the layout"
    )]
    pub fn publish(&mut self, beauty: &[f32], depth: &[u8], samples: u32, converged: bool, epoch: u64) {
        assert_eq!(beauty.len() as u64 * 4, fb::beauty_bytes(self.width, self.height));
        assert_eq!(depth.len() as u64, fb::depth_bytes(self.width, self.height));
        let front = self.map.atomic_u32(header::FRONT_INDEX);
        let back = 1 - front.load(Ordering::Relaxed);
        // SAFETY: the offsets address the back buffer's planes, in bounds
        // by construction and f32-aligned (page + multiples of 4); the
        // reader never trusts a buffer the counter hasn't released.
        unsafe {
            std::ptr::copy_nonoverlapping(
                beauty.as_ptr(),
                self.map
                    .base
                    .add(beauty_offset(self.width, self.height, back) as usize)
                    .cast::<f32>(),
                beauty.len(),
            );
            std::ptr::copy_nonoverlapping(
                depth.as_ptr(),
                self.map
                    .base
                    .add(depth_offset(self.width, self.height, back) as usize),
                depth.len(),
            );
        }
        self.map
            .atomic_u32(header::SAMPLES)
            .store(samples, Ordering::Relaxed);
        self.map
            .atomic_u32(header::CONVERGED)
            .store(u32::from(converged), Ordering::Relaxed);
        self.map
            .atomic_u64(header::EPOCH)
            .store(epoch, Ordering::Relaxed);
        front.store(back, Ordering::Relaxed);
        // The release: everything above happens-before a reader's acquire
        // of the new count.
        self.map
            .atomic_u64(header::FRAME_COUNTER)
            .fetch_add(1, Ordering::Release);
    }

    /// Advance the monotonic rejected-edit count — the signal that tells
    /// an idle client its next `Ping` has messages waiting. Moves without
    /// a frame, which is the point.
    pub fn bump_rejected(&self) {
        self.map
            .atomic_u32(header::REJECTED_EDITS)
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// One validated, untorn frame copy out of a [`View`].
pub struct Snapshot {
    /// RGBA f32, linear `Rec.709`, row-major.
    pub beauty: Vec<f32>,
    /// f32 camera-plane depth, +∞ where every sample missed.
    pub depth: Vec<f32>,
    /// Samples accumulated into this frame.
    pub samples: u32,
    /// Whether accumulation had settled when this frame published.
    pub converged: bool,
    /// The session epoch this frame incorporates (D-113) — everything
    /// acknowledged at or below this value is in the picture, applied or
    /// rejected.
    pub epoch: u64,
    /// The frame counter at the copy — monotonic across snapshots.
    pub counter: u64,
}

/// The reader: maps an existing segment read-only and copies frames out
/// under the tear protocol. What the C++ delegate does; the integration
/// test drives this one.
pub struct View {
    map: Mapping,
    width: u32,
    height: u32,
}

impl View {
    /// Map the segment a [`FbDesc`] names and validate its header.
    ///
    /// # Errors
    ///
    /// The OS error from `shm_open`/`mmap`, or `InvalidData` when the
    /// header's magic, version, or dimensions disagree with the
    /// descriptor — a layout drift caught at map time, not as garbage
    /// pixels.
    pub fn open(desc: &FbDesc) -> io::Result<Self> {
        let map = Mapping::new(&desc.shm_name, desc.bytes, false)?;
        let view = Self {
            map,
            width: desc.width,
            height: desc.height,
        };
        let check = |what, offset, expected: u32| {
            let got = view.map.atomic_u32(offset).load(Ordering::Relaxed);
            if got == expected {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("shm header {what}: expected {expected}, found {got}"),
                ))
            }
        };
        check("magic", header::MAGIC, MAGIC)?;
        check("layout version", header::LAYOUT_VERSION, LAYOUT_VERSION)?;
        check("width", header::WIDTH, desc.width)?;
        check("height", header::HEIGHT, desc.height)?;
        Ok(view)
    }

    /// The publish count so far — 0 means no frame yet.
    #[must_use]
    pub fn frame_counter(&self) -> u64 {
        self.map
            .atomic_u64(header::FRAME_COUNTER)
            .load(Ordering::Acquire)
    }

    /// Whether accumulation has settled — the delegate's `IsConverged`.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.map.atomic_u32(header::CONVERGED).load(Ordering::Relaxed) != 0
    }

    /// The monotonic rejected-edit count.
    #[must_use]
    pub fn rejected_edits(&self) -> u32 {
        self.map
            .atomic_u32(header::REJECTED_EDITS)
            .load(Ordering::Relaxed)
    }

    /// The session epoch the front frame incorporates (D-113) — 0 before
    /// the first publish. Paired with [`Self::converged`], this is the
    /// honest convergence read: settled *and* current.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        if self.frame_counter() == 0 {
            return 0;
        }
        self.map.atomic_u64(header::EPOCH).load(Ordering::Relaxed)
    }

    /// Copy the front frame out under the tear protocol. `None` when no
    /// frame has been published yet, or when the writer advanced the
    /// counter by more than one during the copy (the copy may be torn —
    /// discard and retry).
    #[must_use]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "plane offsets are the page plus multiples of 4 — f32-aligned by the layout"
    )]
    pub fn snapshot(&self) -> Option<Snapshot> {
        let counter = self.map.atomic_u64(header::FRAME_COUNTER);
        let before = counter.load(Ordering::Acquire);
        if before == 0 {
            return None;
        }
        let front = self.map.atomic_u32(header::FRONT_INDEX).load(Ordering::Relaxed);
        let samples = self.map.atomic_u32(header::SAMPLES).load(Ordering::Relaxed);
        let converged = self.map.atomic_u32(header::CONVERGED).load(Ordering::Relaxed) != 0;
        let epoch = self.map.atomic_u64(header::EPOCH).load(Ordering::Relaxed);
        let texels = self.width as usize * self.height as usize;
        let mut beauty = vec![0.0f32; texels * 4];
        let mut depth = vec![0.0f32; texels];
        // SAFETY: in-bounds, f32-aligned plane offsets of a mapped
        // segment; a copy that raced the writer is discarded below.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.map
                    .base
                    .add(beauty_offset(self.width, self.height, front) as usize)
                    .cast::<f32>(),
                beauty.as_mut_ptr(),
                beauty.len(),
            );
            std::ptr::copy_nonoverlapping(
                self.map
                    .base
                    .add(depth_offset(self.width, self.height, front) as usize)
                    .cast::<f32>(),
                depth.as_mut_ptr(),
                depth.len(),
            );
        }
        let after = counter.load(Ordering::Acquire);
        (after - before <= 1).then_some(Snapshot {
            beauty,
            depth,
            samples,
            converged,
            epoch,
            counter: after,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writer and reader across one segment (same process — the mapping
    /// doesn't care): publish two frames and read them back untorn, with
    /// the double buffer alternating underneath.
    #[test]
    fn a_snapshot_reads_back_what_publish_wrote() {
        let (w, h) = (4, 3);
        // The test generation is offset so a concurrently running server
        // (same pid namespace, different process) can't collide.
        let mut segment = Segment::create(9_000_001, w, h).expect("create");
        let view = View::open(&segment.desc()).expect("open");
        assert!(view.snapshot().is_none(), "no frame before first publish");

        let texels = (w * h) as usize;
        let beauty: Vec<f32> = (0..texels * 4).map(|i| i as f32).collect();
        let depth_values: Vec<f32> = (0..texels).map(|i| 0.5 + i as f32).collect();
        let depth_bytes: Vec<u8> = depth_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(view.epoch(), 0, "no epoch before the first publish");
        segment.publish(&beauty, &depth_bytes, 7, false, 3);

        let first = view.snapshot().expect("first frame");
        assert_eq!(first.beauty, beauty);
        assert_eq!(first.depth, depth_values);
        assert_eq!(first.samples, 7);
        assert_eq!(first.counter, 1);
        assert!(!first.converged);
        assert_eq!(first.epoch, 3);
        assert!(!view.converged());
        assert_eq!(view.epoch(), 3);

        let brighter: Vec<f32> = beauty.iter().map(|v| v + 100.0).collect();
        segment.publish(&brighter, &depth_bytes, 8, true, 4);
        let second = view.snapshot().expect("second frame");
        assert_eq!(second.beauty, brighter);
        assert_eq!(second.samples, 8);
        assert_eq!(second.counter, 2);
        assert!(second.converged);
        assert_eq!(second.epoch, 4);
        assert!(view.converged());
        assert_eq!(view.epoch(), 4);

        assert_eq!(view.rejected_edits(), 0);
        segment.bump_rejected();
        assert_eq!(view.rejected_edits(), 1);
    }

    /// Dropping the creating side unlinks the name: a fresh open fails.
    #[test]
    fn drop_unlinks_the_name() {
        let segment = Segment::create(9_000_002, 2, 2).expect("create");
        let desc = segment.desc();
        drop(segment);
        assert!(View::open(&desc).is_err(), "the name must be gone");
    }
}
