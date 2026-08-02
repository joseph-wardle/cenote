//! Per-view reservoir ownership: the buffers one viewport's `ReSTIR` reuse
//! reads and writes, and the stable identity they hang off.
//!
//! Reservoirs are per-view because reuse is screen-space: each viewport
//! resolves its own history, so a Hydra delegate driving N viewports owns N
//! sets. That ownership keys off a stable *viewport* identity ([`ViewId`]),
//! deliberately NOT the `RenderInputs.generation` counter — the distinction
//! that makes the warm-start work, spelled out on `ViewId` below.
//!
//! Temporal reuse (step 5) is what makes the history *persist* across frames:
//! the [`Film`](crate::render::Film) now owns one [`ViewState`], swapping
//! `prev`/`curr` at frame end so the ping-pong carries the reservoirs across a
//! camera move — the warm-start. As of step 5b the `restir_temporal` combine
//! reads `prev` and writes `curr` from `cand` + `prev` — the four-buffer routing
//! of D-094 is now fully live; step 5c adds the reprojection substrate that rides
//! it: a per-pixel G-buffer pair (`gbuffer_prev`/`gbuffer_curr`), ping-ponged in
//! lockstep so each committed reservoir keeps the surface it was resolved for,
//! and a small `reproject` block the host rewrites each frame with the previous
//! camera. What remains dormant keeps a scoped `dead_code` allowance: the
//! [`ViewId`]-keyed [`Views`] map stays unused until a second viewport exists
//! (step 6). The allowance lifts when `Views` goes live.
#![allow(dead_code)]

use ash::vk;

use super::StoredPathReservoir;
use crate::error::Result;
use crate::gpu::{Buffer, Context, MemoryLocation, Pass};

/// An opaque, stable viewport identity — the key per-view reservoir (and, from
/// step 6, film) ownership hangs off.
///
/// Constant for the single viewer today; a Hydra delegate later mints one per
/// viewport (from its `SdfPath`). Crucially it is **not** the
/// `RenderInputs.generation` counter: generation bumps on *every* camera move,
/// but an orbit is the same view, and its reservoirs must carry across the move
/// — that carry is the warm-start. If the id tracked generation, every orbit
/// would mint a new state and drop the history, defeating the whole point. So
/// a camera move keeps the id (and the reservoirs); only a genuinely new
/// viewport mints a new id, and a resize rebuilds the state under the same id
/// (pixel correspondence breaks, so the history can't carry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewId(pub u64);

impl ViewId {
    /// The one viewport every current front-end drives.
    pub const PRIMARY: ViewId = ViewId(0);
}

/// Bytes one reservoir occupies — the [`StoredPathReservoir`] stride, the plan's
/// 96 B/pixel figure for the unified path reservoir (D-128).
const RESERVOIR_STRIDE: u64 = size_of::<StoredPathReservoir>() as u64;

/// Host layout mirror of `GBufEntry` in `shaders/restir_reproject.slang`, sizing
/// the per-pixel G-buffer below. The entry is written and read only by the shader
/// — the host never inspects a field — so this exists purely so `size_of` pins
/// the stride to the shader's layout, the way `Reproject` pins its own (never
/// instantiated; the module's `dead_code` allowance covers it).
#[repr(C)]
struct GBufEntry {
    hit: [u32; 4],       // Hit: instance + primitive + float2 barycentrics
    origin: [f32; 4],    // xyz ray origin (w unused)
    direction: [f32; 4], // xyz unit direction; w = camera-forward cosine
}

/// Bytes one G-buffer entry occupies — one per pixel — pinned to the shader's
/// `GBufEntry` layout by the mirror above rather than hand-counted.
const GBUFFER_STRIDE: u64 = size_of::<GBufEntry>() as u64;
const _: () = assert!(GBUFFER_STRIDE == 48);

/// Bytes the per-frame reprojection block occupies — the `Reproject` struct in
/// `shaders/restir_reproject.slang`, mirrored by `Reproject` in `wavefront.rs`
/// (whose `size_of` a static assert pins to this). One struct per view, not one
/// per pixel.
const REPROJECT_STRIDE: u64 = 96;

/// The four reservoir buffers one view owns (D-094's fixed-role routing), sized
/// to its current resolution. `AoS` and row-major (see the note in the parent
/// module); one per pixel.
pub struct ViewState {
    id: ViewId,
    width: u32,
    height: u32,
    /// This frame's fresh candidate reservoir, written by `restir_candidates`
    /// when temporal reuse is on — the temporal combine's canonical input. Held
    /// distinct from `curr` (next frame's `prev`) so the candidate lineage that
    /// seeds history never aliases the history buffer itself. Rebuilt each frame,
    /// never swapped.
    cand: Buffer,
    /// Last frame's committed reservoir — the temporal combine reads it as its
    /// history. Ping-pongs with `curr` at frame end.
    prev: Buffer,
    /// This frame's committed reservoir: written by the candidate stage (temporal
    /// off) or the temporal combine (temporal on), read by spatial; becomes next
    /// frame's `prev` on [`Self::swap`].
    curr: Buffer,
    /// The spatial pass's working buffer: it reads the committed prior-pass
    /// reservoir and writes here, never feeding its own output back (the
    /// determinism-preserving ping-pong the plan's §2 determinism note names).
    scratch: Buffer,
    /// Last frame's per-pixel primary hits — the G-buffer temporal reprojection
    /// reads to rebuild the previous surface for its disocclusion gate and its
    /// `p̂_i` (D-094). Ping-pongs with `gbuffer_curr` alongside `prev`/`curr`, so
    /// the surface that produced a `prev` reservoir travels with it.
    gbuffer_prev: Buffer,
    /// This frame's per-pixel primary hits, written by `restir_temporal`;
    /// becomes `gbuffer_prev` on the next [`Self::swap`].
    gbuffer_curr: Buffer,
    /// The per-frame reprojection block (`Reproject`): the previous camera basis,
    /// the two G-buffer addresses, the frame dimensions, and the reuse-gate
    /// thresholds. Host-written each frame ([`Buffer::write`]) before the wave —
    /// `CpuToGpu`, one small struct — and reached by `restir_temporal` through a
    /// single push pointer.
    reproject: Buffer,
    /// The scene build (`Scene::epoch`) `prev` was rendered against, recorded
    /// at each [`Self::swap`]. When the *current* build differs — an edit
    /// landed between frames — an indirect prev sample's `rcVertex.instance`
    /// is a raw TLAS custom index the rebuild may have renumbered, so the
    /// temporal stage drops such pairs before any dereference (the epoch
    /// gate); NEE history survives the edit through the light-id registry.
    /// Starts at 0, matching a fresh scene's — harmless either way, since a
    /// fresh state's `prev` is empty.
    prev_epoch: u64,
    /// Whether `prev` carries a committed frame yet — false on a freshly
    /// allocated state, whose reservoirs are all zero-filled, and true from the
    /// first [`Self::swap`] onward.
    ///
    /// The warm-start's precondition, made checkable (M7 step 7a). It lives here
    /// rather than on the film because it must survive a film reset, exactly as
    /// the reservoirs do: a moving camera restarts accumulation every frame onto
    /// a history that is still there. What it rules out is the genuinely cold
    /// film — a batch render's first frame, a viewport that just opened — where
    /// an estimator leaning on history would be leaning on zeros.
    warm: bool,
}

impl ViewState {
    /// Allocate a view's four reservoir buffers for a `width`×`height` frame,
    /// every reservoir initialized empty.
    pub fn new(gpu: &Context, id: ViewId, width: u32, height: u32) -> Result<Self> {
        let pixels = u64::from(width) * u64::from(height);
        let bytes = pixels * RESERVOIR_STRIDE;
        // Reuse stages address the reservoirs through buffer-reference
        // pointers, like the path pool; TRANSFER_DST is for the empty-init fill.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |name: &str| gpu.create_buffer(name, bytes, usage, MemoryLocation::GpuOnly);
        let gbuffer_bytes = pixels * GBUFFER_STRIDE;
        let gbuffer = |name: &str| gpu.create_buffer(name, gbuffer_bytes, usage, MemoryLocation::GpuOnly);
        let state = Self {
            id,
            width,
            height,
            cand: buffer("restir.reservoir.cand")?,
            prev: buffer("restir.reservoir.prev")?,
            curr: buffer("restir.reservoir.curr")?,
            scratch: buffer("restir.reservoir.scratch")?,
            gbuffer_prev: gbuffer("restir.gbuffer.prev")?,
            gbuffer_curr: gbuffer("restir.gbuffer.curr")?,
            // Host-written every frame before the wave (never through a transfer),
            // so it needs neither TRANSFER_DST nor an init fill.
            reproject: gpu.create_buffer(
                "restir.reproject",
                REPROJECT_STRIDE,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )?,
            prev_epoch: 0,
            warm: false,
        };

        // An all-zero reservoir is the empty one: sample 0, W 0, and — the
        // part that matters — confidence 0, which reads as "nothing selected".
        // So a frame-0 temporal read of `prev`, before any candidate has been
        // streamed, contributes nothing rather than interpreting uninitialized
        // VRAM as a bogus high-confidence sample. The G-buffers zero the same
        // way: their entries are only ever read where the matching `prev`
        // reservoir carries confidence, so a zeroed (never-written) entry is
        // structurally unreachable, but the fill keeps VRAM defined regardless.
        let fill = |buffer, size| Pass::Fill {
            buffer,
            offset: 0,
            size,
            value: 0,
        };
        gpu.submit_passes(&[
            fill(&state.cand, bytes),
            fill(&state.prev, bytes),
            fill(&state.curr, bytes),
            fill(&state.scratch, bytes),
            fill(&state.gbuffer_prev, gbuffer_bytes),
            fill(&state.gbuffer_curr, gbuffer_bytes),
        ])?;
        Ok(state)
    }

    /// Commit this frame's reservoir: `curr` becomes next frame's `prev`, and
    /// the matching `gbuffer_curr` becomes `gbuffer_prev` so the surface that
    /// produced each committed reservoir travels with it (the disocclusion gate
    /// compares against exactly that surface). One swap of each pair, at frame
    /// end — the ping-pong that lets the temporal pass read a fully-committed
    /// prior buffer with a barrier between, never a half-written one. `epoch`
    /// is the scene build the committed frame rendered against
    /// (`Scene::epoch`), recorded so the next frame can tell whether the
    /// history is still index-safe — see [`Self::prev_epoch`].
    pub fn swap(&mut self, epoch: u64) {
        std::mem::swap(&mut self.prev, &mut self.curr);
        std::mem::swap(&mut self.gbuffer_prev, &mut self.gbuffer_curr);
        self.prev_epoch = epoch;
        // A committed frame has landed in `prev`: from here the history is real
        // and the warm-start has something to start from.
        self.warm = true;
    }

    /// The scene build `prev` was rendered against — see the field.
    pub fn prev_epoch(&self) -> u64 {
        self.prev_epoch
    }

    /// Whether `prev` carries a committed frame — see the field.
    pub fn warm(&self) -> bool {
        self.warm
    }

    /// The view this state belongs to.
    pub fn id(&self) -> ViewId {
        self.id
    }

    /// The resolution the buffers are sized for.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Reservoirs per buffer — one per pixel.
    pub fn reservoir_count(&self) -> u32 {
        self.width * self.height
    }

    /// This frame's candidate reservoir — `restir_candidates`' write target with
    /// temporal reuse on, the temporal combine's canonical input.
    pub fn cand(&self) -> &Buffer {
        &self.cand
    }

    /// The committed prior-frame reservoir — the temporal pass's history read.
    pub fn prev(&self) -> &Buffer {
        &self.prev
    }

    /// This frame's reservoir — the candidate/reuse stages' write target, and
    /// next frame's `prev` after [`Self::swap`].
    pub fn curr(&self) -> &Buffer {
        &self.curr
    }

    /// The spatial pass's working buffer — it reads the committed prior-pass
    /// reservoir and writes here, never feeding its own output back.
    pub fn scratch(&self) -> &Buffer {
        &self.scratch
    }

    /// Last frame's per-pixel G-buffer — temporal reprojection reads it at the
    /// reprojected pixel to rebuild the previous surface.
    pub fn gbuffer_prev(&self) -> &Buffer {
        &self.gbuffer_prev
    }

    /// This frame's per-pixel G-buffer — `restir_temporal` writes each hit's
    /// surface here; it becomes `gbuffer_prev` on the next [`Self::swap`].
    pub fn gbuffer_curr(&self) -> &Buffer {
        &self.gbuffer_curr
    }

    /// The per-frame reprojection block, for the host to rewrite each frame
    /// before the wave (the previous camera basis and the current G-buffer
    /// addresses), then hand to `restir_temporal` by address.
    pub fn reproject(&self) -> &Buffer {
        &self.reproject
    }

    /// The reprojection block as mutable, for [`Buffer::write`].
    pub fn reproject_mut(&mut self) -> &mut Buffer {
        &mut self.reproject
    }
}

/// The per-view reservoir states, keyed by [`ViewId`]. One entry today (the
/// single viewport), but keyed so N viewports own N states without reshaping
/// this — the multi-view seam, laid but not yet machinery.
///
/// [`Self::for_view`] encodes the whole lifecycle: a camera move (same id, same
/// size) returns the existing state untouched, so the reservoirs carry; a
/// resize (same id, different size) rebuilds it, dropping the now-meaningless
/// history; a new id mints a fresh state. Note what `for_view` does *not* take:
/// the generation counter — the carry-across-move behaviour is structural, not
/// a runtime check.
#[derive(Default)]
pub struct Views {
    states: std::collections::BTreeMap<ViewId, ViewState>,
}

impl Views {
    /// An empty set — no views realized yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The view's reservoir state at the current resolution, (re)building it if
    /// it is absent or was sized for a different resolution. Returns the
    /// existing state untouched when the size matches — the carry-across-move.
    pub fn for_view(
        &mut self,
        gpu: &Context,
        id: ViewId,
        width: u32,
        height: u32,
    ) -> Result<&mut ViewState> {
        let stale = self
            .states
            .get(&id)
            .is_none_or(|state| state.dimensions() != (width, height));
        if stale {
            self.states
                .insert(id, ViewState::new(gpu, id, width, height)?);
        }
        Ok(self
            .states
            .get_mut(&id)
            .expect("the view was just inserted or already present"))
    }

    /// Forget a view's reservoirs — a viewport closed. A no-op if unknown.
    pub fn remove(&mut self, id: ViewId) {
        self.states.remove(&id);
    }

    /// How many views hold reservoir state.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether any view holds reservoir state.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{ViewId, Views};

    /// T4: the per-view lifecycle. A camera move carries the reservoirs (same
    /// buffers survive), a resize rebuilds them (dropping the history pixel
    /// correspondence just broke), and a second viewport owns a wholly
    /// separate set — all without the state ever seeing a generation counter.
    #[test]
    fn view_state_carries_across_moves_and_rebuilds_on_resize() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut views = Views::new();

        // First frame: the primary view realizes at 128×72.
        let addresses = {
            let state = views
                .for_view(&gpu, ViewId::PRIMARY, 128, 72)
                .expect("allocate view");
            assert_eq!(state.dimensions(), (128, 72));
            assert_eq!(state.reservoir_count(), 128 * 72);
            (
                state.cand().device_address(),
                state.prev().device_address(),
                state.curr().device_address(),
                state.scratch().device_address(),
                state.gbuffer_prev().device_address(),
                state.gbuffer_curr().device_address(),
                state.reproject().device_address(),
            )
        };
        assert_eq!(views.len(), 1);

        // A camera move is the same view at the same size: the very same
        // buffers come back (their addresses are unchanged), so the reservoir
        // history carries across the move — the warm-start.
        {
            let state = views
                .for_view(&gpu, ViewId::PRIMARY, 128, 72)
                .expect("same-size lookup");
            assert_eq!(
                (
                    state.cand().device_address(),
                    state.prev().device_address(),
                    state.curr().device_address(),
                    state.scratch().device_address(),
                    state.gbuffer_prev().device_address(),
                    state.gbuffer_curr().device_address(),
                    state.reproject().device_address()
                ),
                addresses,
                "a camera move must not reallocate the reservoirs or G-buffers"
            );
        }
        assert_eq!(views.len(), 1, "a move mints no new view");

        // A resize is the same id but a new size: the state rebuilds, because
        // pixel-to-reservoir correspondence no longer holds.
        {
            let state = views
                .for_view(&gpu, ViewId::PRIMARY, 200, 100)
                .expect("resize");
            assert_eq!(state.dimensions(), (200, 100));
            assert_eq!(state.reservoir_count(), 200 * 100);
        }
        assert_eq!(views.len(), 1, "a resize rebuilds in place, not alongside");

        // A second viewport owns a wholly separate set of reservoirs.
        let second = ViewId(1);
        let second_curr = {
            let state = views.for_view(&gpu, second, 128, 72).expect("second view");
            state.curr().device_address()
        };
        assert_eq!(views.len(), 2);
        // The primary view (now 200×100) is untouched and distinct.
        let primary_curr = views
            .for_view(&gpu, ViewId::PRIMARY, 200, 100)
            .expect("primary intact")
            .curr()
            .device_address();
        assert_ne!(
            primary_curr, second_curr,
            "two views must not share a reservoir buffer"
        );

        // Closing the second viewport forgets exactly its state.
        views.remove(second);
        assert_eq!(views.len(), 1);
        assert!(!views.is_empty());
    }

    /// The ping-pong swap exchanges prev and curr — nothing else moves — and
    /// records the scene build the committed frame rendered against (the
    /// temporal epoch gate's host half, §4c decision 2).
    #[test]
    fn swap_exchanges_prev_and_curr() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut views = Views::new();
        let state = views
            .for_view(&gpu, ViewId::PRIMARY, 64, 64)
            .expect("allocate");
        let (cand, prev, curr, scratch, gprev, gcurr, reproject) = (
            state.cand().device_address(),
            state.prev().device_address(),
            state.curr().device_address(),
            state.scratch().device_address(),
            state.gbuffer_prev().device_address(),
            state.gbuffer_curr().device_address(),
            state.reproject().device_address(),
        );
        // A fresh state's history matches build 0 — vacuously safe, its
        // `prev` is empty.
        assert_eq!(state.prev_epoch(), 0, "a fresh state starts at build 0");
        state.swap(3);
        assert_eq!(state.prev().device_address(), curr, "prev takes curr's buffer");
        assert_eq!(state.curr().device_address(), prev, "curr takes prev's buffer");
        // The G-buffer pair ping-pongs in lockstep with the reservoirs, so a
        // committed reservoir's surface stays paired with it.
        assert_eq!(state.gbuffer_prev().device_address(), gcurr, "gbuffer_prev takes gbuffer_curr");
        assert_eq!(state.gbuffer_curr().device_address(), gprev, "gbuffer_curr takes gbuffer_prev");
        assert_eq!(state.cand().device_address(), cand, "cand is untouched");
        assert_eq!(state.scratch().device_address(), scratch, "scratch is untouched");
        assert_eq!(state.reproject().device_address(), reproject, "reproject is not swapped");
        // The committed frame's scene build travels with the history: the
        // next frame compares it against the current `Scene::epoch` to decide
        // whether indirect samples may cross the boundary.
        assert_eq!(state.prev_epoch(), 3, "swap records the committed build");
        state.swap(3);
        assert_eq!(state.prev_epoch(), 3, "an editless frame keeps the build");
        state.swap(7);
        assert_eq!(state.prev_epoch(), 7, "an edit's new build replaces it");
    }
}
