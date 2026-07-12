//! Per-view reservoir ownership: the buffers one viewport's `ReSTIR` reuse
//! reads and writes, and the stable identity they hang off.
//!
//! Reservoirs are per-view because reuse is screen-space: each viewport
//! resolves its own history, so a Hydra delegate driving N viewports owns N
//! sets. That ownership keys off a stable *viewport* identity ([`ViewId`]),
//! deliberately NOT the `RenderInputs.generation` counter — the distinction
//! that makes the warm-start work, spelled out on `ViewId` below.
//!
//! This substrate lands before its consumer. Single-frame initial RIS (step
//! 3.2) needs only one reservoir per pixel and hands the wavefront a bare
//! per-wave target; it is temporal reuse (step 5) that first makes the history
//! *persist* across frames, and only then do the prev/curr ping-pong, the
//! carry-across-move, and per-view ownership below earn their keep. Until then
//! the state is exercised by its lifecycle test (T4), so the module keeps one
//! scoped `dead_code` allowance — removed when the render loop owns per-view
//! reservoirs for the warm-start.
#![allow(dead_code)]

use ash::vk;

use super::StoredReservoir;
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

/// Bytes one reservoir occupies — the [`StoredReservoir`] stride, the plan's
/// 24 B/pixel figure.
const RESERVOIR_STRIDE: u64 = size_of::<StoredReservoir>() as u64;

/// The three reservoir buffers one view owns, sized to its current resolution.
/// `AoS` and row-major (see the note in the parent module); one per pixel.
pub struct ViewState {
    id: ViewId,
    width: u32,
    height: u32,
    /// Last frame's committed reservoir — temporal/spatial reuse reads it.
    /// Ping-pongs with `curr` at frame end.
    prev: Buffer,
    /// This frame's reservoir, written by the candidate and reuse stages and
    /// committed at frame end; becomes next frame's `prev` on [`Self::swap`].
    curr: Buffer,
    /// The spatial pass's working buffer: it reads the committed prior-pass
    /// reservoir and writes here, never feeding its own output back (the
    /// determinism-preserving ping-pong the plan's §2 determinism note names).
    scratch: Buffer,
}

impl ViewState {
    /// Allocate a view's three reservoir buffers for a `width`×`height` frame,
    /// every reservoir initialized empty.
    pub fn new(gpu: &Context, id: ViewId, width: u32, height: u32) -> Result<Self> {
        let bytes = u64::from(width) * u64::from(height) * RESERVOIR_STRIDE;
        // Reuse stages address the reservoirs through buffer-reference
        // pointers, like the path pool; TRANSFER_DST is for the empty-init fill.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |name: &str| gpu.create_buffer(name, bytes, usage, MemoryLocation::GpuOnly);
        let state = Self {
            id,
            width,
            height,
            prev: buffer("restir.reservoir.prev")?,
            curr: buffer("restir.reservoir.curr")?,
            scratch: buffer("restir.reservoir.scratch")?,
        };

        // An all-zero reservoir is the empty one: sample 0, W 0, and — the
        // part that matters — confidence 0, which reads as "nothing selected".
        // So a frame-0 temporal read of `prev`, before any candidate has been
        // streamed, contributes nothing rather than interpreting uninitialized
        // VRAM as a bogus high-confidence sample.
        let fill = |buffer| Pass::Fill {
            buffer,
            offset: 0,
            size: bytes,
            value: 0,
        };
        gpu.submit_passes(&[fill(&state.prev), fill(&state.curr), fill(&state.scratch)])?;
        Ok(state)
    }

    /// Commit this frame's reservoir: `curr` becomes next frame's `prev`. One
    /// swap, at frame end — the ping-pong that lets the temporal pass read a
    /// fully-committed prior buffer with a barrier between, never a
    /// half-written one.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.curr);
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

    /// The committed prior-frame reservoir's GPU address (temporal read).
    pub fn prev_address(&self) -> vk::DeviceAddress {
        self.prev.device_address()
    }

    /// This frame's reservoir's GPU address (candidate/reuse write).
    pub fn curr_address(&self) -> vk::DeviceAddress {
        self.curr.device_address()
    }

    /// The spatial pass's working buffer's GPU address.
    pub fn scratch_address(&self) -> vk::DeviceAddress {
        self.scratch.device_address()
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
                state.prev_address(),
                state.curr_address(),
                state.scratch_address(),
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
                    state.prev_address(),
                    state.curr_address(),
                    state.scratch_address()
                ),
                addresses,
                "a camera move must not reallocate the reservoirs"
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
            state.curr_address()
        };
        assert_eq!(views.len(), 2);
        // The primary view (now 200×100) is untouched and distinct.
        let primary_curr = views
            .for_view(&gpu, ViewId::PRIMARY, 200, 100)
            .expect("primary intact")
            .curr_address();
        assert_ne!(
            primary_curr, second_curr,
            "two views must not share a reservoir buffer"
        );

        // Closing the second viewport forgets exactly its state.
        views.remove(second);
        assert_eq!(views.len(), 1);
        assert!(!views.is_empty());
    }

    /// The ping-pong swap exchanges prev and curr and nothing else.
    #[test]
    fn swap_exchanges_prev_and_curr() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        let mut views = Views::new();
        let state = views
            .for_view(&gpu, ViewId::PRIMARY, 64, 64)
            .expect("allocate");
        let (prev, curr, scratch) = (
            state.prev_address(),
            state.curr_address(),
            state.scratch_address(),
        );
        state.swap();
        assert_eq!(state.prev_address(), curr, "prev takes curr's buffer");
        assert_eq!(state.curr_address(), prev, "curr takes prev's buffer");
        assert_eq!(state.scratch_address(), scratch, "scratch is untouched");
    }
}
