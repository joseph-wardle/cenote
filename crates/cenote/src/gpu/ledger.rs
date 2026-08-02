//! Where the device memory went, counted as it is allocated.
//!
//! Every allocation already carries a name for the allocator's own
//! bookkeeping, and those names already follow a dotted convention
//! (`scene.geometry`, `film.albedo.sum`, `wavefront.queue.ray`). The
//! ledger reads the bucket off those leading
//! segments, so an allocation is counted correctly the moment it is *named*
//! correctly, and there is no second table to keep in sync with the first.
//!
//! The trade is that the naming convention became load-bearing rather than
//! decorative, and adopting it caught a group sitting in the wrong place:
//! the mesh buffers carried no prefix at all (`{mesh}.vertices`) and would
//! have counted as scratch. They were renamed rather than special-cased,
//! which is the point — see [`Bucket::of`] for the distinction that needs
//! explaining.
//!
//! One relaxed atomic add per allocation and one subtract per free is not a
//! measurable cost against creating and destroying a Vulkan buffer, which
//! is why this can be tier one — see [`crate::stats`]. Reading it is five
//! relaxed loads.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::gpu::Context;
use crate::stats::Memory;

/// Which of [`Memory`]'s four buckets an allocation belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Bucket {
    Scene,
    Film,
    Textures,
    Scratch,
}

impl Bucket {
    /// The bucket an allocation's name puts it in.
    ///
    /// Matched on whole dotted segments, most specific first: the image
    /// data under `scene.texture.*` and `scene.environment` is what a
    /// person means by "textures", while the environment's *sampling*
    /// tables (`scene.env.*`) are scene data and stay with the geometry.
    /// Segment matching is what keeps those two apart — a plain string
    /// prefix would swallow `scene.environment` into `scene.env`.
    ///
    /// Anything unrecognized is scratch. That is the honest default: an
    /// uncounted allocation would quietly make the total a lie, and scratch
    /// is the bucket whose unexplained growth we most want to see. Staging
    /// buffers on their way to or from the host (`download.*`,
    /// `present.transfer`) land there too, deliberately: they carry the
    /// film, they are not it.
    ///
    /// Every prefix here is one the renderer really allocates under; a
    /// speculative arm would read as coverage the ledger does not have.
    fn of(name: &str) -> Self {
        let mut segments = name.split('.');
        // Test fixtures allocate under `test.`; nobody reads a stats line
        // during a unit test, so they fall through to scratch with
        // everything else unclaimed.
        match (segments.next().unwrap_or_default(), segments.next()) {
            ("scene", Some("texture" | "environment")) => Bucket::Textures,
            ("scene" | "accel", _) => Bucket::Scene,
            ("film" | "session" | "tonemap", _) => Bucket::Film,
            _ => Bucket::Scratch,
        }
    }
}

/// Live bytes per bucket, shared by every allocation-owning resource.
///
/// Relaxed ordering throughout: these counters are read for display, never
/// to order anything, and a reader that catches a load mid-flight sees a
/// number that was true a microsecond ago. Paying for stronger ordering to
/// make a memory readout exact would be paying for the wrong thing.
#[derive(Default)]
pub(super) struct Ledger {
    scene: AtomicU64,
    film: AtomicU64,
    textures: AtomicU64,
    scratch: AtomicU64,
}

impl Ledger {
    fn counter(&self, bucket: Bucket) -> &AtomicU64 {
        match bucket {
            Bucket::Scene => &self.scene,
            Bucket::Film => &self.film,
            Bucket::Textures => &self.textures,
            Bucket::Scratch => &self.scratch,
        }
    }

    /// Count `bytes` against the bucket `name` selects, returning it so the
    /// resource can hand the same bucket back on drop.
    pub(super) fn add(&self, name: &str, bytes: u64) -> Bucket {
        let bucket = Bucket::of(name);
        self.counter(bucket).fetch_add(bytes, Ordering::Relaxed);
        bucket
    }

    /// Give `bytes` back, from a resource's `Drop`.
    pub(super) fn remove(&self, bucket: Bucket, bytes: u64) {
        self.counter(bucket).fetch_sub(bytes, Ordering::Relaxed);
    }

    fn read(&self, budget: Option<u64>) -> Memory {
        Memory {
            scene: self.scene.load(Ordering::Relaxed),
            film: self.film.load(Ordering::Relaxed),
            textures: self.textures.load(Ordering::Relaxed),
            scratch: self.scratch.load(Ordering::Relaxed),
            budget,
        }
    }
}

impl Context {
    /// A clone of the shared ledger handle, for resources that count
    /// themselves in and out of it.
    pub(super) fn ledger_handle(&self) -> Arc<Ledger> {
        Arc::clone(&self.ledger)
    }

    /// Live device memory, by bucket — the tier-one memory read.
    ///
    /// Sums what the renderer has allocated, not what the driver has
    /// reserved: the difference is the driver's own overhead, which is not
    /// ours to report. The budget beside it is the device-local heap, which
    /// is.
    #[must_use]
    pub fn memory(&self) -> Memory {
        self.ledger.read(self.device_local_heap)
    }
}

/// The largest device-local heap `physical_device` reports, or `None` where
/// there is no device-local memory to speak of (a shared-memory integrated
/// part, where the question has no useful answer).
///
/// Read once at bring-up and kept on the [`Context`]: it is a constant of
/// the device, and asking the driver on every read would put a round-trip
/// inside the five relaxed loads this module promises.
pub(super) fn device_local_heap(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u64> {
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    properties.memory_heaps[..properties.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bucket is the allocation's name, and the mapping has to survive
    /// the two names that could plausibly go either way.
    ///
    /// Every name here is one the renderer really allocates — a test over
    /// invented names would pass while the convention drifted underneath
    /// it, which is the one failure a naming-based ledger has.
    #[test]
    fn the_name_picks_the_bucket() {
        assert_eq!(Bucket::of("scene.geometry"), Bucket::Scene);
        assert_eq!(Bucket::of("scene.tlas"), Bucket::Scene);
        assert_eq!(Bucket::of("accel.scratch"), Bucket::Scene);
        assert_eq!(Bucket::of("film.sum"), Bucket::Film);
        assert_eq!(Bucket::of("film.albedo.sum"), Bucket::Film);
        // Image data is textures...
        assert_eq!(Bucket::of("scene.texture.brass_albedo.png"), Bucket::Textures);
        assert_eq!(Bucket::of("scene.environment"), Bucket::Textures);
        // ...but not the params table beside them, whose name only *looks*
        // like the texture prefix.
        assert_eq!(Bucket::of("scene.texture_params"), Bucket::Scene);
        // Nor the environment's sampling tables, which are scene data — and
        // only whole-segment matching tells `scene.env` from
        // `scene.environment`.
        assert_eq!(Bucket::of("scene.env.marginal"), Bucket::Scene);
        assert_eq!(Bucket::of("wavefront.queue.ray"), Bucket::Scratch);
        // Staging on its way to or from the host carries the film without
        // being it, so it is scratch by decision, not by oversight.
        assert_eq!(Bucket::of("download.staging"), Bucket::Scratch);
        assert_eq!(Bucket::of("present.transfer"), Bucket::Scratch);
        // Unrecognized never silently vanishes from the total.
        assert_eq!(Bucket::of("something.new"), Bucket::Scratch);
    }

    /// Bytes come back out again — the property that makes a bucket a live
    /// cost rather than a high-water mark of everything ever allocated.
    #[test]
    fn allocations_are_counted_out_as_well_as_in() {
        let ledger = Ledger::default();
        let bucket = ledger.add("film.beauty.sum", 1024);
        assert_eq!(ledger.read(None).film, 1024);
        ledger.add("film.albedo.sum", 512);
        assert_eq!(ledger.read(None).total(), 1536);
        ledger.remove(bucket, 1024);
        assert_eq!(ledger.read(None).film, 512);
    }
}
