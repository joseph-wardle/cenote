//! Compiled GPU kernels: the embedded SPIR-V registry, plus the hot-reload
//! path.
//!
//! `build.rs` runs `slangc` over every kernel in `shaders/` and
//! [`Kernels::embedded`] carries the results, so binaries are self-contained
//! — no shader files needed at run time. For interactive kernel editing,
//! [`ShaderWatcher`] wakes on source changes and [`Kernels::recompile`]
//! reproduces the build-time compile at run time — same flags, same kernel
//! list, both `include!`ing `slangc.rs` next to `build.rs` so nothing can
//! drift.
//!
//! Hot reload is a source-checkout feature: shader paths are baked from
//! `CARGO_MANIFEST_DIR` at compile time. A deployed binary renders from its
//! embedded kernels and never touches the filesystem.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use notify::Watcher as _;

use crate::error::{Error, Result};

/// The build-time `slangc` invocation and kernel list, shared with `build.rs`.
mod slangc {
    include!("../slangc.rs");
}

/// One compiled kernel: SPIR-V bytes (no alignment guarantee — convert to
/// `u32` words at pipeline-creation time) plus the entry-point name inside
/// them, preserved from the Slang source by `-fvk-use-entrypoint-name`.
pub struct Kernel {
    /// The compiled SPIR-V module.
    pub spirv: Vec<u8>,
    /// The entry-point name inside it — the kernel's file stem.
    pub entry: &'static CStr,
}

/// The full kernel set, one field per entry in `slangc.rs`'s `KERNELS`.
/// Both constructors produce the same set — [`Kernels::embedded`] from the
/// bytes `build.rs` baked in, [`Kernels::recompile`] fresh from the source
/// checkout — so swapping a whole set is the reload unit.
pub struct Kernels {
    /// Wave entry: camera rays, path init, ray-queue push.
    pub raygen: Kernel,
    /// Pure TLAS traversal; routes each path to hit or miss.
    pub intersect: Kernel,
    /// Escaped rays (constant sky until the HDRI lands).
    pub shade_miss: Kernel,
    /// Surface shading and path termination/continuation.
    pub shade_surface: Kernel,
    /// Occlusion tests for queued shadow rays.
    pub trace_shadow: Kernel,
    /// Film: add a wave's sample into the running sums (NaN/Inf-guarded).
    pub accumulate: Kernel,
    /// Film: sums → exposed, ACES-mapped, sRGB-encoded display RGBA8.
    pub tonemap: Kernel,
}

impl Kernels {
    /// The kernels `build.rs` compiled into this binary.
    #[must_use]
    pub fn embedded() -> Self {
        fn kernel(spirv: &[u8], entry: &'static CStr) -> Kernel {
            Kernel {
                spirv: spirv.to_vec(),
                entry,
            }
        }
        macro_rules! spirv {
            ($name:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
            };
        }
        Self {
            raygen: kernel(spirv!("raygen"), c"raygen"),
            intersect: kernel(spirv!("intersect"), c"intersect"),
            shade_miss: kernel(spirv!("shade_miss"), c"shade_miss"),
            shade_surface: kernel(spirv!("shade_surface"), c"shade_surface"),
            trace_shadow: kernel(spirv!("trace_shadow"), c"trace_shadow"),
            accumulate: kernel(spirv!("accumulate"), c"accumulate"),
            tonemap: kernel(spirv!("tonemap"), c"tonemap"),
        }
    }

    /// Recompile the whole set from the source checkout — the hot-reload
    /// path. Kernels compile in parallel (one `slangc` process each), which
    /// keeps a whole-set reload inside the sub-second dev-loop budget.
    ///
    /// # Errors
    ///
    /// [`Error::ShaderCompile`] with `slangc`'s diagnostics if any kernel's
    /// source doesn't compile — the caller keeps its current pipelines — or
    /// [`Error::Io`] if compiled SPIR-V can't be read back.
    ///
    /// # Panics
    ///
    /// Only if a compile thread itself panics — a bug, not an environment
    /// failure (compiler problems come back as errors above).
    pub fn recompile() -> Result<Self> {
        // Destructured in KERNELS order — the one place that order matters.
        let [
            raygen,
            intersect,
            shade_miss,
            shade_surface,
            trace_shadow,
            accumulate,
            tonemap,
        ] = std::thread::scope(|scope| {
            slangc::KERNELS
                .map(|name| {
                    scope.spawn(move || compile(&shader_dir().join(format!("{name}.slang"))))
                })
                .map(|handle| handle.join().expect("compile thread panicked"))
        });
        Ok(Self {
            raygen: Kernel {
                spirv: raygen?,
                entry: c"raygen",
            },
            intersect: Kernel {
                spirv: intersect?,
                entry: c"intersect",
            },
            shade_miss: Kernel {
                spirv: shade_miss?,
                entry: c"shade_miss",
            },
            shade_surface: Kernel {
                spirv: shade_surface?,
                entry: c"shade_surface",
            },
            trace_shadow: Kernel {
                spirv: trace_shadow?,
                entry: c"trace_shadow",
            },
            accumulate: Kernel {
                spirv: accumulate?,
                entry: c"accumulate",
            },
            tonemap: Kernel {
                spirv: tonemap?,
                entry: c"tonemap",
            },
        })
    }
}

/// The crate's shader sources in the checkout this binary was built from.
fn shader_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

/// Run `slangc` on `src` and return the SPIR-V bytes. The compiler writes
/// binaries to paths, not stdout, so the output round-trips through a temp
/// file — the same invocation shape as `build.rs`.
fn compile(src: &Path) -> Result<Vec<u8>> {
    let stem = src.file_stem().unwrap_or_default().to_string_lossy();
    let dst = std::env::temp_dir().join(format!("cenote-{stem}-{}.spv", std::process::id()));
    slangc::run_slangc(src, &dst).map_err(Error::ShaderCompile)?;
    let spirv = std::fs::read(&dst)?;
    // Best-effort tidy-up; a stale temp file is not worth failing a reload.
    let _ = std::fs::remove_file(&dst);
    Ok(spirv)
}

/// How long a burst of file events may go quiet before
/// [`ShaderWatcher::wait`] decides the edit is over.
const BURST_WINDOW: Duration = Duration::from_millis(50);

/// Blocks a dev loop on shader edits: watches the crate's `shaders/`
/// directory and wakes [`ShaderWatcher::wait`] when a `.slang` source
/// changes.
pub struct ShaderWatcher {
    events: mpsc::Receiver<()>,
    /// Owns the OS watch; dropping it ends the event stream.
    _watcher: notify::RecommendedWatcher,
}

impl ShaderWatcher {
    /// Start watching the crate's shader sources.
    ///
    /// # Errors
    ///
    /// [`Error::Watch`] if the OS file watch can't be established.
    pub fn new() -> Result<Self> {
        Self::watching(&shader_dir())
    }

    fn watching(dir: &Path) -> Result<Self> {
        let (tx, events) = mpsc::channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if event.as_ref().is_ok_and(is_shader_mutation) {
                    // A failed send only means nobody is waiting anymore.
                    let _ = tx.send(());
                }
            })?;
        watcher.watch(dir, notify::RecursiveMode::Recursive)?;
        Ok(Self {
            events,
            _watcher: watcher,
        })
    }

    /// Block until a shader source changes. An editor save arrives as a
    /// burst of events (write, rename, metadata); this coalesces the burst
    /// so one save means one wake-up.
    ///
    /// # Errors
    ///
    /// [`Error::Watch`] if the watcher backend shut down.
    pub fn wait(&self) -> Result<()> {
        self.events
            .recv()
            .map_err(|_| notify::Error::generic("shader watcher shut down"))?;
        while self.events.recv_timeout(BURST_WINDOW).is_ok() {}
        Ok(())
    }
}

/// True for events that change a `.slang` source: a mutating kind touching a
/// `.slang` path. Access events are deliberately excluded — recompiling
/// *reads* the source, and reacting to our own reads would loop forever.
fn is_shader_mutation(event: &notify::Event) -> bool {
    let mutates = matches!(
        event.kind,
        notify::EventKind::Create(_) | notify::EventKind::Modify(_) | notify::EventKind::Remove(_)
    );
    mutates
        && event
            .paths
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "slang"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_spirv(bytes: &[u8]) -> bool {
        // SPIR-V words are little-endian; a valid module opens with the
        // magic number.
        u32::from_le_bytes(bytes[..4].try_into().unwrap()) == 0x0723_0203
    }

    fn all(kernels: &Kernels) -> [&Kernel; 7] {
        [
            &kernels.raygen,
            &kernels.intersect,
            &kernels.shade_miss,
            &kernels.shade_surface,
            &kernels.trace_shadow,
            &kernels.accumulate,
            &kernels.tonemap,
        ]
    }

    #[test]
    fn embedded_kernels_are_spirv() {
        // Catches a broken/empty build-time slangc invocation in GPU-less CI.
        for kernel in all(&Kernels::embedded()) {
            assert!(is_spirv(&kernel.spirv), "{:?}", kernel.entry);
        }
    }

    #[test]
    fn runtime_recompile_produces_spirv() {
        // The hot-reload path end to end, minus the file watch. Byte
        // equality with the embedded set is deliberately not asserted:
        // debug info embeds source paths, which differ between the
        // build-time (relative) and runtime (absolute) invocations.
        let kernels = Kernels::recompile().expect("recompile kernel set");
        for kernel in all(&kernels) {
            assert!(is_spirv(&kernel.spirv), "{:?}", kernel.entry);
        }
    }

    #[test]
    fn broken_source_reports_diagnostics() {
        let src = std::env::temp_dir().join(format!("cenote-broken-{}.slang", std::process::id()));
        std::fs::write(&src, "this is not slang").expect("write temp source");
        let result = compile(&src);
        let _ = std::fs::remove_file(&src);
        match result {
            Err(Error::ShaderCompile(diagnostics)) => {
                assert!(!diagnostics.is_empty(), "diagnostics should not be empty");
            }
            other => panic!("expected ShaderCompile error, got {other:?}"),
        }
    }

    #[test]
    fn watcher_wakes_on_slang_edit() {
        let dir = std::env::temp_dir().join(format!("cenote-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create watch dir");
        let watcher = ShaderWatcher::watching(&dir).expect("start watcher");
        std::fs::write(dir.join("edit.slang"), "// edited").expect("write shader");
        // Poll the channel directly rather than wait(): a filter bug would
        // hang forever, a timeout fails loudly.
        let woke = watcher.events.recv_timeout(Duration::from_secs(10));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            woke.is_ok(),
            "no event within 10 s of editing a .slang file"
        );
    }

    #[test]
    fn mutation_filter_ignores_noise() {
        let modify = notify::EventKind::Modify(notify::event::ModifyKind::Any);
        let access = notify::EventKind::Access(notify::event::AccessKind::Any);

        let shader_edit = notify::Event::new(modify).add_path("a.slang".into());
        assert!(is_shader_mutation(&shader_edit));

        // Our own compile reading the source must not re-trigger it.
        let shader_read = notify::Event::new(access).add_path("a.slang".into());
        assert!(!is_shader_mutation(&shader_read));

        // Editor swap files and the like are not shader edits.
        let other_file = notify::Event::new(modify).add_path("a.slang.swp".into());
        assert!(!is_shader_mutation(&other_file));
    }
}
