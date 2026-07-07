//! Compiled GPU kernels: embedded SPIR-V, plus the hot-reload path.
//!
//! `build.rs` runs `slangc` over every kernel in `shaders/` and this module
//! embeds the resulting SPIR-V, so binaries are self-contained — no shader
//! files needed at run time. For interactive kernel editing, [`ShaderWatcher`]
//! wakes on source changes and [`recompile_primary`] reproduces the build-time
//! compile at run time — same binary, same flags, both `include!`ing
//! `slangc.rs` next to `build.rs` so the paths can't drift (decision D-004).
//!
//! Hot reload is a source-checkout feature: shader paths are baked from
//! `CARGO_MANIFEST_DIR` at compile time. A deployed binary renders from its
//! embedded kernels and never touches the filesystem.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use notify::Watcher as _;

use crate::error::{Error, Result};

/// The build-time `slangc` invocation, shared with `build.rs` (D-004).
mod slangc {
    include!("../slangc.rs");
}

/// SPIR-V for the primary-visibility kernel (`shaders/primary.slang`).
/// Entry point: `primary`.
///
/// A byte slice with no alignment guarantee — convert to `u32` words at
/// pipeline-creation time (e.g. `ash::util::read_spv`).
pub const PRIMARY_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/primary.spv"));

/// Entry-point name inside [`PRIMARY_SPIRV`], preserved from the Slang
/// source by `-fvk-use-entrypoint-name`.
pub const PRIMARY_ENTRY: &std::ffi::CStr = c"primary";

/// The crate's shader sources in the checkout this binary was built from.
fn shader_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

/// Recompile the primary kernel from its source and return fresh SPIR-V.
///
/// # Errors
///
/// [`Error::ShaderCompile`] with `slangc`'s diagnostics if the source doesn't
/// compile — the caller keeps rendering with its current pipeline (D-004) —
/// or [`Error::Io`] if the compiled SPIR-V can't be read back.
pub fn recompile_primary() -> Result<Vec<u8>> {
    compile(&shader_dir().join("primary.slang"))
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

    #[test]
    fn embedded_primary_kernel_is_spirv() {
        // Catches a broken/empty build-time slangc invocation in GPU-less CI.
        assert!(is_spirv(PRIMARY_SPIRV));
    }

    #[test]
    fn runtime_recompile_produces_spirv() {
        // The hot-reload path end to end, minus the file watch. Byte
        // equality with PRIMARY_SPIRV is deliberately not asserted: debug
        // info embeds source paths, which differ between the build-time
        // (relative) and runtime (absolute) invocations.
        let spirv = recompile_primary().expect("recompile primary kernel");
        assert!(is_spirv(&spirv));
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
