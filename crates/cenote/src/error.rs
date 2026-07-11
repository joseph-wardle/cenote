//! The crate-wide error type: one coarse enum, variants refined only when a
//! caller actually matches on them. Binaries wrap this in `anyhow`; panics
//! are reserved for programmer bugs — a missing GPU or a broken shader is
//! always an `Err`.

/// Anything that can go wrong inside the core library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A Vulkan API call failed.
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] ash::vk::Result),

    /// libvulkan itself could not be loaded — no Vulkan driver on this machine.
    #[error("Vulkan loader unavailable: {0}")]
    Loader(#[from] ash::LoadingError),

    /// No physical device satisfies the ray-tracing baseline. The payload
    /// lists every enumerated device and what it lacked.
    #[error("no capable GPU found:\n{0}")]
    NoCapableGpu(String),

    /// GPU memory allocation failed.
    #[error("GPU allocation failed: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),

    /// Writing or reading an EXR failed (encoding, decoding, or I/O).
    #[error("EXR I/O failed: {0}")]
    Image(#[from] exr::error::Error),

    /// `slangc` rejected a kernel during hot reload — or couldn't be run at
    /// all. The payload is the compiler's diagnostics; the caller keeps its
    /// last good pipeline.
    #[error("shader compile failed:\n{0}")]
    ShaderCompile(String),

    /// The egui overlay renderer failed (pipeline creation, texture upload,
    /// or draw recording).
    #[error("UI overlay failed: {0}")]
    Overlay(#[from] egui_ash_renderer::RendererError),

    /// A filesystem operation failed (e.g. reading hot-reloaded SPIR-V).
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),

    /// The shader-source watcher couldn't start, or its backend shut down.
    #[error("shader watch failed: {0}")]
    Watch(#[from] notify::Error),

    /// A change-set failed validation and was not applied — the scene
    /// description is untouched. The payload names the offending object.
    #[error("change-set rejected: {0}")]
    Scene(String),

    /// A scene file couldn't be understood: not RON, an unknown field
    /// (typos fail loudly, never silently no-op), or a format version this
    /// build doesn't read.
    #[error("scene file rejected: {0}")]
    SceneFormat(String),

    /// The render thread panicked. Its own errors are ordinary `Err`s that
    /// travel back through the join; this is the fallback for an actual
    /// panic — an assertion or `unwrap` on that thread — carrying whatever
    /// message it left, so the fault surfaces on the main thread instead of
    /// vanishing with the thread.
    #[error("render thread panicked: {0}")]
    RenderThreadPanicked(String),

    /// `OpenImageDenoise` refused the filter setup or failed mid-run. The
    /// payload is OIDN's own diagnostic.
    #[cfg(feature = "denoise")]
    #[error("denoise failed: {0}")]
    Denoise(String),
}

/// Crate-wide result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
