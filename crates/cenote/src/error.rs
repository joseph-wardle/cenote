//! The crate-wide error type (decision D-010): one coarse enum, variants
//! refined only when a caller actually matches on them. Binaries wrap this
//! in `anyhow`; panics are reserved for programmer bugs — a missing GPU or
//! a broken shader is always an `Err`.

/// Anything that can go wrong inside the core library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A Vulkan API call failed.
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] ash::vk::Result),

    /// libvulkan itself could not be loaded — no Vulkan driver on this machine.
    #[error("Vulkan loader unavailable: {0}")]
    Loader(#[from] ash::LoadingError),

    /// No physical device satisfies the ray-tracing baseline (D-015). The
    /// payload lists every enumerated device and what it lacked.
    #[error("no capable GPU found:\n{0}")]
    NoCapableGpu(String),

    /// GPU memory allocation failed.
    #[error("GPU allocation failed: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),

    /// Writing a rendered image to disk failed (encoding or I/O).
    #[error("image write failed: {0}")]
    ImageWrite(#[from] exr::error::Error),
}

/// Crate-wide result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
