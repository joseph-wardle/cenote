//! Acceleration-structure builds: a BLAS per triangle mesh, one TLAS over
//! placed instances.
//!
//! Creation and building are two steps, split at the one seam that matters:
//! a structure's storage, handle, and *device address* are all valid the
//! moment it is created, and only the traversal data has to wait for a
//! build. So [`Context::create_structure`] hands back a finished
//! [`AccelerationStructure`] plus a [`BuildJob`] describing the work still
//! owed, and the caller decides when to record it —
//! [`Context::build_blas`] immediately, on its own blocking submit;
//! [`Upload`](super::Upload) batched with a chunk's worth of others. The
//! shape is otherwise deliberately minimal — no compaction, no refits —
//! because every structure is built exactly once; rebuilds only become
//! interesting with dynamic scenes, which nothing on the roadmap requires.

use std::slice;

use ash::vk;
use glam::Mat4;

use crate::error::Result;
use crate::gpu::{Buffer, Context, MemoryLocation};

/// One build policy for every structure Cenote makes: trace fast, build
/// once. Shared by the size query and the recording, which Vulkan requires
/// to agree.
const BUILD_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;

/// A built BLAS or TLAS, destroyed on drop (before its [`Context`], like
/// every `gpu` resource).
pub struct AccelerationStructure {
    handle: vk::AccelerationStructureKHR,
    /// The structure's own device address — what TLAS instances reference.
    address: vk::DeviceAddress,
    /// Backing storage, kept alive exactly as long as the handle.
    _buffer: Buffer,
    loader: ash::khr::acceleration_structure::Device,
}

impl AccelerationStructure {
    /// The raw handle, for the TLAS descriptor write in
    /// [`Context::dispatch`] — the one binding that isn't a device address.
    /// Stays inside `gpu`: the quarantine boundary.
    pub(super) fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}

/// The build a freshly created [`AccelerationStructure`] still owes, and
/// the scratch memory it will need to pay it.
///
/// Holds no borrows: the geometry it describes is a pair of device
/// addresses, so a job can outlive the buffers' Rust bindings and sit in a
/// queue until someone records it. What it *does* hold is a raw handle to
/// the structure it fills — whose owner must therefore still be alive when
/// the recording happens — and the scratch buffer, whose lifetime is
/// exactly this job's.
pub(super) struct BuildJob {
    ty: vk::AccelerationStructureTypeKHR,
    geometry: vk::AccelerationStructureGeometryKHR<'static>,
    primitive_count: u32,
    dst: vk::AccelerationStructureKHR,
    scratch: Buffer,
    /// The alignment-corrected address inside `scratch`.
    scratch_address: vk::DeviceAddress,
}

impl BuildJob {
    /// Bytes of scratch this job holds — what a batching caller budgets
    /// against, beside its staging.
    pub(super) fn scratch_size(&self) -> vk::DeviceSize {
        self.scratch.size()
    }
}

/// Record `jobs` into `cmd` as one batched build. Per-job scratch means no
/// barriers between them: nothing two builds touch is shared, so the driver
/// is free to run them together.
///
/// The caller owes the usual guarantee that the geometry buffers are
/// readable by the time this executes — within a submission that is a
/// barrier, across one the fence.
pub(super) fn record_builds(
    loader: &ash::khr::acceleration_structure::Device,
    cmd: vk::CommandBuffer,
    jobs: &[BuildJob],
) {
    let infos: Vec<_> = jobs
        .iter()
        .map(|job| {
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(job.ty)
                .flags(BUILD_FLAGS)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(slice::from_ref(&job.geometry))
                .dst_acceleration_structure(job.dst)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: job.scratch_address,
                })
        })
        .collect();
    let ranges: Vec<_> = jobs
        .iter()
        .map(|job| {
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(job.primitive_count)
        })
        .collect();
    // One single-element range list per build info, as the entry point's
    // array-of-arrays shape wants.
    let per_build: Vec<&[_]> = ranges.iter().map(slice::from_ref).collect();
    unsafe { loader.cmd_build_acceleration_structures(cmd, &infos, &per_build) };
}

/// The triangle geometry a BLAS is built over: tightly packed `[f32; 3]`
/// positions in `vertices`, `u32` index triples in `indices`. Both buffers
/// need `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR` and
/// `SHADER_DEVICE_ADDRESS` usage.
///
/// `'static` is not a widening: the returned struct holds two device
/// addresses and no pointers, so it borrows the buffers only for the
/// duration of this call.
pub(super) fn triangle_geometry(
    vertices: &Buffer,
    vertex_count: u32,
    indices: &Buffer,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    assert!(vertex_count > 0, "cannot build a BLAS over an empty mesh");
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertices.device_address(),
        })
        .vertex_stride(3 * size_of::<f32>() as vk::DeviceSize)
        .max_vertex(vertex_count - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: indices.device_address(),
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        // No any-hit logic anywhere on the roadmap; alpha
        // testing revisits this flag.
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

/// The device's minimum alignment for build scratch addresses, read once at
/// bring-up and kept on the [`Context`].
///
/// It is a constant of the device, and the query behind it is a full
/// `vkGetPhysicalDeviceProperties2` round trip — asked per build it cost
/// one of those for every mesh in the scene.
pub(super) fn scratch_alignment(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::DeviceSize {
    let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_props);
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props) };
    vk::DeviceSize::from(accel_props.min_acceleration_structure_scratch_offset_alignment)
}

/// One entry in a TLAS: a BLAS placed in the world.
pub struct TlasInstance<'a> {
    /// The mesh being instanced.
    pub blas: &'a AccelerationStructure,
    /// Object-to-world transform. Must be affine — the bottom row is
    /// discarded (Vulkan instance transforms are 3×4).
    pub transform: Mat4,
    /// 24-bit value the kernel reads back from a hit as the instance's
    /// custom index; Cenote uses it to index the geometry lookup table.
    pub custom_index: u32,
    /// Visibility mask: a ray traversal sees this instance only when its
    /// own 8-bit mask ANDs nonzero with this one. `0xFF` is visible to
    /// every ray; the scene clears bits for per-ray-type visibility
    /// (today, camera-invisible instances).
    pub mask: u8,
    /// Whether traversal commits this instance's hits on its own. The
    /// scene clears it for fractional-opacity materials, whose crossings
    /// surface as candidates for the kernels' stochastic pass-through and
    /// shadow attenuation.
    pub opaque: bool,
}

impl Context {
    /// Build a BLAS over `triangle_count` triangles — tightly packed
    /// `[f32; 3]` positions in `vertices`, `u32` index triples in
    /// `indices`, both needing
    /// `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR` and
    /// `SHADER_DEVICE_ADDRESS` usage — on its own blocking submit.
    ///
    /// One mesh at a time is the wrong unit for loading a scene — see
    /// [`Upload`](super::Upload), which batches this — but the right one
    /// for the handful of callers that build a single structure and want it
    /// traceable when the call returns.
    pub fn build_blas(
        &self,
        name: &str,
        vertices: &Buffer,
        vertex_count: u32,
        indices: &Buffer,
        triangle_count: u32,
    ) -> Result<AccelerationStructure> {
        assert!(triangle_count > 0, "cannot build a BLAS over an empty mesh");
        self.build_structure(
            name,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            triangle_geometry(vertices, vertex_count, indices),
            triangle_count,
        )
    }

    /// Build a TLAS over `instances`. Blocks until the build completes,
    /// so the instance staging buffer is transient.
    pub fn build_tlas(
        &self,
        name: &str,
        instances: &[TlasInstance<'_>],
    ) -> Result<AccelerationStructure> {
        let mut raw: Vec<vk::AccelerationStructureInstanceKHR> =
            instances.iter().map(raw_instance).collect();
        // Vulkan forbids empty buffers, so a zero-instance TLAS (an empty
        // scene — every ray misses) stages one zeroed instance the build
        // never reads: the primitive count below stays 0.
        if raw.is_empty() {
            raw.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: [0.0; 12] },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: 0,
                },
            });
        }
        // A plain-old-data FFI struct; viewing it as bytes for upload is sound.
        let bytes =
            unsafe { slice::from_raw_parts(raw.as_ptr().cast::<u8>(), size_of_val(&raw[..])) };
        let instance_buffer = self.upload_buffer(
            &format!("{name}.instances"),
            bytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let instance_data = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
            vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(),
            },
        );
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instance_data,
            });
        self.build_structure(
            name,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            geometry,
            instances.len() as u32,
        )
    }

    /// Create a structure and record its build on one blocking submit —
    /// [`Context::create_structure`] and its [`BuildJob`], immediately.
    fn build_structure(
        &self,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'static>,
        primitive_count: u32,
    ) -> Result<AccelerationStructure> {
        let (structure, job) = self.create_structure(name, ty, geometry, primitive_count)?;
        // On failure `structure` drops, destroying the handle: the funnel
        // the two-step split replaced is now just ownership.
        self.submit_once(|_, cmd| {
            record_builds(&self.accel_loader, cmd, slice::from_ref(&job));
        })?;
        Ok(structure)
    }

    /// Create a structure's storage and handle, and allocate the scratch
    /// its build will need: everything except the recording.
    ///
    /// The returned [`AccelerationStructure`] is already complete in the
    /// only sense the host cares about — it owns its handle and storage and
    /// its device address is final — so it can be handed to a caller, put in
    /// a scene, and referenced by a TLAS build while its own [`BuildJob`]
    /// waits in a queue. What it is not yet is *traceable*: the job has to
    /// run first.
    pub(super) fn create_structure(
        &self,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'static>,
        primitive_count: u32,
    ) -> Result<(AccelerationStructure, BuildJob)> {
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(slice::from_ref(&geometry));
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[primitive_count],
                &mut sizes,
            );
        }

        let buffer = self.create_buffer(
            name,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.handle())
            .size(sizes.acceleration_structure_size)
            .ty(ty);
        let handle = unsafe {
            self.accel_loader
                .create_acceleration_structure(&create_info, None)?
        };

        // The one span where a handle is live and unowned: if scratch can't
        // be allocated, destroy it here.
        match self.scratch_for(sizes.build_scratch_size) {
            Ok((scratch, scratch_address)) => {
                let info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle);
                let address = unsafe {
                    self.accel_loader
                        .get_acceleration_structure_device_address(&info)
                };
                Ok((
                    AccelerationStructure {
                        handle,
                        address,
                        _buffer: buffer,
                        loader: self.accel_loader.clone(),
                    },
                    BuildJob {
                        ty,
                        geometry,
                        primitive_count,
                        dst: handle,
                        scratch,
                        scratch_address,
                    },
                ))
            }
            Err(err) => {
                unsafe {
                    self.accel_loader
                        .destroy_acceleration_structure(handle, None);
                }
                Err(err)
            }
        }
    }

    /// A scratch buffer of at least `scratch_size` usable bytes, and the
    /// address inside it a build may start from. The scratch *address* must
    /// honor the device's alignment minimum, which plain buffer alignment
    /// doesn't guarantee — over-allocate and round up.
    fn scratch_for(&self, scratch_size: vk::DeviceSize) -> Result<(Buffer, vk::DeviceAddress)> {
        let alignment = self.scratch_alignment;
        let scratch = self.create_buffer(
            "accel.scratch",
            scratch_size + alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;
        let address = scratch.device_address().next_multiple_of(alignment);
        Ok((scratch, address))
    }
}

fn raw_instance(instance: &TlasInstance<'_>) -> vk::AccelerationStructureInstanceKHR {
    // Strictly below the all-ones value: the kernels' packed path state
    // reserves 0xffffff as its "no medium" sentinel.
    assert!(
        instance.custom_index < (1 << 24) - 1,
        "instance custom_index must fit below the 24-bit sentinel"
    );
    // Vulkan wants row-major 3×4 (rotation | translation): transpose the
    // column-major glam matrix and keep the first three rows.
    let rows = instance.transform.transpose().to_cols_array();
    let matrix: [f32; 12] = rows[..12].try_into().expect("3x4 of a 4x4");
    vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR { matrix },
        instance_custom_index_and_mask: vk::Packed24_8::new(instance.custom_index, instance.mask),
        // No culling: the kernel flips geometric normals toward the ray, so
        // both faces of everything are hittable. Non-opaque overrides the
        // BLAS's baked opaque flag, surfacing this instance's hits as
        // candidates.
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, {
            let mut flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE;
            if !instance.opaque {
                flags |= vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            }
            flags.as_raw() as u8
        }),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: instance.blas.address,
        },
    }
}
