//! Acceleration-structure builds: a BLAS per triangle mesh, one TLAS over
//! placed instances.
//!
//! Builds run as blocking one-shot submits, so scratch memory and
//! the instance staging buffer live only for the call. The shape is
//! deliberately minimal — no compaction, no refits — because every
//! structure is built exactly once; rebuilds only become interesting with
//! dynamic scenes, which nothing on the roadmap requires.

use std::slice;

use ash::vk;
use glam::Mat4;

use crate::error::Result;
use crate::gpu::{Buffer, Context, MemoryLocation};

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
    /// Build a BLAS over `triangle_count` triangles: tightly packed
    /// `[f32; 3]` positions in `vertices`, `u32` index triples in `indices`.
    /// Both buffers need `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR`
    /// and `SHADER_DEVICE_ADDRESS` usage.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from allocation or the build submission.
    ///
    /// # Panics
    ///
    /// On an empty mesh — a programmer bug.
    pub fn build_blas(
        &self,
        name: &str,
        vertices: &Buffer,
        vertex_count: u32,
        indices: &Buffer,
        triangle_count: u32,
    ) -> Result<AccelerationStructure> {
        assert!(
            vertex_count > 0 && triangle_count > 0,
            "cannot build a BLAS over an empty mesh"
        );
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
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            // No any-hit logic anywhere on the roadmap; alpha
            // testing (M2 textures) revisits this flag.
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        self.build_structure(
            name,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &geometry,
            triangle_count,
        )
    }

    /// Build a TLAS over `instances`. Blocks until the build completes,
    /// so the instance staging buffer is transient.
    ///
    /// # Errors
    ///
    /// Any [`crate::Error`] from allocation or the build submission.
    ///
    /// # Panics
    ///
    /// If an instance's `custom_index` exceeds 24 bits — Vulkan would
    /// silently truncate it.
    pub fn build_tlas(
        &self,
        name: &str,
        instances: &[TlasInstance<'_>],
    ) -> Result<AccelerationStructure> {
        let raw: Vec<vk::AccelerationStructureInstanceKHR> =
            instances.iter().map(raw_instance).collect();
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
            &geometry,
            instances.len() as u32,
        )
    }

    /// The BLAS/TLAS-common tail: size query, storage + handle creation,
    /// scratch allocation, blocking build, address query.
    fn build_structure(
        &self,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: &vk::AccelerationStructureGeometryKHR<'_>,
        primitive_count: u32,
    ) -> Result<AccelerationStructure> {
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(slice::from_ref(geometry));
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

        // From here every failure must destroy the handle: funnel through
        // one exit point.
        match self.run_build(
            &mut build_info,
            handle,
            sizes.build_scratch_size,
            primitive_count,
        ) {
            Ok(address) => Ok(AccelerationStructure {
                handle,
                address,
                _buffer: buffer,
                loader: self.accel_loader.clone(),
            }),
            Err(err) => {
                unsafe {
                    self.accel_loader
                        .destroy_acceleration_structure(handle, None);
                }
                Err(err)
            }
        }
    }

    fn run_build(
        &self,
        build_info: &mut vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
        dst: vk::AccelerationStructureKHR,
        scratch_size: vk::DeviceSize,
        primitive_count: u32,
    ) -> Result<vk::DeviceAddress> {
        // The scratch *address* must honor the device's alignment minimum,
        // which plain buffer alignment doesn't guarantee — over-allocate and
        // round up.
        let alignment = self.scratch_alignment();
        let scratch = self.create_buffer(
            "accel.scratch",
            scratch_size + alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;
        *build_info =
            build_info
                .dst_acceleration_structure(dst)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch.device_address().next_multiple_of(alignment),
                });

        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
        self.submit_once(|_, cmd| unsafe {
            self.accel_loader.cmd_build_acceleration_structures(
                cmd,
                slice::from_ref(build_info),
                &[slice::from_ref(&range)],
            );
        })?;

        let info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(dst);
        Ok(unsafe {
            self.accel_loader
                .get_acceleration_structure_device_address(&info)
        })
    }

    fn scratch_alignment(&self) -> vk::DeviceSize {
        let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_props);
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut props);
        }
        vk::DeviceSize::from(accel_props.min_acceleration_structure_scratch_offset_alignment)
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
