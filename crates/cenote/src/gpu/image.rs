//! Sampled images. Buffers travel as device addresses in push constants,
//! but filtered texture reads need real `VkImage`s behind descriptors, so
//! images are what the binding model carries: the environment map (set 0,
//! binding 1, next to the TLAS) and the material textures in the bindless
//! table (binding 2). Two upload paths, one per producer:
//! [`Context::upload_sampled_image`] for the environment's RGBA `f32`
//! radiance, [`Context::upload_texture`] for the BC blocks `texture.rs`
//! prepares.

use std::mem::ManuallyDrop;
use std::slice;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::{Error, Result};
use crate::gpu::buffer::free_allocation;
use crate::gpu::{Context, MemoryLocation};

/// A 2D image with a view and its sampler, ready to bind for filtered
/// shader reads; freed on drop (before the [`Context`], like every `gpu`
/// resource). Always bilinear; the address modes are the upload path's —
/// equirect-shaped for the environment (wrapping azimuth, clamped poles),
/// wrapping both ways for material textures.
pub struct SampledImage {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: ManuallyDrop<Allocation>,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
}

impl SampledImage {
    /// The view and sampler, in the shape descriptor writes want.
    pub(crate) fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.view)
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }
}

impl Drop for SampledImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
        }
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        free_allocation(&self.allocator, allocation, "image");
    }
}

/// Texel format: full `f32` — the environment is radiance data (an
/// unclipped sun overflows `f16`), and one small image doesn't earn a
/// precision compromise.
const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

impl Context {
    /// Create a `width`×`height` [`SampledImage`] holding `texels` (tightly
    /// packed row-major RGBA `f32`, row 0 on top), moved through a transient
    /// staging buffer and transitioned for shader reads.
    ///
    /// # Errors
    ///
    /// [`crate::Error::NoCapableGpu`] if the device can't filter this format
    /// (linear filtering of `f32` textures is universal on ray-tracing
    /// hardware but formally optional, so it's checked, not assumed); other
    /// [`crate::Error`]s from creation, allocation, or the copy submission.
    ///
    /// # Panics
    ///
    /// On zero dimensions or a texel slice that doesn't match them —
    /// programmer bugs.
    pub fn upload_sampled_image(
        &self,
        name: &str,
        width: u32,
        height: u32,
        texels: &[f32],
    ) -> Result<SampledImage> {
        assert!(width > 0 && height > 0, "zero-sized image");
        assert_eq!(
            texels.len() as u64,
            u64::from(width) * u64::from(height) * 4,
            "texel count doesn't match image dimensions"
        );
        self.upload_image(
            name,
            (width, height),
            FORMAT,
            // Equirect addressing: azimuth is periodic, the poles are
            // edges, not seams.
            (
                vk::SamplerAddressMode::REPEAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ),
            bytemuck::cast_slice(texels),
            (width, height),
        )
    }

    /// Create a [`SampledImage`] holding one level of a prepped material
    /// texture: BC blocks over block-padded rows (`padded` names the
    /// texel extent the rows actually span), wrapping bilinear sampler —
    /// texture coordinates tile.
    ///
    /// # Errors
    ///
    /// As [`Context::upload_sampled_image`].
    ///
    /// # Panics
    ///
    /// On zero dimensions or data that doesn't match them — programmer
    /// bugs (prep validated the cache before handing it over).
    pub fn upload_texture(
        &self,
        name: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        data: &[u8],
    ) -> Result<SampledImage> {
        let padded = (width.next_multiple_of(4), height.next_multiple_of(4));
        assert_eq!(
            data.len() as u64,
            u64::from(padded.0 / 4) * u64::from(padded.1 / 4) * crate::texture::block_size(format),
            "block data doesn't match the texture's padded dimensions"
        );
        self.upload_image(
            name,
            (width, height),
            format,
            (
                vk::SamplerAddressMode::REPEAT,
                vk::SamplerAddressMode::REPEAT,
            ),
            data,
            padded,
        )
    }

    /// The shared upload: create the image, check the device can filter
    /// its format, allocate, build view and sampler, stage the bytes.
    /// `row_extent` is the texel span of the source rows — the image's own
    /// size for tightly packed data, the block-padded size for BC data.
    fn upload_image(
        &self,
        name: &str,
        (width, height): (u32, u32),
        format: vk::Format,
        address_modes: (vk::SamplerAddressMode, vk::SamplerAddressMode),
        data: &[u8],
        row_extent: (u32, u32),
    ) -> Result<SampledImage> {
        assert!(width > 0 && height > 0, "zero-sized image");
        let device = self.device();

        let features = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device(), format)
        }
        .optimal_tiling_features;
        if !features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        ) {
            return Err(Error::NoCapableGpu(format!(
                "  selected device: cannot linearly filter {format:?} sampled images\n"
            )));
        }

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };

        // Everything after image creation funnels failures through one
        // cleanup that unwinds exactly what was built.
        match self.finish_sampled_image(
            name,
            image,
            (width, height),
            format,
            address_modes,
            data,
            row_extent,
        ) {
            Ok(sampled) => Ok(sampled),
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                Err(err)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the private back half of upload_image, split only for its cleanup seam"
    )]
    fn finish_sampled_image(
        &self,
        name: &str,
        image: vk::Image,
        (width, height): (u32, u32),
        format: vk::Format,
        address_modes: (vk::SamplerAddressMode, vk::SamplerAddressMode),
        data: &[u8],
        row_extent: (u32, u32),
    ) -> Result<SampledImage> {
        let device = self.device();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = self
            .allocator_handle()
            .lock()
            .expect("allocator mutex poisoned")
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        // From here the allocation belongs to the SampledImage-in-progress;
        // build it now so its Drop is the single cleanup path.
        let mut sampled = SampledImage {
            image,
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            allocation: ManuallyDrop::new(allocation),
            device: device.clone(),
            allocator: self.allocator_handle(),
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        sampled.view = unsafe { device.create_image_view(&view_info, None)? };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(address_modes.0)
            .address_mode_v(address_modes.1)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        sampled.sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        self.upload_texels(&sampled, (width, height), data, row_extent)?;
        Ok(sampled)
    }

    /// Stage `data` into the image: transition to transfer target, copy,
    /// transition to its permanent shader-read layout. `row_extent` is the
    /// texel span of the staged rows; when it exceeds the image (BC data's
    /// block padding), the copy skips the padding on its way in.
    fn upload_texels(
        &self,
        sampled: &SampledImage,
        (width, height): (u32, u32),
        data: &[u8],
        row_extent: (u32, u32),
    ) -> Result<()> {
        let staging = self.staging_buffer("image.staging", data)?;
        self.submit_once(|device, cmd| {
            image_barrier(
                device,
                cmd,
                sampled.image,
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
                (
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            );
            let region = vk::BufferImageCopy::default()
                .buffer_row_length(if row_extent.0 == width {
                    0
                } else {
                    row_extent.0
                })
                .buffer_image_height(if row_extent.1 == height {
                    0
                } else {
                    row_extent.1
                })
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.handle(),
                    sampled.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&region),
                );
            }
            image_barrier(
                device,
                cmd,
                sampled.image,
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ),
                (
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                ),
            );
        })
    }
}

/// One full-image layout transition, `(old, new)` layouts ordered between
/// the `(stage, access)` source and destination scopes. Shared with
/// `present.rs`, the other module recording image transitions.
pub(super) fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
    (src_stage, src_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
    (dst_stage, dst_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let info = vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
}
