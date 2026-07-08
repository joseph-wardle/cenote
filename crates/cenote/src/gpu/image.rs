//! Sampled images — the environment map's home. Buffers travel as device
//! addresses in push constants, but filtered texture reads need real
//! `VkImage`s behind descriptors, so this is the one image the binding
//! model carries (set 0, binding 1, next to the TLAS) until M2's bindless
//! texture table.

use std::mem::ManuallyDrop;
use std::slice;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::{Error, Result};
use crate::gpu::{Context, MemoryLocation};

/// A 2D RGBA `f32` image with a view and its sampler, ready to bind for
/// filtered shader reads; freed on drop (before the [`Context`], like every
/// `gpu` resource). The sampler is equirect-shaped: bilinear, wrapping
/// horizontally (azimuth is periodic) and clamping vertically (the poles
/// are edges, not seams).
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
        match self.allocator.lock() {
            Ok(mut allocator) => {
                if let Err(err) = allocator.free(allocation) {
                    log::error!("failed to free image allocation: {err}");
                }
            }
            Err(_) => log::error!("allocator mutex poisoned — leaking image allocation"),
        }
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
        let device = self.device();

        let features = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device(), FORMAT)
        }
        .optimal_tiling_features;
        if !features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        ) {
            return Err(Error::NoCapableGpu(
                "  selected device: cannot linearly filter RGBA32F sampled images\n".into(),
            ));
        }

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
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
        match self.finish_sampled_image(name, image, width, height, texels) {
            Ok(sampled) => Ok(sampled),
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                Err(err)
            }
        }
    }

    fn finish_sampled_image(
        &self,
        name: &str,
        image: vk::Image,
        width: u32,
        height: u32,
        texels: &[f32],
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
            .format(FORMAT)
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
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        sampled.sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        self.upload_texels(&sampled, width, height, texels)?;
        Ok(sampled)
    }

    /// Stage `texels` into the image: transition to transfer target, copy,
    /// transition to its permanent shader-read layout.
    fn upload_texels(
        &self,
        sampled: &SampledImage,
        width: u32,
        height: u32,
        texels: &[f32],
    ) -> Result<()> {
        let staging = self.staging_buffer("image.staging", bytemuck::cast_slice(texels))?;
        self.submit_once(|device, cmd| {
            barrier(
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
            barrier(
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

/// One full-image layout transition between the given stage/access scopes.
fn barrier(
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
