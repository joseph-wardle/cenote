//! Texture prep: the one path from an image file to a GPU-ready block of
//! compressed texels. [`prepare`] runs decode → mip-cap downscale → BC
//! encode, and caches the result as a DDS written next to the source
//! (Cycles' `blender_tx` pattern) so a scene's second prep skips
//! everything but a hash check. Uploading the result and indexing it from
//! the bindless table are scene prep's job (`scene/prep.rs`).
//!
//! One BC level per texture, hardware bilinear, no mip chain: jittered
//! accumulation integrates the pixel footprint, so converged output is
//! unbiased — ray-cone mip selection is a bandwidth optimization that
//! waits for the measured performance pass. The mip-cap bounds VRAM
//! instead: anything larger than [`MAX_DIM`] is box-downscaled (in linear
//! light) before encoding.
//!
//! Formats by usage: color is BC7 (an sRGB view keeps the storage in
//! source space and lets the hardware decode it), HDR float color is
//! BC6H, scalar masks are BC4 baked from their chosen source channel
//! (red when unstated), and tangent-space
//! normals are BC5 on x/y with z reconstructed in-shader. Color textures
//! stay `Rec.709` on disk and in VRAM; the shader applies the working-
//! space conversion after the hardware's sRGB decode.
//!
//! Cache invalidation is by content: the DDS header's reserved words
//! carry an FNV-1a hash of the source bytes and the prep parameters, so
//! an edited image or a policy change re-encodes while git checkouts and
//! touched mtimes stay hits.

use std::fs;
use std::path::{Path, PathBuf};

use ash::vk;

use crate::error::{Error, Result};

/// Textures larger than this on either axis are box-downscaled until they
/// fit — the mip-cap that keeps an everything-resident renderer's VRAM
/// bounded no matter what an asset ships.
pub(crate) const MAX_DIM: u32 = 4096;

/// Bumping this invalidates every DDS cache — the knob to turn when the
/// prep pipeline itself changes (filters, encoder settings, layout).
const PIPELINE_VERSION: u32 = 1;

/// What a texture feeds, which decides its channel count, encoding, and
/// default color space. Part of the cache identity: one source image used
/// as both color and mask becomes two prepared textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Usage {
    /// RGB color (base color, emission): BC7. 8-bit sources default to
    /// sRGB storage with hardware decode; float sources are linear BC6H.
    Color,
    /// A single scalar from one source channel (roughness, metalness,
    /// opacity): BC4, linear.
    Scalar,
    /// Three channels that are data rather than color — a per-channel
    /// mean free path, say. The same BC7/BC6H encode [`Usage::Color`]
    /// takes, because the storage question is identical, but linear by
    /// default like a scalar mask, and sampled without the working-space
    /// conversion: a length has no color space to convert *from*.
    Vector,
    /// A tangent-space normal map: BC5 on x/y, always linear.
    Normal,
}

impl Usage {
    /// The usage class as it appears in the DDS cache path and content
    /// hash, so a color and a data use of one image cache separately.
    fn tag(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Scalar => "scalar",
            Self::Vector => "vector",
            Self::Normal => "normal",
        }
    }
}

/// The source channel a scalar prep bakes. BC4 stores exactly one
/// component, so the choice is prep-time identity rather than a runtime
/// swizzle: a packed image (say occlusion/roughness/metalness) preps once
/// per channel used. Only [`Usage::Scalar`] reads it — color and normal
/// encodes take fixed channels, and their callers pass [`Channel::R`] so
/// a stray selector can't fork their cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Channel {
    R,
    G,
    B,
    A,
}

impl Channel {
    /// The channel's index into an RGBA texel.
    fn index(self) -> usize {
        match self {
            Self::R => 0,
            Self::G => 1,
            Self::B => 2,
            Self::A => 3,
        }
    }

    /// The channel as it appears in the DDS cache path and content hash.
    /// Red is the empty tag on purpose: it spells exactly what prep did
    /// before channels existed, so every cache written then stays a hit
    /// instead of becoming an orphan beside its source.
    fn tag(self) -> &'static str {
        match self {
            Self::R => "",
            Self::G => ".g",
            Self::B => ".b",
            Self::A => ".a",
        }
    }
}

/// What one scene-file texture reference asks of prep — and therefore the
/// identity of a prepared texture: two materials sharing a source image
/// *and* its interpretation share one bindless slot, while a color and a
/// mask use of the same file are two (as are two channels of one mask).
/// Sample-time [`Params`] fork the key too — they ride the bindless index,
/// so two transforms of one image are two slots (the image preps and
/// uploads twice; the prep cache below is keyed on content alone, so only
/// the upload duplicates).
pub(crate) type Key = (std::path::PathBuf, Usage, Option<bool>, Channel, Params);

/// The sample-time parameters a texture reference carries beside its image:
/// the affine UV remap and the value multiplier
/// ([`description::TextureRef`](crate::scene::description::TextureRef)'s
/// `uv` and `scale`, identity defaults filled in). Stored as float bits so
/// the key stays `Ord`; NaN can't arise (lowering rejects nothing here, but
/// serde never parses NaN from RON) and bit-equality is exactly the
/// don't-fork-the-cache rule anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Params {
    uv_scale: [u32; 2],
    uv_offset: [u32; 2],
    scale: u32,
}

impl Params {
    /// Pack the sample-time parameters, identity for `None`.
    pub(crate) fn new(uv: Option<([f32; 2], [f32; 2])>, scale: Option<f32>) -> Self {
        let (uv_scale, uv_offset) = uv.unwrap_or(([1.0; 2], [0.0; 2]));
        Self {
            uv_scale: uv_scale.map(f32::to_bits),
            uv_offset: uv_offset.map(f32::to_bits),
            scale: scale.unwrap_or(1.0).to_bits(),
        }
    }

    /// The parameters as the floats the GPU record carries:
    /// `(uv_scale, uv_offset, value_scale)`.
    pub(crate) fn unpack(self) -> ([f32; 2], [f32; 2], f32) {
        (
            self.uv_scale.map(f32::from_bits),
            self.uv_offset.map(f32::from_bits),
            f32::from_bits(self.scale),
        )
    }
}

/// One texture, prepped and ready to upload: single-level BC blocks over
/// block-padded rows (dimensions rounded up to multiples of 4 by edge
/// replication), with `width`/`height` the true texel size the sampler
/// addresses. `hash` identifies the source content and prep parameters —
/// residency compares it to decide whether an edit actually changed the
/// image.
pub(crate) struct Prepared {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: vk::Format,
    pub(crate) data: Vec<u8>,
    pub(crate) hash: u64,
}

/// Prep `path` for `usage`, through the DDS cache beside the source.
/// `srgb` is the scene file's explicit color-space override; `None`
/// derives it from the usage and the source's bit depth (8-bit color is
/// sRGB, float color is linear, data is linear — pbrt's rule). `channel`
/// is the scalar bake's source component; color and normal usages read
/// fixed channels and take [`Channel::R`].
///
/// # Errors
///
/// [`Error::Scene`] when the file can't be read or doesn't decode — bad
/// texture data must not end a live session. A cache that fails to
/// *write* only warns: the render proceeds from memory.
pub(crate) fn prepare(
    path: &Path,
    usage: Usage,
    srgb: Option<bool>,
    channel: Channel,
) -> Result<Prepared> {
    let bytes = fs::read(path).map_err(|error| {
        Error::Scene(format!(
            "texture \"{}\": can't read: {error}",
            path.display()
        ))
    })?;
    let hash = cache_hash(&bytes, usage, srgb, channel);
    let cache = cache_path(path, usage, srgb, channel);
    if let Some(prepared) = read_cache(&cache, hash) {
        return Ok(prepared);
    }
    let prepared = encode(path, &bytes, usage, srgb, channel, hash)?;
    if let Err(error) = fs::write(&cache, dds::compose(&prepared, hash)) {
        log::warn!(
            "texture \"{}\": couldn't write its cache \"{}\": {error}",
            path.display(),
            cache.display()
        );
    }
    Ok(prepared)
}

/// The cache lives next to its source, named by source name and usage —
/// `wood.png` used as color caches as `wood.png.color.dds`, its green
/// channel as a mask as `wood.png.scalar.g.dds`. An explicit color-space
/// override changes the prepared bytes, so it joins the name.
fn cache_path(path: &Path, usage: Usage, srgb: Option<bool>, channel: Channel) -> PathBuf {
    let space = match srgb {
        None => "",
        Some(true) => ".srgb",
        Some(false) => ".linear",
    };
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}{}{space}.dds", usage.tag(), channel.tag()));
    path.with_file_name(name)
}

/// The cache-validity hash: source bytes plus everything that changes the
/// prepared result. The channel folds in as its tag — empty for red, so
/// pre-channel caches keep validating.
fn cache_hash(bytes: &[u8], usage: Usage, srgb: Option<bool>, channel: Channel) -> u64 {
    finish_hash(
        Stamp {
            fnv: fnv1a(0xcbf2_9ce4_8422_2325, bytes),
            radiance: bytes.starts_with(b"#?"),
        },
        usage,
        srgb,
        channel,
    )
}

/// A source file's content identity: its byte hash plus the decode-routing
/// sniff, the two inputs [`finish_hash`] needs beyond the reference's own
/// parameters.
#[derive(Clone, Copy)]
struct Stamp {
    fnv: u64,
    /// Radiance sources take the float decode path; the sniff is part of
    /// cache identity so the pre-float caches of exactly those files
    /// retire without invalidating anyone else's.
    radiance: bool,
}

/// Fold the reference's parameters over a source stamp — the tail of
/// [`cache_hash`], split out so [`expected_hash`] can reuse a memoized
/// stamp without re-reading the file.
fn finish_hash(stamp: Stamp, usage: Usage, srgb: Option<bool>, channel: Channel) -> u64 {
    let mut hash = stamp.fnv;
    hash = fnv1a(hash, usage.tag().as_bytes());
    hash = fnv1a(hash, channel.tag().as_bytes());
    if stamp.radiance {
        hash = fnv1a(hash, b"radiance-float");
    }
    hash = fnv1a(
        hash,
        &[
            PIPELINE_VERSION as u8,
            match srgb {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            },
        ],
    );
    fnv1a(hash, &MAX_DIM.to_le_bytes())
}

/// The hash [`prepare`] would stamp on this reference, computed without
/// reading the file when its size and mtime match the last look — a stat
/// instead of a full read + hash per edit tick, the same trade the VDB
/// prep cache makes (an in-place edit preserving both goes unseen until
/// either moves). Residency compares this against the resident hash to
/// skip prep entirely for an unchanged source.
pub(crate) fn expected_hash(
    path: &Path,
    usage: Usage,
    srgb: Option<bool>,
    channel: Channel,
) -> Result<u64> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::SystemTime;

    /// Per-source memo: (size, mtime) validates the recorded stamp.
    type Stamps = HashMap<PathBuf, (u64, SystemTime, Stamp)>;
    static STAMPS: Mutex<Option<Stamps>> = Mutex::new(None);

    let meta = fs::metadata(path).map_err(|error| {
        Error::Scene(format!(
            "texture \"{}\": can't read: {error}",
            path.display()
        ))
    })?;
    let (size, mtime) = (meta.len(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH));
    let mut stamps = STAMPS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let stamps = stamps.get_or_insert_default();
    let stamp = match stamps.get(path) {
        Some(&(seen_size, seen_mtime, stamp)) if (seen_size, seen_mtime) == (size, mtime) => stamp,
        _ => {
            let bytes = fs::read(path).map_err(|error| {
                Error::Scene(format!(
                    "texture \"{}\": can't read: {error}",
                    path.display()
                ))
            })?;
            let stamp = Stamp {
                fnv: fnv1a(0xcbf2_9ce4_8422_2325, &bytes),
                radiance: bytes.starts_with(b"#?"),
            };
            stamps.insert(path.to_path_buf(), (size, mtime, stamp));
            stamp
        }
    };
    Ok(finish_hash(stamp, usage, srgb, channel))
}

/// FNV-1a over `bytes`, continuing from `state` — the one hash both
/// prep caches (DDS and `.nvdb` sidecar) fold with. Not cryptographic,
/// and doesn't need to be: a collision costs one stale cache the user
/// can delete, never correctness of committed data.
pub(crate) fn fnv1a(state: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(state, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A cached prep, iff the cache exists, parses, and carries `hash` — any
/// mismatch (edited source, older pipeline, truncation) reads as a miss
/// and the encode below overwrites it.
fn read_cache(cache: &Path, hash: u64) -> Option<Prepared> {
    dds::parse(&fs::read(cache).ok()?, hash)
}

// -- Decode → downscale → encode -------------------------------------------

/// A decoded source image, kept in its storage representation: 8-bit
/// stays 8-bit (so an image within the mip cap encodes its exact source
/// texels), float stays float.
enum Source {
    Bytes {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    Float {
        rgba: Vec<f32>,
        width: u32,
        height: u32,
    },
}

fn decode(path: &Path, bytes: &[u8]) -> Result<Source> {
    // EXR is the float source (the format this crate already speaks);
    // everything else goes through `image`.
    if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        let (width, height, rgba) = crate::output::read_exr_bytes(bytes).map_err(|error| {
            Error::Scene(format!(
                "texture \"{}\": doesn't decode as an EXR: {error}",
                path.display()
            ))
        })?;
        return Ok(Source::Float {
            rgba,
            width,
            height,
        });
    }
    let image = image::load_from_memory(bytes)
        .or_else(|sniff_error| {
            // TGA carries no magic bytes, so sniffing can't recognize it;
            // when the content doesn't identify itself, the extension gets
            // a say before the sniff error stands.
            match path
                .extension()
                .and_then(|ext| ext.to_str())
                .and_then(image::ImageFormat::from_extension)
            {
                Some(format) => image::load_from_memory_with_format(bytes, format),
                None => Err(sniff_error),
            }
        })
        .map_err(|error| {
            Error::Scene(format!(
                "texture \"{}\": doesn't decode: {error}",
                path.display()
            ))
        })?;
    // A float-decoded source (Radiance HDR) keeps its range — quantizing
    // through `to_rgba8` would clip it to LDR and apply an sRGB view it
    // never had.
    if matches!(
        image,
        image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
    ) {
        let rgba = image.to_rgba32f();
        let (width, height) = rgba.dimensions();
        return Ok(Source::Float {
            rgba: rgba.into_raw(),
            width,
            height,
        });
    }
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(Source::Bytes {
        rgba: rgba.into_raw(),
        width,
        height,
    })
}

fn encode(
    path: &Path,
    bytes: &[u8],
    usage: Usage,
    srgb: Option<bool>,
    channel: Channel,
    hash: u64,
) -> Result<Prepared> {
    let source = decode(path, bytes)?;
    let (Source::Bytes { width, height, .. } | Source::Float { width, height, .. }) = &source;
    if *width == 0 || *height == 0 {
        return Err(Error::Scene(format!(
            "texture \"{}\": zero-sized image",
            path.display()
        )));
    }
    let mut prepared = match usage {
        Usage::Color => encode_color(source, srgb),
        Usage::Scalar => encode_scalar(source, srgb, channel),
        // A color's encode, a scalar's default: linear unless the
        // reference says sRGB outright.
        Usage::Vector => encode_color(source, Some(srgb.unwrap_or(false))),
        Usage::Normal => encode_normal(source),
    };
    prepared.hash = hash;
    Ok(prepared)
}

/// Color: 8-bit sources encode BC7 with the color space as the *view*
/// (sRGB decode is the sampler's job, so storage keeps the source's
/// quantization); float sources encode linear BC6H, where an sRGB
/// override has no meaning and is ignored.
fn encode_color(source: Source, srgb: Option<bool>) -> Prepared {
    match source {
        Source::Bytes {
            rgba,
            width,
            height,
        } => {
            let srgb = srgb.unwrap_or(true);
            let (rgba, width, height) = mip_cap_bytes(rgba, width, height, srgb, false);
            let (padded, pw, ph) = pad_to_blocks(&rgba, width, height, 4);
            let surface = intel_tex_2::RgbaSurface {
                data: &padded,
                width: pw,
                height: ph,
                stride: pw * 4,
            };
            Prepared {
                width,
                height,
                format: if srgb {
                    vk::Format::BC7_SRGB_BLOCK
                } else {
                    vk::Format::BC7_UNORM_BLOCK
                },
                data: intel_tex_2::bc7::compress_blocks(
                    &intel_tex_2::bc7::opaque_basic_settings(),
                    &surface,
                ),
                hash: 0, // encode() stamps the real hash
            }
        }
        Source::Float {
            rgba,
            width,
            height,
        } => {
            let (rgba, width, height) = mip_cap_floats(rgba, width, height, false);
            // BC6H is unsigned halves: negative radiance clamps to zero.
            let halves: Vec<u16> = rgba.iter().map(|&value| half_bits(value)).collect();
            let (padded, pw, ph) = pad_to_blocks(&halves, width, height, 4);
            let surface = intel_tex_2::RgbaSurface {
                data: bytemuck::cast_slice(&padded),
                width: pw,
                height: ph,
                stride: pw * 8,
            };
            Prepared {
                width,
                height,
                format: vk::Format::BC6H_UFLOAT_BLOCK,
                data: intel_tex_2::bc6h::compress_blocks(
                    &intel_tex_2::bc6h::basic_settings(),
                    &surface,
                ),
                hash: 0,
            }
        }
    }
}

/// Scalar masks: one source channel, BC4. Data defaults to linear; an
/// explicit sRGB override linearizes at prep (BC4 has no sRGB view).
fn encode_scalar(source: Source, srgb: Option<bool>, channel: Channel) -> Prepared {
    let srgb = srgb == Some(true);
    let picked = channel.index();
    let (mut mask, width, height): (Vec<u8>, u32, u32) = match source {
        Source::Bytes {
            rgba,
            width,
            height,
        } => {
            let (rgba, width, height) = mip_cap_bytes(rgba, width, height, srgb, false);
            (
                rgba.chunks_exact(4).map(|texel| texel[picked]).collect(),
                width,
                height,
            )
        }
        Source::Float {
            rgba,
            width,
            height,
        } => {
            let (rgba, width, height) = mip_cap_floats(rgba, width, height, false);
            (
                rgba.chunks_exact(4)
                    .map(|texel| quantize(texel[picked]))
                    .collect(),
                width,
                height,
            )
        }
    };
    if srgb {
        for value in &mut mask {
            *value = quantize(srgb_to_linear(f32::from(*value) / 255.0));
        }
    }
    let (padded, pw, ph) = pad_to_blocks(&mask, width, height, 1);
    let surface = intel_tex_2::RSurface {
        data: &padded,
        width: pw,
        height: ph,
        stride: pw,
    };
    Prepared {
        width,
        height,
        format: vk::Format::BC4_UNORM_BLOCK,
        data: intel_tex_2::bc4::compress_blocks(&surface),
        hash: 0,
    }
}

/// Tangent-space normals: x/y as BC5 (the shader reconstructs z), always
/// linear. Downscaling renormalizes, so averaged texels stay directions.
fn encode_normal(source: Source) -> Prepared {
    let (rgba, width, height): (Vec<u8>, u32, u32) = match source {
        Source::Bytes {
            rgba,
            width,
            height,
        } => mip_cap_bytes(rgba, width, height, false, true),
        Source::Float {
            rgba,
            width,
            height,
        } => {
            let (rgba, width, height) = mip_cap_floats(rgba, width, height, true);
            (
                rgba.iter().map(|&value| quantize(value)).collect(),
                width,
                height,
            )
        }
    };
    let xy: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|texel| [texel[0], texel[1]])
        .collect();
    let (padded, pw, ph) = pad_to_blocks(&xy, width, height, 2);
    let surface = intel_tex_2::RgSurface {
        data: &padded,
        width: pw,
        height: ph,
        stride: pw * 2,
    };
    Prepared {
        width,
        height,
        format: vk::Format::BC5_UNORM_BLOCK,
        data: intel_tex_2::bc5::compress_blocks(&surface),
        hash: 0,
    }
}

// -- Downscale ---------------------------------------------------------------

/// Mip-cap an 8-bit RGBA image. Within the cap it returns the exact
/// source texels; past it, texels convert to f32 (through the sRGB
/// transfer when the storage is sRGB — filtering happens in linear
/// light), box-halve until they fit, and re-quantize.
fn mip_cap_bytes(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    srgb: bool,
    normals: bool,
) -> (Vec<u8>, u32, u32) {
    if width.max(height) <= MAX_DIM {
        return (rgba, width, height);
    }
    let floats: Vec<f32> = rgba
        .iter()
        .map(|&byte| {
            let value = f32::from(byte) / 255.0;
            if srgb { srgb_to_linear(value) } else { value }
        })
        .collect();
    let (floats, width, height) = mip_cap_floats(floats, width, height, normals);
    let bytes = floats
        .iter()
        .map(|&value| quantize(if srgb { linear_to_srgb(value) } else { value }))
        .collect();
    (bytes, width, height)
}

/// Box-halve RGBA f32 until both axes fit [`MAX_DIM`]. `normals` treats
/// texels as packed directions: decode to vectors, average, renormalize,
/// re-pack — a plain average of packed components would shorten and bend
/// them.
fn mip_cap_floats(
    mut rgba: Vec<f32>,
    mut width: u32,
    mut height: u32,
    normals: bool,
) -> (Vec<f32>, u32, u32) {
    while width.max(height) > MAX_DIM {
        let nw = (width / 2).max(1);
        let nh = (height / 2).max(1);
        let mut halved = vec![0.0f32; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = [0.0f32; 4];
                for (sy, sx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let px = (2 * x + sx).min(width - 1) as usize;
                    let py = (2 * y + sy).min(height - 1) as usize;
                    let texel = &rgba[(py * width as usize + px) * 4..][..4];
                    if normals {
                        for (accumulated, &component) in sum.iter_mut().zip(texel) {
                            *accumulated += component * 2.0 - 1.0;
                        }
                    } else {
                        for (accumulated, &component) in sum.iter_mut().zip(texel) {
                            *accumulated += component;
                        }
                    }
                }
                let out = &mut halved[((y * nw + x) * 4) as usize..][..4];
                if normals {
                    let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2])
                        .sqrt()
                        .max(1e-6);
                    out[0] = sum[0] / length * 0.5 + 0.5;
                    out[1] = sum[1] / length * 0.5 + 0.5;
                    out[2] = sum[2] / length * 0.5 + 0.5;
                    out[3] = sum[3] * 0.25;
                } else {
                    for (target, value) in out.iter_mut().zip(sum) {
                        *target = value * 0.25;
                    }
                }
            }
        }
        rgba = halved;
        width = nw;
        height = nh;
    }
    (rgba, width, height)
}

// -- Small numeric helpers ---------------------------------------------------

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn quantize(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// IEEE half bits of `value` for BC6H's unsigned-float input: negatives
/// clamp to zero, overflow saturates at the largest finite half, normals
/// round to nearest even, subnormals to nearest.
fn half_bits(value: f32) -> u16 {
    let value = if value > 0.0 { value.min(65504.0) } else { 0.0 };
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    if exponent >= 113 {
        let half = ((exponent - 112) << 10) | ((bits >> 13) & 0x3ff);
        let rest = bits & 0x1fff;
        let round_up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
        (half + u32::from(round_up)) as u16
    } else if exponent >= 100 {
        // Half subnormal: value = mantissa × 2^-24.
        let mantissa = (bits & 0x7f_ffff) | 0x80_0000;
        let shift = 126 - exponent;
        let half = mantissa >> shift;
        let round_up = mantissa & (1 << (shift - 1)) != 0;
        (half + u32::from(round_up)) as u16
    } else {
        0
    }
}

/// Round an image's rows up to multiples of 4 texels by edge replication —
/// the BC encoders assume whole blocks, and the padded tail rows/columns
/// never sample (the true extent is what the image view carries; the
/// upload's row length spans the padding).
fn pad_to_blocks<T: Copy>(
    data: &[T],
    width: u32,
    height: u32,
    channels: usize,
) -> (Vec<T>, u32, u32) {
    let pw = width.next_multiple_of(4);
    let ph = height.next_multiple_of(4);
    if pw == width && ph == height {
        return (data.to_vec(), width, height);
    }
    let mut padded = Vec::with_capacity((pw * ph) as usize * channels);
    for y in 0..ph {
        let sy = y.min(height - 1) as usize;
        let row = &data[sy * width as usize * channels..][..width as usize * channels];
        padded.extend_from_slice(row);
        let edge = &row[row.len() - channels..];
        for _ in width..pw {
            padded.extend_from_slice(edge);
        }
    }
    (padded, pw, ph)
}

// -- The DDS cache format ----------------------------------------------------

/// Just enough DDS to be our own cache: magic, the legacy header with the
/// validity hash parked in its reserved words, a DX10 extension naming
/// the BC format, then the block data. Written and read only by this
/// module — external tools opening one is a debugging nicety, not a
/// contract — which is why ~90 lines beat a dependency here.
mod dds {
    use super::Prepared;
    use ash::vk;

    const MAGIC: u32 = 0x2053_4444; // "DDS "
    /// reserved1[0]: marks the file as this module's cache, versioned
    /// implicitly through the hash's `PIPELINE_VERSION` input.
    const CACHE_MARK: u32 = 0x434e_5431; // "CNT1"
    // DDSD_CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE
    const FLAGS: u32 = 0x1 | 0x2 | 0x4 | 0x1000 | 0x0008_0000;
    const FOURCC_DX10: u32 = 0x3031_5844; // "DX10"

    fn dxgi_format(format: vk::Format) -> u32 {
        match format {
            vk::Format::BC4_UNORM_BLOCK => 80,
            vk::Format::BC5_UNORM_BLOCK => 83,
            vk::Format::BC6H_UFLOAT_BLOCK => 95,
            vk::Format::BC7_UNORM_BLOCK => 98,
            vk::Format::BC7_SRGB_BLOCK => 99,
            _ => unreachable!("every prepared format is a BC block format"),
        }
    }

    fn vk_format(dxgi: u32) -> Option<vk::Format> {
        Some(match dxgi {
            80 => vk::Format::BC4_UNORM_BLOCK,
            83 => vk::Format::BC5_UNORM_BLOCK,
            95 => vk::Format::BC6H_UFLOAT_BLOCK,
            98 => vk::Format::BC7_UNORM_BLOCK,
            99 => vk::Format::BC7_SRGB_BLOCK,
            _ => return None,
        })
    }

    /// Bytes per 4×4 block.
    pub(crate) fn block_size(format: vk::Format) -> u64 {
        match format {
            vk::Format::BC4_UNORM_BLOCK => 8,
            _ => 16,
        }
    }

    pub(super) fn compose(prepared: &Prepared, hash: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(148 + prepared.data.len());
        let mut word = |value: u32| out.extend_from_slice(&value.to_le_bytes());
        word(MAGIC);
        word(124); // header size
        word(FLAGS);
        word(prepared.height);
        word(prepared.width);
        word(prepared.data.len() as u32); // linear size of the one level
        word(0); // depth
        word(0); // mip count (0 and 1 both mean "just the base")
        word(CACHE_MARK);
        word(hash as u32);
        word((hash >> 32) as u32);
        for _ in 3..11 {
            word(0);
        }
        // DDS_PIXELFORMAT: size, DDPF_FOURCC, "DX10", five zeros.
        word(32);
        word(0x4);
        word(FOURCC_DX10);
        for _ in 0..5 {
            word(0);
        }
        word(0x1000); // caps: DDSCAPS_TEXTURE
        for _ in 0..4 {
            word(0); // caps2..4, reserved2
        }
        // DX10 extension: format, TEXTURE2D, no misc, one layer.
        word(dxgi_format(prepared.format));
        word(3);
        word(0);
        word(1);
        word(0);
        debug_assert_eq!(out.len(), 148);
        out.extend_from_slice(&prepared.data);
        out
    }

    /// Parse a cache written by [`compose`], accepting it only when it
    /// carries `hash` and its data length matches its own dimensions —
    /// anything else is a miss for the caller to re-encode over.
    pub(super) fn parse(bytes: &[u8], hash: u64) -> Option<Prepared> {
        let word = |index: usize| -> Option<u32> {
            Some(u32::from_le_bytes(
                bytes.get(index * 4..index * 4 + 4)?.try_into().ok()?,
            ))
        };
        if word(0)? != MAGIC || word(1)? != 124 || word(8)? != CACHE_MARK {
            return None;
        }
        if u64::from(word(9)?) | (u64::from(word(10)?) << 32) != hash {
            return None;
        }
        if word(21)? != FOURCC_DX10 {
            return None;
        }
        let height = word(3)?;
        let width = word(4)?;
        let format = vk_format(word(32)?)?;
        let expected =
            u64::from(width.div_ceil(4)) * u64::from(height.div_ceil(4)) * block_size(format);
        let data = bytes.get(148..)?;
        if width == 0 || height == 0 || data.len() as u64 != expected {
            return None;
        }
        Some(Prepared {
            width,
            height,
            format,
            data: data.to_vec(),
            hash,
        })
    }
}

pub(crate) use dds::block_size;

/// Write an RGBA PNG — the fixture builder every texture test in the
/// crate (prep's edit walk, the furnace and probe scenes) shares.
#[cfg(test)]
pub(crate) fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let mut bytes = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut bytes),
        rgba,
        width,
        height,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .expect("in-memory PNG encode");
    fs::write(path, bytes).expect("test PNG write");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cenote-texture-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Decode one BC4 block (8 bytes) into its 16 texel values — the
    /// reference decoder that keeps the scalar path honest without a GPU.
    fn decode_bc4(block: &[u8]) -> [u8; 16] {
        let (e0, e1) = (block[0], block[1]);
        let palette: [u8; 8] = if e0 > e1 {
            std::array::from_fn(|i| match i {
                0 => e0,
                1 => e1,
                _ => ((u16::from(e0) * (8 - i as u16) + u16::from(e1) * (i as u16 - 1)) / 7) as u8,
            })
        } else {
            std::array::from_fn(|i| match i {
                0 => e0,
                1 => e1,
                6 => 0,
                7 => 255,
                _ => ((u16::from(e0) * (6 - i as u16) + u16::from(e1) * (i as u16 - 1)) / 5) as u8,
            })
        };
        let mut indices = u64::from_le_bytes(block.try_into().expect("8-byte block")) >> 16;
        std::array::from_fn(|_| {
            let value = palette[(indices & 0x7) as usize];
            indices >>= 3;
            value
        })
    }

    #[test]
    fn half_conversion_hits_reference_values() {
        for (value, bits) in [
            (0.0, 0x0000),
            (1.0, 0x3c00),
            (0.5, 0x3800),
            (2.0, 0x4000),
            (65504.0, 0x7bff),
            (1e9, 0x7bff),            // saturates, never inf
            (-3.0, 0x0000),           // unsigned: negatives clamp
            (5.960_464_5e-8, 0x0001), // smallest half subnormal
            (0.333_251_95, 0x3555),   // 1/3 rounded to nearest even
        ] {
            assert_eq!(half_bits(value), bits, "half({value})");
        }
    }

    /// TGA is the one source format with no magic bytes: decode must fall
    /// back to the extension hint, and must read the format's BGR order.
    #[test]
    fn tga_decodes_via_the_extension_hint() {
        // Uncompressed 24-bit truecolor, 1×1, one pure-red texel (BGR).
        let mut tga = vec![0u8; 18];
        tga[2] = 2; // image type: uncompressed truecolor
        tga[12] = 1; // width
        tga[14] = 1; // height
        tga[16] = 24; // bits per pixel
        tga.extend_from_slice(&[0, 0, 255]);

        let Source::Bytes {
            rgba,
            width,
            height,
        } = decode(Path::new("texel.tga"), &tga).expect("TGA decodes")
        else {
            panic!("byte source expected");
        };
        assert_eq!((width, height), (1, 1));
        assert_eq!(rgba, [255, 0, 0, 255]);

        // Without the extension there is nothing to hint with — the sniff
        // error must survive untouched.
        assert!(decode(Path::new("texel"), &tga).is_err());
    }

    #[test]
    fn srgb_transfer_round_trips_every_byte() {
        for byte in 0..=255u8 {
            let linear = srgb_to_linear(f32::from(byte) / 255.0);
            assert_eq!(quantize(linear_to_srgb(linear)), byte);
        }
    }

    #[test]
    fn mip_cap_halves_to_the_cap_and_averages() {
        let width = MAX_DIM * 2;
        let rgba = vec![0.5f32; (width * 2 * 4) as usize];
        let (halved, w, h) = mip_cap_floats(rgba, width, 2, false);
        assert_eq!((w, h), (MAX_DIM, 1));
        assert!(halved.iter().all(|&v| (v - 0.5).abs() < 1e-6));

        // A checker of 0 and 1 wider than the cap averages to exactly 0.5.
        let checker: Vec<f32> = (0..width).flat_map(|x| [(x % 2) as f32; 4]).collect();
        let (avg, w, h) = mip_cap_floats(checker, width, 1, false);
        assert_eq!((w, h), (MAX_DIM, 1));
        assert!(
            avg.iter().all(|&v| (v - 0.5).abs() < 1e-6),
            "halving should average texel pairs"
        );
    }

    #[test]
    fn padding_replicates_edges_to_block_multiples() {
        // A 3×2 single-channel image pads to 4×4.
        let data = [1u8, 2, 3, 4, 5, 6];
        let (padded, pw, ph) = pad_to_blocks(&data, 3, 2, 1);
        assert_eq!((pw, ph), (4, 4));
        assert_eq!(padded, [1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6, 4, 5, 6, 6]);
    }

    /// The scalar path end to end, decoded by the reference BC4 decoder:
    /// a flat block must come back exact, a two-value block within BC4's
    /// 3-bit palette quantization.
    #[test]
    fn scalar_textures_encode_to_faithful_bc4() {
        let dir = scratch_dir("bc4");
        let path = dir.join("mask.png");
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for (index, texel) in rgba.chunks_exact_mut(4).enumerate() {
            let value = if index < 8 { 64 } else { 192 };
            texel.copy_from_slice(&[value, 0, 0, 255]);
        }
        write_png(&path, 4, 4, &rgba);
        let prepared = prepare(&path, Usage::Scalar, None, Channel::R).expect("prepare");
        assert_eq!(prepared.format, vk::Format::BC4_UNORM_BLOCK);
        assert_eq!(prepared.data.len(), 8);
        let decoded = decode_bc4(&prepared.data);
        for (index, value) in decoded.iter().enumerate() {
            let expected = if index < 8 { 64i32 } else { 192 };
            assert!(
                (i32::from(*value) - expected).abs() <= 2,
                "texel {index}: {value} vs {expected}"
            );
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// Each channel of one packed image bakes its own BC4 — four distinct
    /// caches, each holding that channel's flat value exactly — and red
    /// spells the cache name prep used before channels existed.
    #[test]
    fn scalar_channels_bake_and_cache_separately() {
        let dir = scratch_dir("channels");
        let path = dir.join("orm.png");
        let texel = [10u8, 120, 200, 250];
        let rgba: Vec<u8> = texel.iter().copied().cycle().take(4 * 4 * 4).collect();
        write_png(&path, 4, 4, &rgba);
        for (channel, expected) in [
            (Channel::R, 10u8),
            (Channel::G, 120),
            (Channel::B, 200),
            (Channel::A, 250),
        ] {
            let prepared = prepare(&path, Usage::Scalar, None, channel).expect("prepare");
            assert_eq!(prepared.format, vk::Format::BC4_UNORM_BLOCK);
            let decoded = decode_bc4(&prepared.data);
            assert!(
                decoded.iter().all(|&value| value == expected),
                "channel {channel:?}: {decoded:?} vs flat {expected}"
            );
            assert!(cache_path(&path, Usage::Scalar, None, channel).exists());
        }
        assert_eq!(
            cache_path(&path, Usage::Scalar, None, Channel::R),
            dir.join("orm.png.scalar.dds"),
            "red must keep the pre-channel cache name"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// The cache lifecycle: a first prep writes the DDS, a second returns
    /// the identical bytes from it, corruption re-encodes over it, and a
    /// source edit invalidates it by hash.
    #[test]
    fn the_dds_cache_hits_and_invalidates_by_content() {
        let dir = scratch_dir("cache");
        let path = dir.join("wood.png");
        write_png(&path, 8, 8, &vec![180u8; 8 * 8 * 4]);

        let first = prepare(&path, Usage::Color, None, Channel::R).expect("prepare");
        let cache = cache_path(&path, Usage::Color, None, Channel::R);
        assert!(cache.exists(), "prep should write {}", cache.display());
        assert_eq!(first.format, vk::Format::BC7_SRGB_BLOCK);

        let second = prepare(&path, Usage::Color, None, Channel::R).expect("prepare again");
        assert_eq!(first.data, second.data);
        assert_eq!((first.width, first.height), (second.width, second.height));

        // Corrupt the cache: parse fails, prep re-encodes and rewrites.
        fs::write(&cache, b"not a dds").expect("corrupt");
        let repaired = prepare(&path, Usage::Color, None, Channel::R).expect("repair");
        assert_eq!(first.data, repaired.data);
        assert!(
            dds::parse(
                &fs::read(&cache).expect("read"),
                cache_hash(
                    &fs::read(&path).expect("read"),
                    Usage::Color,
                    None,
                    Channel::R
                )
            )
            .is_some()
        );

        // Edit the source: the stale hash reads as a miss.
        write_png(&path, 8, 8, &vec![20u8; 8 * 8 * 4]);
        let edited = prepare(&path, Usage::Color, None, Channel::R).expect("re-encode");
        assert_ne!(first.data, edited.data);

        // Distinct usages cache separately.
        let scalar = prepare(&path, Usage::Scalar, None, Channel::R).expect("scalar");
        assert_ne!(scalar.format, edited.format);
        assert!(cache_path(&path, Usage::Scalar, None, Channel::R).exists());
        fs::remove_dir_all(&dir).ok();
    }

    /// Color-space handling: default 8-bit color is an sRGB view over the
    /// exact source bytes; an explicit linear override switches the view;
    /// a float source becomes BC6H regardless.
    #[test]
    fn color_spaces_resolve_by_slot_and_source() {
        let dir = scratch_dir("spaces");
        let path = dir.join("c.png");
        write_png(&path, 4, 4, &[255u8; 4 * 4 * 4]);
        assert_eq!(
            prepare(&path, Usage::Color, None, Channel::R)
                .expect("srgb")
                .format,
            vk::Format::BC7_SRGB_BLOCK
        );
        assert_eq!(
            prepare(&path, Usage::Color, Some(false), Channel::R)
                .expect("linear")
                .format,
            vk::Format::BC7_UNORM_BLOCK
        );

        let exr = dir.join("c.exr");
        crate::output::write_exr(&exr, 4, 4, &vec![1.5f32; 4 * 4 * 4]).expect("exr");
        assert_eq!(
            prepare(&exr, Usage::Color, None, Channel::R)
                .expect("hdr")
                .format,
            vk::Format::BC6H_UFLOAT_BLOCK
        );

        // Radiance HDR is a float source too — quantizing it to BC7 would
        // clip its range and apply an sRGB view RGBE never had.
        let hdr = dir.join("c.hdr");
        let mut rgbe = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        rgbe.extend_from_slice(&[128, 64, 32, 129, 128, 64, 32, 129]);
        fs::write(&hdr, rgbe).expect("hdr bytes");
        assert_eq!(
            prepare(&hdr, Usage::Color, None, Channel::R)
                .expect("radiance")
                .format,
            vk::Format::BC6H_UFLOAT_BLOCK
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// The stat-revalidated fast path must agree with the full read: same
    /// hash as [`prepare`] stamps, stable across the memoized second look,
    /// and moved by a content edit (which moves size or mtime).
    #[test]
    fn the_expected_hash_matches_prepare_without_rereading() {
        let dir = scratch_dir("stamp");
        let path = dir.join("s.png");
        write_png(&path, 2, 2, &[10u8; 2 * 2 * 4]);
        let first = expected_hash(&path, Usage::Color, None, Channel::R).expect("hash");
        let prepared = prepare(&path, Usage::Color, None, Channel::R).expect("prep");
        assert_eq!(first, prepared.hash);
        assert_eq!(
            first,
            expected_hash(&path, Usage::Color, None, Channel::R).expect("memoized")
        );
        assert_ne!(
            first,
            expected_hash(&path, Usage::Scalar, None, Channel::G).expect("other usage")
        );

        write_png(&path, 4, 4, &[20u8; 4 * 4 * 4]);
        let edited = expected_hash(&path, Usage::Color, None, Channel::R).expect("edited");
        assert_ne!(first, edited);
        assert_eq!(
            edited,
            prepare(&path, Usage::Color, None, Channel::R).expect("prep").hash
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unreadable_and_undecodable_sources_are_scene_errors() {
        let dir = scratch_dir("errors");
        let missing = prepare(&dir.join("nope.png"), Usage::Color, None, Channel::R);
        assert!(matches!(missing, Err(Error::Scene(_))));

        let garbage = dir.join("garbage.png");
        fs::write(&garbage, b"not an image").expect("write");
        let undecodable = prepare(&garbage, Usage::Color, None, Channel::R);
        assert!(matches!(undecodable, Err(Error::Scene(_))));
        fs::remove_dir_all(&dir).ok();
    }
}
