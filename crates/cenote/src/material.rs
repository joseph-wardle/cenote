//! `OpenPBR` surface parameters — the host half of the material schema.
//! `struct Material` in `shaders/openpbr.slang` mirrors [`Material`] field
//! for field; the scene uploads one record per instance, indexed by TLAS
//! custom index, and adding a parameter touches those two definitions and
//! nothing else.
//!
//! The M2 closure set: an EON diffuse base under a dielectric GGX specular
//! layer at a variable IOR, blended toward a conductor by `metalness` and
//! toward rough glass by `transmission_weight`, under an optional clear
//! coat and a fuzz (sheen) layer, plus emission, fractional opacity, and
//! thin-walled surfaces — each a named `OpenPBR` attribute. The texturable
//! subset (base color, roughness, metalness, emission, opacity, plus a
//! tangent-space normal map) carries a bindless-table index next to its
//! constant; [`TEXTURE_NONE`] means constant-everywhere, and prep resolves
//! scene-file texture references into live indices.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// "No texture" in a material's texture-index slots — the parameter is
/// its constant everywhere. Matches `TEXTURE_NONE` in
/// `shaders/textures.slang`.
pub const TEXTURE_NONE: u32 = u32::MAX;

/// One instance's surface, as the shading kernel reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    /// Diffuse albedo — and the conductor's F0 as `metalness` rises — in
    /// `ACEScg`; convert authored `Rec.709` values through
    /// [`crate::color::acescg_from_rec709`] first.
    pub base_color: Vec3,
    /// Diffuse (Oren-Nayar) roughness in [0, 1]; 0 is Lambert.
    pub base_roughness: f32,
    /// Emitted radiance in `ACEScg`, from both faces (surfaces are
    /// two-sided throughout). Nonzero marks the instance as a light.
    pub emission: Vec3,
    /// Conductor blend in [0, 1]: 0 is the dielectric base, 1 pure metal.
    pub metalness: f32,
    /// The dielectric specular layer's weight in [0, 1]: 0 removes the
    /// layer (pure EON diffuse), 1 a full Fresnel reflection at
    /// `specular_ior`.
    pub specular_weight: f32,
    /// GGX roughness in [0, 1] of the base's specular lobes (conductor,
    /// dielectric, and glass); values below the kernel's 0.035 floor are
    /// clamped — true mirrors are a later, delta-lobe feature.
    pub specular_roughness: f32,
    /// Index of refraction of the dielectric specular layer and of
    /// transmission. `OpenPBR`'s default is 1.5.
    pub specular_ior: f32,
    /// Rough-glass blend of the dielectric base in [0, 1]: 0 opaque
    /// (specular over diffuse), 1 pure transmissive glass.
    pub transmission_weight: f32,
    /// The color transmitted light has picked up after traveling
    /// `transmission_depth` through the interior (Beer–Lambert), in
    /// `ACEScg`, clamped positive so the extinction it implies stays
    /// finite. White transmits everything.
    pub transmission_color: Vec3,
    /// Distance in meters at which `transmission_color` is reached; 0
    /// applies the tint at the interface itself instead of any interior
    /// absorption.
    pub transmission_depth: f32,
    /// Tint the coat multiplies onto the base below it, in `ACEScg`.
    /// White is untinted.
    pub coat_color: Vec3,
    /// Weight of the clear-coat layer in [0, 1]; 0 removes it.
    pub coat_weight: f32,
    /// GGX roughness of the coat lobe — also roughens the base's specular
    /// through the spec's variance-sum remap.
    pub coat_roughness: f32,
    /// Index of refraction of the coat. `OpenPBR`'s default is 1.6.
    pub coat_ior: f32,
    /// How strongly the coat's internal reflections darken and saturate
    /// the base, 0 (off) to 1 (physical). Default 1.
    pub coat_darkening: f32,
    /// Weight of the fuzz (sheen) layer in [0, 1]; 0 removes it.
    pub fuzz_weight: f32,
    /// Fuzz color in `ACEScg`. White is neutral fiber scatter.
    pub fuzz_color: Vec3,
    /// Fuzz fiber roughness in [0, 1]. Default 0.5.
    pub fuzz_roughness: f32,
    /// Coverage: 1 opaque, 0 invisible. Fractional values resolve
    /// stochastically on camera and bounce rays, multiplicatively on
    /// shadow rays.
    pub opacity: f32,
    /// Bool: thin-walled surfaces (leaves, soap bubbles, paper) have no
    /// interior — transmission passes straight through without refraction
    /// or Beer–Lambert.
    pub thin_walled: u32,
    /// Bindless-table index of the `base_color` map, or [`TEXTURE_NONE`]:
    /// a textured parameter keeps its constant as written and the kernel
    /// replaces it per hit (constants multiply only where a slot's
    /// semantics say so — emission). Same convention for every `*_texture`
    /// slot below.
    pub base_color_texture: u32,
    /// `specular_roughness` map (scalar, red channel), or [`TEXTURE_NONE`].
    pub specular_roughness_texture: u32,
    /// `base_metalness` map (scalar), or [`TEXTURE_NONE`].
    pub metalness_texture: u32,
    /// `emission_color` map — multiplied onto `emission`, the
    /// LDR-map × luminance-scale convention — or [`TEXTURE_NONE`].
    pub emission_texture: u32,
    /// `geometry_opacity` map (scalar), multiplied onto `opacity` at each
    /// traversal crossing, or [`TEXTURE_NONE`].
    pub opacity_texture: u32,
    /// Tangent-space normal map (BC5 x/y), or [`TEXTURE_NONE`].
    pub normal_texture: u32,
}

// The GPU reads this as a std430 record, where every `float3` aligns to 16
// bytes. The fields are hand-ordered so each `Vec3` lands on a 16-byte
// offset with no host padding; this pins the total, so a reorder that
// reintroduces a gap fails to compile instead of silently misreading on the
// GPU.
const _: () = assert!(size_of::<Material>() == 144);

impl Material {
    /// A pure diffuse surface — no specular layer. The exact-energy base
    /// case the furnace tests lean on.
    #[must_use]
    pub fn matte(base_color: Vec3, base_roughness: f32) -> Self {
        Self {
            base_color,
            base_roughness,
            emission: Vec3::ZERO,
            metalness: 0.0,
            specular_weight: 0.0,
            specular_roughness: 0.0,
            specular_ior: 1.5,
            transmission_weight: 0.0,
            transmission_color: Vec3::ONE,
            transmission_depth: 0.0,
            coat_color: Vec3::ONE,
            coat_weight: 0.0,
            coat_roughness: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: Vec3::ONE,
            fuzz_roughness: 0.5,
            opacity: 1.0,
            thin_walled: 0,
            base_color_texture: TEXTURE_NONE,
            specular_roughness_texture: TEXTURE_NONE,
            metalness_texture: TEXTURE_NONE,
            emission_texture: TEXTURE_NONE,
            opacity_texture: TEXTURE_NONE,
            normal_texture: TEXTURE_NONE,
        }
    }

    /// A dielectric with a full specular layer over its diffuse base —
    /// plastic, ceramic, paint.
    #[must_use]
    pub fn glossy(base_color: Vec3, base_roughness: f32, specular_roughness: f32) -> Self {
        Self {
            specular_weight: 1.0,
            specular_roughness,
            ..Self::matte(base_color, base_roughness)
        }
    }

    /// A conductor: `base_color` becomes F0, the normal-incidence
    /// reflectivity.
    #[must_use]
    pub fn metal(base_color: Vec3, specular_roughness: f32) -> Self {
        Self {
            metalness: 1.0,
            specular_weight: 1.0,
            specular_roughness,
            ..Self::matte(base_color, 0.0)
        }
    }

    /// Solid rough glass: a fully transmissive dielectric base with an
    /// untinted interior.
    #[must_use]
    pub fn glass(specular_roughness: f32, specular_ior: f32) -> Self {
        Self {
            transmission_weight: 1.0,
            specular_weight: 1.0,
            specular_roughness,
            specular_ior,
            ..Self::matte(Vec3::ONE, 0.0)
        }
    }

    /// A light: pure emitter, scattering nothing (black base).
    #[must_use]
    pub fn emitter(emission: Vec3) -> Self {
        Self {
            emission,
            ..Self::matte(Vec3::ZERO, 0.0)
        }
    }

    /// The same surface at a different conductor blend — how the demo's
    /// grid sweeps its spheres from plastic to metal.
    #[must_use]
    pub fn with_metalness(self, metalness: f32) -> Self {
        Self { metalness, ..self }
    }

    /// The same surface under a clear coat.
    #[must_use]
    pub fn with_coat(self, coat_weight: f32, coat_roughness: f32) -> Self {
        Self {
            coat_weight,
            coat_roughness,
            ..self
        }
    }

    /// The same surface under a fuzz (sheen) layer.
    #[must_use]
    pub fn with_fuzz(self, fuzz_weight: f32, fuzz_roughness: f32) -> Self {
        Self {
            fuzz_weight,
            fuzz_roughness,
            ..self
        }
    }

    /// The same surface at a different dielectric IOR.
    #[must_use]
    pub fn with_ior(self, specular_ior: f32) -> Self {
        Self {
            specular_ior,
            ..self
        }
    }

    /// The same surface at fractional coverage.
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        Self { opacity, ..self }
    }

    /// The same surface as a thin-walled shell (no interior).
    #[must_use]
    pub fn thin_walled(self) -> Self {
        Self {
            thin_walled: 1,
            ..self
        }
    }
}
