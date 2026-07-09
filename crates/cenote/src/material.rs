//! `OpenPBR` surface parameters — the host half of the material schema.
//! `struct Material` in `shaders/openpbr.slang` mirrors [`Material`] field
//! for field; the scene uploads one record per instance, indexed by TLAS
//! custom index, and adding a parameter touches those two definitions and
//! nothing else.
//!
//! The M1 subset: an EON diffuse base under a dielectric GGX specular
//! layer, blended toward a conductor by `metalness`, plus emission — each
//! a named `OpenPBR` attribute, so M2 grows the set (textures, coat,
//! transmission) instead of rewriting it. The specular IOR is fixed at
//! `OpenPBR`'s default 1.5 (the energy-compensation fit in the shader is
//! specialized to it).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

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
    /// two-sided throughout). Nonzero marks the instance as a light — in
    /// M1 that means its mesh must be a parallelogram quad.
    pub emission: Vec3,
    /// Conductor blend in [0, 1]: 0 is the dielectric base, 1 pure metal.
    pub metalness: f32,
    /// The dielectric specular layer's weight in [0, 1]: 0 removes the
    /// layer (pure EON diffuse), 1 is a full IOR-1.5 coat.
    pub specular_weight: f32,
    /// GGX roughness in [0, 1] of both specular lobes (conductor and
    /// dielectric); values below the kernel's 0.035 floor are clamped —
    /// true mirrors are a later, delta-lobe feature.
    pub specular_roughness: f32,
    /// Explicit std430 padding (`Pod` forbids implicit padding bytes);
    /// private, so the constructors below are the ways to build one.
    pad: [f32; 2],
}

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
            pad: [0.0; 2],
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
}
