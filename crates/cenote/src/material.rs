//! `OpenPBR` surface parameters — the host half of the material schema.
//! `struct Material` in `shaders/openpbr.slang` mirrors [`Material`] field
//! for field; the scene uploads one record per instance, indexed by TLAS
//! custom index, and adding a parameter touches those two definitions and
//! nothing else.
//!
//! The M1 subset is the EON diffuse base plus emission. The conductor and
//! dielectric specular parameters join in step 9 — each a named `OpenPBR`
//! attribute, so M2 grows the set instead of rewriting it.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// One instance's surface, as the shading kernel reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    /// Diffuse single-scattering albedo, in `ACEScg` — convert authored
    /// `Rec.709` values through [`crate::color::acescg_from_rec709`] first.
    pub base_color: Vec3,
    /// Diffuse (Oren-Nayar) roughness in [0, 1]; 0 is Lambert.
    pub base_roughness: f32,
    /// Emitted radiance in `ACEScg`, from both faces (surfaces are
    /// two-sided throughout). Nonzero marks the instance as a light — in
    /// M1 that means its mesh must be a parallelogram quad.
    pub emission: Vec3,
    /// Explicit std430 padding (`Pod` forbids implicit padding bytes);
    /// private, so the constructors below are the ways to build one.
    pad: f32,
}

impl Material {
    /// A non-emitting surface — the common case.
    #[must_use]
    pub fn surface(base_color: Vec3, base_roughness: f32) -> Self {
        Self {
            base_color,
            base_roughness,
            emission: Vec3::ZERO,
            pad: 0.0,
        }
    }

    /// A light: pure emitter, scattering nothing (black base).
    #[must_use]
    pub fn emitter(emission: Vec3) -> Self {
        Self {
            base_color: Vec3::ZERO,
            base_roughness: 0.0,
            emission,
            pad: 0.0,
        }
    }
}
