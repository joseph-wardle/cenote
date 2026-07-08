//! `OpenPBR` surface parameters — the host half of the material schema.
//! `struct Material` in `shaders/openpbr.slang` mirrors [`Material`] field
//! for field; the scene uploads one record per instance, indexed by TLAS
//! custom index, and adding a parameter touches those two definitions and
//! nothing else.
//!
//! The M1 step-7 subset is the EON diffuse base alone. Emission joins in
//! step 8, the conductor and dielectric specular parameters in step 9 —
//! each a named `OpenPBR` attribute, so M2 grows the set instead of
//! rewriting it.

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
}
