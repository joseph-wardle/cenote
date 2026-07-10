//! pbrt semantics → cenote change-set. This is where the graphics-state
//! machine lives: the current transform, `AttributeBegin`/`End` stack,
//! named textures and materials, the pending area light, and
//! `ObjectBegin` recordings — walked once, front to back, emitting ops.
//!
//! The five fidelity traps this layer owns (each pinned by a test):
//!
//! 1. **Photometric lights.** pbrt divides every light scale by
//!    `SpectrumToPhotometric`, which for an RGB spectrum considers only
//!    the color space's illuminant — so `rgb L [4 4 4]` means *4 nits*,
//!    and RGB light values import verbatim into cenote's ~nit working
//!    convention. `blackbody` spectra *are* normalized (to 1 nit × scale),
//!    so those convert to a luminance-normalized chromaticity.
//! 2. **Roughness remap.** Under the default `remaproughness`, pbrt maps
//!    perceptual roughness to `α = √roughness`; `OpenPBR` maps its slug as
//!    `α = roughness²` — so pbrt's value imports as `roughness^(1/4)`
//!    (and as `√roughness` when remapping is off).
//! 3. **`fov` names the shorter image axis.** Landscape frames import it
//!    as the vertical fov directly; portrait frames convert through
//!    `tan(vfov/2) = tan(fov/2)·height/width`. Resolved at `WorldBegin`,
//!    when both `Camera` and `Film` have been seen.
//! 4. **Handedness.** pbrt is left-handed (`LookAt` builds
//!    `right = up × dir`); cenote is right-handed (`right = forward × up`).
//!    Every world-space transform is conjugated by `M = diag(1, 1, −1)`,
//!    which maps pbrt's camera space exactly onto cenote's — same screen
//!    orientation, no mirror. Scenes whose camera transform is itself
//!    *reflective* (Tungsten-converted exports bake their handedness fix
//!    there) already project right-handed, and get the identity instead —
//!    see [`FLIP_Z`]. `ReverseOrientation` XOR a handedness-swapping
//!    `CTM` flips authored normals and winding, per pbrt's rule; emission
//!    *sidedness* is the one honest divergence (cenote emitters are
//!    two-sided) and is warned, once, with a count.
//! 5. **Octahedral skies** resample to cenote's equirect at import
//!    ([`crate::env`]), orientation and photometric scale baked in.
//!
//! Everything outside the supported subset warns **by token name** —
//! every directive, shape, material, texture class, or parameter this
//! importer drops is named in the warning list, so silence always means
//! "handled".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cenote::scene::changeset::{
    CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, LightPatch, MaterialPatch, MeshPatch,
    Op, SettingsPatch,
};
use cenote::scene::description::{
    ColorSpace, Light, MeshSource, Texturable, TextureRef, Transform,
};
use cenote::{Error, Result};
use glam::{Mat3, Mat4, Vec3};

use crate::parse::{Directive, Parser};

/// The handedness conjugation: pbrt's left-handed world (camera looks
/// +z) onto cenote's right-handed one (camera looks −z). Applied exactly
/// once, on the left of every world-space transform — *when the scene's
/// camera transform is a proper rotation*. Exporters that convert from
/// right-handed packages (Tungsten's pbrt exports, notably) bake their
/// own handedness fix into a *reflective* camera matrix instead; under
/// one of those, pbrt's projection already behaves right-handed, and the
/// conjugation must be the identity or the image mirrors. [`Mapper`]
/// picks per scene at `WorldBegin`, by the camera transform's
/// determinant.
const FLIP_Z: Mat4 = Mat4::from_cols_array_2d(&[
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]);

/// pbrt's defaults for the options block, replaced as directives arrive.
struct Options {
    resolution: [u32; 2],
    spp: u32,
    max_bounces: u32,
    camera: PendingCamera,
}

/// A `Camera` directive, held until `WorldBegin` — the `fov`→vfov
/// conversion needs the film resolution, which may be declared after it.
struct PendingCamera {
    /// camera→world in pbrt's world space (the `CTM` at the directive is
    /// world→camera; this is its inverse).
    world_from_camera: Mat4,
    /// pbrt `fov`: the full angle of the *shorter* image axis, degrees.
    fov: f32,
    lens_radius: f32,
    focal_distance: f32,
}

impl Default for PendingCamera {
    fn default() -> Self {
        Self {
            world_from_camera: Mat4::IDENTITY,
            fov: 90.0,
            lens_radius: 0.0,
            focal_distance: 1e6,
        }
    }
}

/// What a named `Texture` lowers to when a material slot references it.
#[derive(Clone)]
enum TextureDef {
    Image {
        path: PathBuf,
        color_space: Option<ColorSpace>,
        /// pbrt `scale` folded onto the texture. Only emission can carry
        /// it into cenote; other slots warn when it isn't 1.
        scale: f32,
    },
    Constant([f32; 3]),
}

/// The pending `AreaLightSource`, applied to every subsequent shape in
/// the attribute block.
#[derive(Clone)]
struct AreaLight {
    /// Emission as color × luminance-scale, or an image whose texels the
    /// scale multiplies (both exactly pbrt's semantics — see trap 1).
    color: Texturable<[f32; 3]>,
    luminance: f32,
    two_sided: bool,
    /// Materials forked for this light so far, by base material name —
    /// several shapes under one light and material share one fork.
    forks: BTreeMap<String, String>,
}

/// The graphics state `AttributeBegin`/`End` saves and restores.
#[derive(Clone)]
struct State {
    /// The current transform, in pbrt's own world space (the `FLIP_Z`
    /// conjugation happens at emission, never here).
    ctm: Mat4,
    reverse_orientation: bool,
    /// The current material's emitted name and its patch (kept so an
    /// area light can fork it), or `None` before any `Material`
    /// directive (pbrt's implicit default diffuse).
    material: Option<(String, MaterialPatch)>,
    area_light: Option<AreaLight>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            ctm: Mat4::IDENTITY,
            reverse_orientation: false,
            material: None,
            area_light: None,
        }
    }
}

/// One shape recorded inside `ObjectBegin`…`ObjectEnd`: its mesh op is
/// already emitted; instances materialize at each `ObjectInstance`, whose
/// `CTM` composes on top of the shape's own (pbrt records the full
/// declaration-time transform — no inverse of the `ObjectBegin` state).
struct RecordedShape {
    mesh: String,
    material: String,
    ctm: Mat4,
}

/// Ordered, deduplicated import warnings.
#[derive(Default)]
struct Warnings {
    list: Vec<String>,
    seen: std::collections::BTreeSet<String>,
}

impl Warnings {
    fn push(&mut self, warning: String) {
        if self.seen.insert(warning.clone()) {
            self.list.push(warning);
        }
    }
}

/// The whole mapping pass.
pub(crate) struct Mapper {
    parser: Parser,
    /// Where generated assets (resampled or constant skies) are written.
    generated: PathBuf,
    /// Names generated assets after the scene file.
    stem: String,
    ops: Vec<Op>,
    warnings: Warnings,
    options: Options,
    in_world: bool,
    state: State,
    attribute_stack: Vec<State>,
    transform_stack: Vec<Mat4>,
    named_textures: BTreeMap<String, TextureDef>,
    named_materials: BTreeMap<String, MaterialPatch>,
    /// Finished `ObjectBegin` recordings, by object name.
    objects: BTreeMap<String, Vec<RecordedShape>>,
    /// The recording in progress, if any.
    active_object: Option<(String, Vec<RecordedShape>)>,
    counters: BTreeMap<String, u32>,
    /// The world-space conjugation every emitted transform passes
    /// through: [`FLIP_Z`], or the identity for reflective-camera scenes
    /// (see [`FLIP_Z`]'s doc). Chosen at `WorldBegin`.
    conjugation: Mat4,
    environment_emitted: bool,
    one_sided_lights: u32,
    default_material_emitted: bool,
}

impl Mapper {
    pub fn new(parser: Parser, generated: &Path, stem: String) -> Self {
        Self {
            parser,
            generated: generated.to_owned(),
            stem,
            ops: Vec::new(),
            warnings: Warnings::default(),
            options: Options {
                resolution: [1280, 720],
                spp: 16,
                max_bounces: 5,
                camera: PendingCamera::default(),
            },
            in_world: false,
            state: State::default(),
            attribute_stack: Vec::new(),
            transform_stack: Vec::new(),
            named_textures: BTreeMap::new(),
            named_materials: BTreeMap::new(),
            objects: BTreeMap::new(),
            active_object: None,
            counters: BTreeMap::new(),
            conjugation: FLIP_Z,
            environment_emitted: false,
            one_sided_lights: 0,
            default_material_emitted: false,
        }
    }

    /// Walk the whole directive stream and close out the change-set.
    pub fn run(mut self) -> Result<(ChangeSet, Vec<String>)> {
        while let Some(directive) = self.parser.next_directive()? {
            self.dispatch(&directive)?;
        }
        if !self.in_world {
            return Err(Error::SceneFormat(
                "the scene never reaches WorldBegin — not a pbrt scene?".into(),
            ));
        }
        if self.one_sided_lights > 0 {
            self.warnings.push(format!(
                "{} area light(s) are one-sided in pbrt; cenote emitters are two-sided, so \
                 they also emit from their back faces",
                self.one_sided_lights
            ));
        }
        Ok((ChangeSet { ops: self.ops }, self.warnings.list))
    }

    /// A fresh deterministic name: `prefix-0`, `prefix-1`, …
    fn fresh(&mut self, prefix: &str) -> String {
        let counter = self.counters.entry(prefix.to_owned()).or_insert(0);
        let name = format!("{prefix}-{counter}");
        *counter += 1;
        name
    }

    fn warn(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// The "silence never means handled" backstop, run after every
    /// directive whose parameters were consumed selectively.
    fn warn_unused(&mut self, directive: &Directive, context: &str) {
        let warnings = &mut self.warnings;
        directive
            .params
            .warn_unused(context, |warning| warnings.push(warning));
    }

    /// pbrt's block rules, kept because they catch real mistakes: scene
    /// content before `WorldBegin` (or options after it) means a broken
    /// or truncated file, and importing it quietly would drop objects.
    fn verify_block(&self, directive: &Directive, world: bool) -> Result<()> {
        if world != self.in_world {
            let place = if world { "after" } else { "before" };
            return Err(Error::SceneFormat(format!(
                "{}: {} must appear {place} WorldBegin",
                directive.location, directive.keyword
            )));
        }
        Ok(())
    }

    fn dispatch(&mut self, directive: &Directive) -> Result<()> {
        match directive.keyword.as_str() {
            // The current transform, in pbrt world space.
            "Identity" => self.state.ctm = Mat4::IDENTITY,
            "Translate" => {
                self.state.ctm *= Mat4::from_translation(vec3(&directive.numbers, 0));
            }
            "Scale" => self.state.ctm *= Mat4::from_scale(vec3(&directive.numbers, 0)),
            "Rotate" => {
                let axis = vec3(&directive.numbers, 1);
                let axis = axis.try_normalize().ok_or_else(|| {
                    Error::SceneFormat(format!("{}: Rotate about a zero axis", directive.location))
                })?;
                let angle = (directive.numbers[0] as f32).to_radians();
                self.state.ctm *= Mat4::from_axis_angle(axis, angle);
            }
            "LookAt" => self.look_at(directive)?,
            // The file's sixteen numbers are column-major, like glam.
            "Transform" => self.state.ctm = matrix16(&directive.numbers),
            "ConcatTransform" => self.state.ctm *= matrix16(&directive.numbers),
            "TransformBegin" => self.transform_stack.push(self.state.ctm),
            "TransformEnd" => {
                self.state.ctm = self.transform_stack.pop().ok_or_else(|| {
                    Error::SceneFormat(format!(
                        "{}: TransformEnd without TransformBegin",
                        directive.location
                    ))
                })?;
            }
            "CoordinateSystem" | "CoordSysTransform" => self.warn(format!(
                "{}: named coordinate systems are not supported — {} ignored",
                directive.location, directive.keyword
            )),
            "ActiveTransform" | "TransformTimes" => self.warn(format!(
                "{}: animated transforms are not supported — the static transform is used",
                directive.location
            )),

            // Blocks and flags.
            "WorldBegin" => self.world_begin(directive)?,
            // WorldEnd is accepted for older files (pbrt-v4 ends the
            // world at end of input); filtering, acceleration, and
            // renderer options are cenote's own choices — accepted whole,
            // because none of their parameters describe the *scene*.
            "WorldEnd" | "PixelFilter" | "Accelerator" | "Option" => {}
            "AttributeBegin" => self.attribute_stack.push(self.state.clone()),
            "AttributeEnd" => {
                self.state = self.attribute_stack.pop().ok_or_else(|| {
                    Error::SceneFormat(format!(
                        "{}: AttributeEnd without AttributeBegin",
                        directive.location
                    ))
                })?;
            }
            "ReverseOrientation" => {
                self.verify_block(directive, true)?;
                self.state.reverse_orientation = !self.state.reverse_orientation;
            }

            // The options block.
            "Camera" => self.camera_directive(directive)?,
            "Film" => self.film_directive(directive)?,
            "Sampler" => {
                self.verify_block(directive, false)?;
                if let Some(spp) = directive.params.integer("pixelsamples")? {
                    self.options.spp = spp.max(1) as u32;
                }
                // The sampler *type* is the renderer's own business — any
                // sampler converges to the same image.
                self.warn_unused(directive, "Sampler");
            }
            "Integrator" => {
                self.verify_block(directive, false)?;
                if let Some(depth) = directive.params.integer("maxdepth")? {
                    self.options.max_bounces = depth.max(1) as u32;
                }
                self.warn_unused(directive, "Integrator");
            }
            "ColorSpace" => {
                if directive.names[0] != "srgb" {
                    self.warn(format!(
                        "{}: color space \"{}\" is not supported — colors are read as \
                         sRGB/Rec.709",
                        directive.location, directive.names[0]
                    ));
                }
            }

            // Scene content.
            "Material" => self.material_directive(directive)?,
            "MakeNamedMaterial" => self.make_named_material(directive)?,
            "NamedMaterial" => self.named_material(directive)?,
            "Texture" => self.texture_directive(directive)?,
            "Shape" => self.shape_directive(directive)?,
            "AreaLightSource" => self.area_light_directive(directive)?,
            "LightSource" => self.light_directive(directive)?,
            "ObjectBegin" => self.object_begin(directive)?,
            "ObjectEnd" => self.object_end(directive)?,
            "ObjectInstance" => self.object_instance(directive)?,

            "MakeNamedMedium" | "MediumInterface" | "Attribute" => self.warn(format!(
                "{}: {} is not supported — ignored",
                directive.location, directive.keyword
            )),
            other => {
                // The parser's arity table is closed, so this is a new
                // directive it learned before this map did.
                return Err(Error::SceneFormat(format!(
                    "{}: directive {other} parses but has no mapping",
                    directive.location
                )));
            }
        }
        Ok(())
    }

    fn look_at(&mut self, directive: &Directive) -> Result<()> {
        let eye = vec3(&directive.numbers, 0);
        let look = vec3(&directive.numbers, 3);
        let up = vec3(&directive.numbers, 6);
        // pbrt's construction verbatim — right = up × dir is the
        // left-handed choice trap 4 is about.
        let dir = (look - eye).try_normalize().ok_or_else(|| {
            Error::SceneFormat(format!(
                "{}: LookAt eye and target coincide",
                directive.location
            ))
        })?;
        let right = up.normalize().cross(dir).try_normalize().ok_or_else(|| {
            Error::SceneFormat(format!(
                "{}: LookAt up is parallel to the view direction",
                directive.location
            ))
        })?;
        let new_up = dir.cross(right);
        let world_from_camera = Mat4::from_cols(
            right.extend(0.0),
            new_up.extend(0.0),
            dir.extend(0.0),
            eye.extend(1.0),
        );
        // The CTM is world→camera when the Camera directive reads it.
        self.state.ctm *= world_from_camera.inverse();
        Ok(())
    }

    fn camera_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, false)?;
        if directive.names[0] != "perspective" {
            self.warn(format!(
                "{}: camera \"{}\" is not supported — imported as perspective",
                directive.location, directive.names[0]
            ));
        }
        let world_from_camera = self.state.ctm.inverse();
        if !world_from_camera.is_finite() {
            return Err(Error::SceneFormat(format!(
                "{}: the camera transform is not invertible",
                directive.location
            )));
        }
        self.options.camera = PendingCamera {
            world_from_camera,
            fov: directive.params.float("fov")?.unwrap_or(90.0),
            lens_radius: directive.params.float("lensradius")?.unwrap_or(0.0),
            focal_distance: directive.params.float("focaldistance")?.unwrap_or(1e6),
        };
        self.warn_unused(directive, "Camera");
        Ok(())
    }

    fn film_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, false)?;
        if let Some(x) = directive.params.integer("xresolution")? {
            self.options.resolution[0] = x.max(1) as u32;
        }
        if let Some(y) = directive.params.integer("yresolution")? {
            self.options.resolution[1] = y.max(1) as u32;
        }
        // The output name belongs to whoever renders; the rest of the
        // film parameters (sensor, iso, white balance) fall to the
        // unused warning below — they *do* change pbrt's image, so
        // dropping them must be visible.
        let _ = directive.params.string("filename")?;
        self.warn_unused(directive, "Film");
        Ok(())
    }

    /// `WorldBegin`: the options freeze, so the settings and camera ops
    /// can finally be emitted (the `fov` trap needs Camera *and* Film),
    /// and the transform state resets for the world block.
    fn world_begin(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, false)?;
        self.in_world = true;
        self.state.ctm = Mat4::IDENTITY;

        let [width, height] = self.options.resolution;
        self.ops.push(Op::Settings(SettingsPatch {
            resolution: Some([width, height]),
            spp: Some(self.options.spp),
            max_bounces: Some(self.options.max_bounces),
            ..SettingsPatch::new("main")
        }));

        let camera = &self.options.camera;
        // Trap 4's per-scene half: a reflective camera transform already
        // encodes a handedness conversion, so the world conjugation is
        // the identity under one; a proper camera rotation gets FLIP_Z.
        self.conjugation = if swaps_handedness(camera.world_from_camera) {
            Mat4::IDENTITY
        } else {
            FLIP_Z
        };
        // Trap 3: pbrt's fov spans the shorter image axis.
        let vfov = if width >= height {
            camera.fov
        } else {
            let half = (camera.fov.to_radians() / 2.0).tan() * height as f32 / width as f32;
            2.0 * half.atan().to_degrees()
        };
        let camera_to_world = self.conjugation * camera.world_from_camera;
        let position = camera_to_world.transform_point3(Vec3::ZERO);
        let forward = camera_to_world.transform_vector3(Vec3::Z).normalize();
        let up = camera_to_world.transform_vector3(Vec3::Y);
        self.ops.push(Op::Camera(CameraPatch {
            position: Some(position.into()),
            look_at: Some((position + forward).into()),
            up: Some(up.into()),
            vfov_degrees: Some(vfov),
            // Focus only matters through a lens; a pinhole keeps the
            // schema default (focus at look_at) instead of pbrt's 1e6.
            focus_distance: (camera.lens_radius > 0.0).then_some(Some(camera.focal_distance)),
            aperture_radius: Some(camera.lens_radius),
            ..CameraPatch::new("main")
        }));
        Ok(())
    }
}

/// Three consecutive directive numbers as a vector.
fn vec3(numbers: &[f64], offset: usize) -> Vec3 {
    Vec3::new(
        numbers[offset] as f32,
        numbers[offset + 1] as f32,
        numbers[offset + 2] as f32,
    )
}

/// Sixteen column-major numbers as a matrix.
fn matrix16(numbers: &[f64]) -> Mat4 {
    let mut columns = [0.0f32; 16];
    for (slot, number) in columns.iter_mut().zip(numbers) {
        *slot = *number as f32;
    }
    Mat4::from_cols_array(&columns)
}

/// An affine matrix as the schema's row-major three-row transform.
fn matrix_transform(matrix: Mat4) -> Transform {
    let rows = matrix.transpose().to_cols_array_2d();
    Transform::Matrix([rows[0], rows[1], rows[2]])
}

/// Whether a transform mirrors — half of the trap-4 XOR.
fn swaps_handedness(matrix: Mat4) -> bool {
    Mat3::from_mat4(matrix).determinant() < 0.0
}

/// Normal-incidence reflectance from a conductor's complex IOR — how
/// pbrt's `eta`/`k` spectra land in `base_color`'s F0 convention.
fn conductor_f0(eta: [f32; 3], k: [f32; 3]) -> [f32; 3] {
    let mut f0 = [0.0; 3];
    for channel in 0..3 {
        let (n, k) = (eta[channel], k[channel]);
        f0[channel] = ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k);
    }
    f0
}

/// Linear `Rec.709` F0 for the named conductor spectra the corpus uses
/// (pbrt's `metal-*-eta`/`-k` measurements, reduced to normal-incidence
/// RGB — the standard lookdev values).
fn named_metal_f0(spectrum: &str) -> Option<[f32; 3]> {
    let metal = spectrum
        .strip_prefix("metal-")?
        .strip_suffix("-eta")
        .or_else(|| spectrum.strip_prefix("metal-")?.strip_suffix("-k"))?;
    Some(match metal {
        "Cu" => [0.955, 0.638, 0.538],
        "Au" => [1.000, 0.782, 0.344],
        "Ag" => [0.972, 0.960, 0.915],
        "Al" => [0.913, 0.921, 0.925],
        _ => return None,
    })
}

impl Mapper {
    fn shape_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let ty = directive.names[0].as_str();
        // Trap 4's XOR: ReverseOrientation and a mirroring transform
        // each flip the surface's orientation; together they cancel.
        let flip = self.state.reverse_orientation ^ swaps_handedness(self.state.ctm);
        let (source, mesh_prefix) = match ty {
            "trianglemesh" => (trianglemesh(directive, flip)?, "trianglemesh".to_owned()),
            "plymesh" => {
                let file = directive.params.string("filename")?.ok_or_else(|| {
                    Error::SceneFormat(format!(
                        "{}: plymesh has no \"string filename\"",
                        directive.location
                    ))
                })?;
                let path = self.parser.resolve(file);
                if flip {
                    self.warn(format!(
                        "{}: ReverseOrientation on a plymesh is ignored (cenote shades \
                         and emits two-sided)",
                        directive.location
                    ));
                }
                let prefix = path.file_stem().map_or_else(
                    || "plymesh".to_owned(),
                    |stem| stem.to_string_lossy().into_owned(),
                );
                (MeshSource::Ply { path }, prefix)
            }
            "sphere" => {
                let radius = directive.params.float("radius")?.unwrap_or(1.0);
                for clip in ["zmin", "zmax", "phimax"] {
                    if directive.params.float(clip)?.is_some() {
                        self.warn(format!(
                            "{}: partial spheres (\"{clip}\") are not supported — a full \
                             sphere is tessellated",
                            directive.location
                        ));
                    }
                }
                (sphere_mesh(radius), "sphere".to_owned())
            }
            "disk" => {
                let radius = directive.params.float("radius")?.unwrap_or(1.0);
                let height = directive.params.float("height")?.unwrap_or(0.0);
                for clip in ["innerradius", "phimax"] {
                    if directive.params.float(clip)?.is_some() {
                        self.warn(format!(
                            "{}: partial disks (\"{clip}\") are not supported — a full \
                             disk is tessellated",
                            directive.location
                        ));
                    }
                }
                (disk_mesh(radius, height), "disk".to_owned())
            }
            other => {
                self.warn(format!(
                    "{}: shape \"{other}\" is not supported — skipped",
                    directive.location
                ));
                return Ok(());
            }
        };
        let mesh = self.fresh(&mesh_prefix);
        self.ops.push(Op::Mesh(MeshPatch {
            name: mesh.clone(),
            source: Some(source),
        }));
        let material = self.shape_material(&directive.location);
        self.warn_unused(directive, &format!("shape \"{ty}\""));

        if let Some((_, shapes)) = &mut self.active_object {
            shapes.push(RecordedShape {
                mesh,
                material,
                ctm: self.state.ctm,
            });
        } else {
            self.ops.push(Op::Instance(InstancePatch {
                mesh: Some(mesh.clone()),
                material: Some(material),
                transform: Some(matrix_transform(self.conjugation * self.state.ctm)),
                ..InstancePatch::new(mesh)
            }));
        }
        Ok(())
    }

    /// The material this shape wears: the current one, pbrt's implicit
    /// default diffuse, or — under a pending area light — a fork of it
    /// with the emission folded in (shared across the light's shapes).
    fn shape_material(&mut self, location: &str) -> String {
        let (base_name, base_patch) = if let Some((name, patch)) = &self.state.material {
            (name.clone(), patch.clone())
        } else {
            let patch = MaterialPatch {
                base_color: Some(Texturable::Constant([0.5; 3])),
                specular_weight: Some(0.0),
                ..MaterialPatch::new("pbrt-default")
            };
            if !self.default_material_emitted {
                self.default_material_emitted = true;
                self.ops.push(Op::Material(Box::new(patch.clone())));
            }
            ("pbrt-default".to_owned(), patch)
        };
        let Some(mut area) = self.state.area_light.clone() else {
            return base_name;
        };
        if self.active_object.is_some() {
            // pbrt's own limitation, kept: instanced emitters would need
            // per-instance light-table entries it (and this) can't make.
            self.warn(format!(
                "{location}: area lights are not supported with object instancing — \
                 the shape imports unlit"
            ));
            return base_name;
        }
        if let Some(fork) = area.forks.get(&base_name) {
            return fork.clone();
        }
        let fork = self.fresh(&format!("{base_name}-glow"));
        let mut patch = base_patch;
        patch.name.clone_from(&fork);
        patch.emission_color = Some(area.color.clone());
        patch.emission_luminance = Some(area.luminance);
        self.ops.push(Op::Material(Box::new(patch)));
        if !area.two_sided {
            self.one_sided_lights += 1;
        }
        area.forks.insert(base_name, fork.clone());
        self.state.area_light = Some(area);
        fork
    }

    fn area_light_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        if directive.names[0] != "diffuse" {
            self.warn(format!(
                "{}: area light \"{}\" is not supported — shapes import unlit",
                directive.location, directive.names[0]
            ));
            self.state.area_light = None;
            return Ok(());
        }
        let params = &directive.params;
        let luminance = params.float("scale")?.unwrap_or(1.0);
        if params.float("power")?.is_some() {
            self.warn(format!(
                "{}: area light \"power\" needs the shape's surface area — the plain \
                 photometric scale is used instead",
                directive.location
            ));
        }
        let color = match params.string("filename")? {
            Some(file) => {
                if params
                    .take("L", &["rgb", "color", "blackbody", "spectrum"])?
                    .is_some()
                {
                    return Err(Error::SceneFormat(format!(
                        "{}: area light has both \"L\" and \"filename\"",
                        directive.location
                    )));
                }
                Texturable::Texture(TextureRef {
                    path: self.parser.resolve(file),
                    color_space: None,
                })
            }
            None => Texturable::Constant(self.light_color(directive, "L")?),
        };
        let two_sided = params.boolean("twosided")?.unwrap_or(false);
        self.warn_unused(directive, "area light");
        self.state.area_light = Some(AreaLight {
            color,
            luminance,
            two_sided,
            forks: BTreeMap::new(),
        });
        Ok(())
    }

    /// Trap 1 for a light's spectrum parameter: RGB values pass through
    /// verbatim (pbrt's photometric division sees only the illuminant),
    /// blackbodies become a luminance-1 chromaticity (pbrt normalizes
    /// them to 1 nit), and everything else degrades to white, warned.
    fn light_color(&mut self, directive: &Directive, name: &str) -> Result<[f32; 3]> {
        let Some(param) = directive
            .params
            .take(name, &["rgb", "color", "blackbody", "spectrum", "float"])?
        else {
            // pbrt's default is the color space's illuminant: white.
            return Ok([1.0; 3]);
        };
        Ok(match param.ty.as_str() {
            "rgb" | "color" => match param.as_floats()? {
                [r, g, b] => [*r as f32, *g as f32, *b as f32],
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: parameter \"{name}\" needs three values",
                        param.location
                    )));
                }
            },
            "blackbody" => match param.as_floats()? {
                [kelvin] => blackbody_rec709(*kelvin as f32),
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: blackbody \"{name}\" needs one temperature",
                        param.location
                    )));
                }
            },
            "float" => match param.as_floats()? {
                [value] => [*value as f32; 3],
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: parameter \"{name}\" needs one value",
                        param.location
                    )));
                }
            },
            _ => {
                self.warn(format!(
                    "{}: spectral light data is not supported — \"{name}\" imports as \
                     white at its photometric scale",
                    param.location
                ));
                [1.0; 3]
            }
        })
    }

    fn light_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let params = &directive.params;
        match directive.names[0].as_str() {
            "point" => {
                let mut factor = params.float("scale")?.unwrap_or(1.0);
                // pbrt: a target power φ_v spreads over the full sphere.
                if let Some(power) = params.float("power")? {
                    factor *= power / (4.0 * std::f32::consts::PI);
                }
                let color = self.light_color(directive, "I")?;
                let from = params
                    .take("from", &["point3", "point"])?
                    .map(|param| param.as_floats().map(|values| vec3(values, 0)))
                    .transpose()?
                    .unwrap_or(Vec3::ZERO);
                let position = (self.conjugation * self.state.ctm).transform_point3(from);
                let name = self.fresh("point");
                self.ops.push(Op::Light(LightPatch {
                    light: Some(Light::Point {
                        position: position.into(),
                        intensity: color.map(|channel| channel * factor),
                    }),
                    ..LightPatch::new(name)
                }));
            }
            "distant" => {
                let mut factor = params.float("scale")?.unwrap_or(1.0);
                if let Some(illuminance) = params.float("illuminance")? {
                    factor *= illuminance;
                }
                let color = self.light_color(directive, "L")?;
                let point = |name: &str, default: Vec3| -> Result<Vec3> {
                    Ok(params
                        .take(name, &["point3", "point"])?
                        .map(|param| param.as_floats().map(|values| vec3(values, 0)))
                        .transpose()?
                        .unwrap_or(default))
                };
                let from = point("from", Vec3::ZERO)?;
                let to = point("to", Vec3::Z)?;
                let world = self.conjugation * self.state.ctm;
                let direction = world.transform_point3(to) - world.transform_point3(from);
                let direction = direction.try_normalize().ok_or_else(|| {
                    Error::SceneFormat(format!(
                        "{}: distant light \"from\" and \"to\" coincide",
                        directive.location
                    ))
                })?;
                let name = self.fresh("distant");
                self.ops.push(Op::Light(LightPatch {
                    light: Some(Light::Distant {
                        direction: direction.into(),
                        irradiance: color.map(|channel| channel * factor),
                    }),
                    ..LightPatch::new(name)
                }));
            }
            "infinite" => self.infinite_light(directive)?,
            other => {
                self.warn(format!(
                    "{}: light \"{other}\" is not supported — skipped",
                    directive.location
                ));
                return Ok(());
            }
        }
        self.warn_unused(directive, &format!("light \"{}\"", directive.names[0]));
        Ok(())
    }

    /// Trap 5's home: an image infinite light resamples its equal-area
    /// octahedral image to cenote's equirect, baking orientation and
    /// photometric scale; an image-less one becomes a constant sky file.
    fn infinite_light(&mut self, directive: &Directive) -> Result<()> {
        let params = &directive.params;
        if self.environment_emitted {
            self.warn(format!(
                "{}: cenote renders one environment — this infinite light is skipped",
                directive.location
            ));
            return Ok(());
        }
        if params.take("portal", &["point3", "point"])?.is_some() {
            self.warn(format!(
                "{}: light portals are not supported — the portal is ignored",
                directive.location
            ));
        }
        let scale = params.float("scale")?.unwrap_or(1.0);
        let out = self.generated.join(format!("{}-sky.exr", self.stem));
        if let Some(file) = params.string("filename")? {
            if params
                .take("L", &["rgb", "color", "blackbody", "spectrum"])?
                .is_some()
            {
                return Err(Error::SceneFormat(format!(
                    "{}: infinite light has both \"L\" and \"filename\"",
                    directive.location
                )));
            }
            if params.float("illuminance")?.is_some() {
                self.warn(format!(
                    "{}: \"illuminance\" on an image infinite light needs the image's \
                     hemispherical integral — the plain photometric scale is used",
                    directive.location
                ));
            }
            let orientation = Mat3::from_mat4(self.conjugation * self.state.ctm);
            crate::env::resample_octahedral(&self.parser.resolve(file), orientation, scale, &out)?;
        } else {
            let mut factor = scale;
            // A uniform sky delivering illuminance E_v needs L = E_v/π.
            if let Some(illuminance) = params.float("illuminance")? {
                factor *= illuminance / std::f32::consts::PI;
            }
            let color = self.light_color(directive, "L")?;
            crate::env::write_constant(color.map(|channel| channel * factor), &out)?;
        }
        self.ops.push(Op::Environment(EnvironmentPatch {
            path: Some(out),
            ..EnvironmentPatch::new("sky")
        }));
        self.environment_emitted = true;
        Ok(())
    }

    fn object_begin(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let name = directive.names[0].clone();
        if self.active_object.is_some() {
            return Err(Error::SceneFormat(format!(
                "{}: ObjectBegin inside an object definition",
                directive.location
            )));
        }
        if self.objects.contains_key(&name) {
            return Err(Error::SceneFormat(format!(
                "{}: object \"{name}\" is defined twice",
                directive.location
            )));
        }
        // ObjectBegin doubles as AttributeBegin in pbrt.
        self.attribute_stack.push(self.state.clone());
        self.active_object = Some((name, Vec::new()));
        Ok(())
    }

    fn object_end(&mut self, directive: &Directive) -> Result<()> {
        let Some((name, shapes)) = self.active_object.take() else {
            return Err(Error::SceneFormat(format!(
                "{}: ObjectEnd without ObjectBegin",
                directive.location
            )));
        };
        self.state = self.attribute_stack.pop().ok_or_else(|| {
            Error::SceneFormat(format!(
                "{}: ObjectEnd with a mismatched attribute stack",
                directive.location
            ))
        })?;
        self.objects.insert(name, shapes);
        Ok(())
    }

    fn object_instance(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        if self.active_object.is_some() {
            return Err(Error::SceneFormat(format!(
                "{}: ObjectInstance inside an object definition",
                directive.location
            )));
        }
        let name = &directive.names[0];
        if !self.objects.contains_key(name) {
            return Err(Error::SceneFormat(format!(
                "{}: ObjectInstance of \"{name}\", which was never defined",
                directive.location
            )));
        }
        // pbrt composes the instance-time CTM on top of each shape's full
        // declaration-time CTM — no inverse of the ObjectBegin state.
        let mut instances = Vec::new();
        for index in 0..self.objects[name].len() {
            let instance = self.fresh(name);
            let shape = &self.objects[name][index];
            instances.push(Op::Instance(InstancePatch {
                mesh: Some(shape.mesh.clone()),
                material: Some(shape.material.clone()),
                transform: Some(matrix_transform(
                    self.conjugation * self.state.ctm * shape.ctm,
                )),
                ..InstancePatch::new(instance)
            }));
        }
        self.ops.extend(instances);
        Ok(())
    }

    fn material_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let ty = directive.names[0].clone();
        let mut patch = self.lower_material(&ty, directive)?;
        patch.name = self.fresh(&ty);
        self.warn_unused(directive, &format!("material \"{ty}\""));
        self.ops.push(Op::Material(Box::new(patch.clone())));
        self.state.material = Some((patch.name.clone(), patch));
        Ok(())
    }

    fn make_named_material(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let name = directive.names[0].clone();
        if self.named_materials.contains_key(&name) {
            return Err(Error::SceneFormat(format!(
                "{}: named material \"{name}\" is defined twice",
                directive.location
            )));
        }
        let ty = directive
            .params
            .string("type")?
            .ok_or_else(|| {
                Error::SceneFormat(format!(
                    "{}: MakeNamedMaterial \"{name}\" has no \"string type\"",
                    directive.location
                ))
            })?
            .to_owned();
        let mut patch = self.lower_material(&ty, directive)?;
        patch.name.clone_from(&name);
        self.warn_unused(directive, &format!("material \"{name}\""));
        self.ops.push(Op::Material(Box::new(patch.clone())));
        self.named_materials.insert(name, patch);
        Ok(())
    }

    fn named_material(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let name = &directive.names[0];
        let patch = self.named_materials.get(name).ok_or_else(|| {
            Error::SceneFormat(format!(
                "{}: NamedMaterial \"{name}\" was never made",
                directive.location
            ))
        })?;
        self.state.material = Some((name.clone(), patch.clone()));
        Ok(())
    }

    /// pbrt material semantics onto the `OpenPBR` patch. Unsupported
    /// types fall to the schema defaults so the shape still renders —
    /// visibly gray, and named in the warnings.
    fn lower_material(&mut self, ty: &str, directive: &Directive) -> Result<MaterialPatch> {
        let params = &directive.params;
        let mut patch = MaterialPatch::default();
        match ty {
            "diffuse" => {
                patch.base_color = Some(
                    self.color_slot(directive, "reflectance")?
                        .unwrap_or(Texturable::Constant([0.5; 3])),
                );
                patch.specular_weight = Some(0.0);
            }
            "coateddiffuse" => {
                patch.base_color = Some(
                    self.color_slot(directive, "reflectance")?
                        .unwrap_or(Texturable::Constant([0.5; 3])),
                );
                patch.specular_weight = Some(0.0);
                patch.coat_weight = Some(1.0);
                patch.coat_ior = Some(self.dielectric_eta(directive)?);
                patch.coat_roughness = Some(match self.roughness_slot(directive)? {
                    Texturable::Constant(roughness) => roughness,
                    Texturable::Texture(_) => {
                        self.warn(format!(
                            "{}: coat roughness cannot be textured — the coat imports smooth",
                            directive.location
                        ));
                        0.0
                    }
                });
            }
            "conductor" => {
                patch.base_metalness = Some(Texturable::Constant(1.0));
                patch.base_color = Some(self.conductor_color(directive)?);
                patch.specular_roughness = Some(self.roughness_slot(directive)?);
            }
            "dielectric" | "thindielectric" => {
                patch.transmission_weight = Some(1.0);
                patch.specular_ior = Some(self.dielectric_eta(directive)?);
                if ty == "thindielectric" {
                    patch.geometry_thin_walled = Some(true);
                } else {
                    patch.specular_roughness = Some(self.roughness_slot(directive)?);
                }
            }
            other => {
                self.warn(format!(
                    "{}: material \"{other}\" is not supported — OpenPBR defaults used",
                    directive.location
                ));
            }
        }
        if let Some(file) = params.string("normalmap")? {
            patch.geometry_normal = Some(Some(TextureRef {
                path: self.parser.resolve(file),
                color_space: None,
            }));
        }
        Ok(patch)
    }

    /// A conductor's F0: `reflectance` verbatim, `eta`/`k` through the
    /// normal-incidence formula (RGB values) or the named-metal table,
    /// copper — pbrt's own default — when nothing usable is given.
    fn conductor_color(&mut self, directive: &Directive) -> Result<Texturable<[f32; 3]>> {
        let params = &directive.params;
        if let Some(reflectance) = self.color_slot(directive, "reflectance")? {
            // pbrt refuses reflectance and eta/k together.
            if params
                .take("eta", &["rgb", "color", "spectrum", "float", "texture"])?
                .is_some()
                || params
                    .take("k", &["rgb", "color", "spectrum", "float", "texture"])?
                    .is_some()
            {
                return Err(Error::SceneFormat(format!(
                    "{}: conductor has both \"reflectance\" and \"eta\"/\"k\"",
                    directive.location
                )));
            }
            return Ok(reflectance);
        }
        let eta = params.take("eta", &["rgb", "color", "spectrum", "float", "texture"])?;
        let k = params.take("k", &["rgb", "color", "spectrum", "float", "texture"])?;
        let rgb_of = |param: Option<&crate::parse::Param>| -> Option<[f32; 3]> {
            let param = param?;
            if !matches!(param.ty.as_str(), "rgb" | "color") {
                return None;
            }
            match param.as_floats().ok()? {
                [r, g, b] => Some([*r as f32, *g as f32, *b as f32]),
                _ => None,
            }
        };
        if let (Some(eta), Some(k)) = (rgb_of(eta), rgb_of(k)) {
            return Ok(Texturable::Constant(conductor_f0(eta, k)));
        }
        if let Some(param) = eta.or(k) {
            if param.ty == "spectrum"
                && let Ok(name) = param.as_string()
            {
                if let Some(f0) = named_metal_f0(name) {
                    return Ok(Texturable::Constant(f0));
                }
                self.warn(format!(
                    "{}: conductor spectrum \"{name}\" is not in the metal table — copper used",
                    param.location
                ));
            } else {
                self.warn(format!(
                    "{}: conductor \"eta\"/\"k\" of type \"{}\" is not supported — copper used",
                    param.location, param.ty
                ));
            }
        }
        Ok(Texturable::Constant(
            named_metal_f0("metal-Cu-eta").expect("Cu is in the table"),
        ))
    }

    /// The dielectric IOR: a float (or float-typed spectrum degenerates
    /// with a warning). pbrt's parameter name is `eta`, default 1.5.
    fn dielectric_eta(&mut self, directive: &Directive) -> Result<f32> {
        let Some(param) = directive
            .params
            .take("eta", &["float", "spectrum", "rgb", "color"])?
        else {
            return Ok(1.5);
        };
        if param.ty == "float"
            && let [eta] = param.as_floats()?
        {
            return Ok(*eta as f32);
        }
        self.warn(format!(
            "{}: a spectral IOR (dispersion) is not supported — 1.5 used",
            param.location
        ));
        Ok(1.5)
    }

    /// Trap 2: pbrt roughness → the `OpenPBR` slug. Under the default
    /// `remaproughness`, pbrt's α is `√roughness` and `OpenPBR`'s is
    /// `roughness²`, so the value imports as the fourth root (square
    /// root when remapping is off). The curve can't ride a texture, so
    /// textured roughness imports as-is with a warning.
    fn roughness_slot(&mut self, directive: &Directive) -> Result<Texturable<f32>> {
        let params = &directive.params;
        let remap = params.boolean("remaproughness")?.unwrap_or(true);
        let mut roughness = self.float_slot(directive, "roughness")?;
        let anisotropic: Vec<Texturable<f32>> = ["uroughness", "vroughness"]
            .iter()
            .filter_map(|name| self.float_slot(directive, name).transpose())
            .collect::<Result<_>>()?;
        if !anisotropic.is_empty() {
            self.warn(format!(
                "{}: anisotropic roughness is not supported — the axes are averaged",
                directive.location
            ));
            let constants: Vec<f32> = anisotropic
                .iter()
                .filter_map(|value| match value {
                    Texturable::Constant(value) => Some(*value),
                    Texturable::Texture(_) => None,
                })
                .collect();
            if constants.len() == anisotropic.len() {
                roughness = Some(Texturable::Constant(
                    constants.iter().sum::<f32>() / constants.len() as f32,
                ));
            } else {
                roughness = Some(anisotropic.into_iter().next().expect("non-empty"));
            }
        }
        Ok(match roughness.unwrap_or(Texturable::Constant(0.0)) {
            Texturable::Constant(value) => {
                let alpha_exponent = if remap { 0.25 } else { 0.5 };
                Texturable::Constant(value.max(0.0).powf(alpha_exponent))
            }
            Texturable::Texture(reference) => {
                self.warn(format!(
                    "{}: pbrt's roughness remap cannot ride a texture — texel values are \
                     read as OpenPBR roughness directly",
                    directive.location
                ));
                Texturable::Texture(reference)
            }
        })
    }

    /// A color material slot: constant, texture reference, or a shape
    /// this importer degrades with a warning.
    fn color_slot(
        &mut self,
        directive: &Directive,
        name: &str,
    ) -> Result<Option<Texturable<[f32; 3]>>> {
        let Some(param) = directive.params.take(
            name,
            &["rgb", "color", "float", "spectrum", "blackbody", "texture"],
        )?
        else {
            return Ok(None);
        };
        Ok(Some(match param.ty.as_str() {
            "rgb" | "color" => match param.as_floats()? {
                [r, g, b] => Texturable::Constant([*r as f32, *g as f32, *b as f32]),
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: parameter \"{name}\" needs three values",
                        param.location
                    )));
                }
            },
            "float" => match param.as_floats()? {
                [value] => Texturable::Constant([*value as f32; 3]),
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: parameter \"{name}\" needs one value",
                        param.location
                    )));
                }
            },
            "texture" => match self.texture_lookup(param.as_string()?, &param.location)? {
                TextureDef::Constant(value) => Texturable::Constant(value),
                TextureDef::Image {
                    path,
                    color_space,
                    scale,
                } => {
                    self.warn_dropped_scale(scale, name, &param.location);
                    Texturable::Texture(TextureRef { path, color_space })
                }
            },
            other => {
                self.warn(format!(
                    "{}: \"{other} {name}\" is not supported here — mid-gray used",
                    param.location
                ));
                Texturable::Constant([0.5; 3])
            }
        }))
    }

    /// A scaled texture feeding a slot that can't carry the scale
    /// (anything but emission) drops the factor, visibly.
    fn warn_dropped_scale(&mut self, scale: f32, name: &str, location: &str) {
        if (scale - 1.0).abs() > f32::EPSILON {
            self.warn(format!(
                "{location}: a scaled texture feeds \"{name}\", which cannot carry a \
                 scale — the factor {scale} is dropped"
            ));
        }
    }

    /// A scalar material slot: float constant or texture.
    fn float_slot(&mut self, directive: &Directive, name: &str) -> Result<Option<Texturable<f32>>> {
        let Some(param) = directive.params.take(name, &["float", "texture"])? else {
            return Ok(None);
        };
        Ok(Some(match param.ty.as_str() {
            "float" => match param.as_floats()? {
                [value] => Texturable::Constant(*value as f32),
                _ => {
                    return Err(Error::SceneFormat(format!(
                        "{}: parameter \"{name}\" needs one value",
                        param.location
                    )));
                }
            },
            _ => match self.texture_lookup(param.as_string()?, &param.location)? {
                TextureDef::Constant(value) => Texturable::Constant(value[0]),
                TextureDef::Image {
                    path,
                    color_space,
                    scale,
                } => {
                    self.warn_dropped_scale(scale, name, &param.location);
                    Texturable::Texture(TextureRef { path, color_space })
                }
            },
        }))
    }

    fn texture_lookup(&self, name: &str, location: &str) -> Result<TextureDef> {
        self.named_textures.get(name).cloned().ok_or_else(|| {
            Error::SceneFormat(format!("{location}: texture \"{name}\" was never declared"))
        })
    }

    /// An `imagemap` texture: the filename resolves scene-relative, the
    /// `encoding` override maps onto the schema's color-space field, and
    /// everything cenote can't express (inversion, UV transforms) warns.
    fn imagemap_texture(&mut self, directive: &Directive, name: &str) -> Result<TextureDef> {
        let params = &directive.params;
        let file = params.string("filename")?.ok_or_else(|| {
            Error::SceneFormat(format!(
                "{}: imagemap \"{name}\" has no \"string filename\"",
                directive.location
            ))
        })?;
        let path = self.parser.resolve(file);
        let scale = params.float("scale")?.unwrap_or(1.0);
        let color_space = match params.string("encoding")? {
            None => None,
            Some("linear") => Some(ColorSpace::Linear),
            Some("sRGB") => Some(ColorSpace::Srgb),
            Some(other) => {
                self.warn(format!(
                    "{}: texture encoding \"{other}\" is not supported — the \
                     slot's default is used",
                    directive.location
                ));
                None
            }
        };
        if params.boolean("invert")? == Some(true) {
            self.warn(format!(
                "{}: inverted textures are not supported — \"{name}\" reads direct",
                directive.location
            ));
        }
        let differs = |name: &str, identity: f32| {
            params
                .float(name)
                .ok()
                .flatten()
                .is_some_and(|value| (value - identity).abs() > f32::EPSILON)
        };
        if differs("uscale", 1.0)
            || differs("vscale", 1.0)
            || differs("udelta", 0.0)
            || differs("vdelta", 0.0)
        {
            self.warn(format!(
                "{}: UV transforms are not supported — \"{name}\" samples \
                 authored UVs directly",
                directive.location
            ));
        }
        Ok(TextureDef::Image {
            path,
            color_space,
            scale,
        })
    }

    fn texture_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let name = directive.names[0].clone();
        let class = directive.names[2].clone();
        if self.named_textures.contains_key(&name) {
            return Err(Error::SceneFormat(format!(
                "{}: texture \"{name}\" is defined twice",
                directive.location
            )));
        }
        let params = &directive.params;
        let def = match class.as_str() {
            "imagemap" => self.imagemap_texture(directive, &name)?,
            "constant" => {
                let value = match params.take("value", &["float", "rgb", "color"])? {
                    Some(param) => match param.as_floats()? {
                        [value] => [*value as f32; 3],
                        [r, g, b] => [*r as f32, *g as f32, *b as f32],
                        _ => {
                            return Err(Error::SceneFormat(format!(
                                "{}: constant texture \"{name}\" needs one or three values",
                                param.location
                            )));
                        }
                    },
                    None => [1.0; 3],
                };
                TextureDef::Constant(value)
            }
            "scale" => {
                let inner = match params.take("tex", &["texture", "float", "rgb", "color"])? {
                    Some(param) if param.ty == "texture" => {
                        self.texture_lookup(param.as_string()?, &param.location)?
                    }
                    Some(param) => match param.as_floats()? {
                        [value] => TextureDef::Constant([*value as f32; 3]),
                        [r, g, b] => TextureDef::Constant([*r as f32, *g as f32, *b as f32]),
                        _ => {
                            return Err(Error::SceneFormat(format!(
                                "{}: scale texture \"{name}\" has a malformed \"tex\"",
                                param.location
                            )));
                        }
                    },
                    None => TextureDef::Constant([1.0; 3]),
                };
                let factor = match params.take("scale", &["float", "texture"])? {
                    Some(param) if param.ty == "float" => match param.as_floats()? {
                        [value] => *value as f32,
                        _ => 1.0,
                    },
                    Some(param) => {
                        self.warn(format!(
                            "{}: a textured scale factor is not supported — 1 used",
                            param.location
                        ));
                        1.0
                    }
                    None => 1.0,
                };
                match inner {
                    TextureDef::Constant(value) => {
                        TextureDef::Constant(value.map(|channel| channel * factor))
                    }
                    TextureDef::Image {
                        path,
                        color_space,
                        scale,
                    } => TextureDef::Image {
                        path,
                        color_space,
                        scale: scale * factor,
                    },
                }
            }
            other => {
                self.warn(format!(
                    "{}: texture class \"{other}\" is not supported — \"{name}\" \
                     becomes mid-gray",
                    directive.location
                ));
                TextureDef::Constant([0.5; 3])
            }
        };
        self.warn_unused(directive, &format!("texture \"{name}\""));
        self.named_textures.insert(name, def);
        Ok(())
    }
}

/// A `trianglemesh` shape's streams, verbatim in object space. `flip`
/// (trap 4's XOR) negates authored normals and reverses winding —
/// winding also drives derived normals, so orientation survives either
/// way.
fn trianglemesh(directive: &Directive, flip: bool) -> Result<MeshSource> {
    let params = &directive.params;
    let triples = |name: &str, types: &[&str]| -> Result<Option<Vec<[f32; 3]>>> {
        let Some(param) = params.take(name, types)? else {
            return Ok(None);
        };
        let floats = param.as_floats()?;
        if floats.len() % 3 != 0 {
            return Err(Error::SceneFormat(format!(
                "{}: \"{name}\" needs whole (x, y, z) triples",
                param.location
            )));
        }
        Ok(Some(
            floats
                .chunks_exact(3)
                .map(|triple| [triple[0] as f32, triple[1] as f32, triple[2] as f32])
                .collect(),
        ))
    };
    let positions = triples("P", &["point3", "point"])?.ok_or_else(|| {
        Error::SceneFormat(format!(
            "{}: trianglemesh has no \"point3 P\"",
            directive.location
        ))
    })?;
    let mut normals = triples("N", &["normal", "normal3"])?;
    if flip && let Some(normals) = &mut normals {
        for normal in normals {
            *normal = normal.map(|component| -component);
        }
    }
    let uvs = match params.take("uv", &["point2", "float", "vector2"])? {
        Some(param) => {
            let floats = param.as_floats()?;
            if floats.len() % 2 != 0 {
                return Err(Error::SceneFormat(format!(
                    "{}: \"uv\" needs whole (u, v) pairs",
                    param.location
                )));
            }
            Some(
                floats
                    .chunks_exact(2)
                    .map(|pair| [pair[0] as f32, pair[1] as f32])
                    .collect(),
            )
        }
        None => None,
    };
    let triangles = match params.take("indices", &["integer"])? {
        Some(param) => {
            let values = param.as_floats()?;
            if values.len() % 3 != 0 {
                return Err(Error::SceneFormat(format!(
                    "{}: \"indices\" needs whole triangles",
                    param.location
                )));
            }
            let mut triangles = Vec::with_capacity(values.len() / 3);
            for triple in values.chunks_exact(3) {
                let mut triangle = [0u32; 3];
                for (corner, value) in triangle.iter_mut().zip(triple) {
                    if *value < 0.0 || *value > f64::from(u32::MAX) {
                        return Err(Error::SceneFormat(format!(
                            "{}: index {value} is out of range",
                            param.location
                        )));
                    }
                    *corner = *value as u32;
                }
                triangles.push(triangle);
            }
            triangles
        }
        // pbrt allows exactly one implicit triangle.
        None if positions.len() == 3 => vec![[0, 1, 2]],
        None => {
            return Err(Error::SceneFormat(format!(
                "{}: trianglemesh has no \"integer indices\"",
                directive.location
            )));
        }
    };
    let triangles = if flip {
        triangles.into_iter().map(|[a, b, c]| [a, c, b]).collect()
    } else {
        triangles
    };
    Ok(MeshSource::Inline {
        positions,
        normals,
        uvs,
        triangles,
    })
}

/// A pbrt sphere, tessellated: poles on the object-space z axis,
/// analytic normals, pbrt's parameterization for UVs (`u` around z,
/// `v = 0` at the +z pole). 32 rings × 64 segments keeps silhouettes
/// clean at corpus scales.
fn sphere_mesh(radius: f32) -> MeshSource {
    const RINGS: u32 = 32;
    const SEGMENTS: u32 = 64;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for ring in 0..=RINGS {
        let v = ring as f32 / RINGS as f32;
        let theta = v * std::f32::consts::PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for segment in 0..=SEGMENTS {
            let u = segment as f32 / SEGMENTS as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
            positions.push((normal * radius).into());
            normals.push(normal.into());
            uvs.push([u, v]);
        }
    }
    let mut triangles = Vec::new();
    let row = SEGMENTS + 1;
    for ring in 0..RINGS {
        for segment in 0..SEGMENTS {
            let a = ring * row + segment;
            let b = a + row;
            // The two pole rows collapse to points; their degenerate
            // half of each quad is skipped.
            if ring != 0 {
                triangles.push([a, b, a + 1]);
            }
            if ring != RINGS - 1 {
                triangles.push([a + 1, b, b + 1]);
            }
        }
    }
    MeshSource::Inline {
        positions,
        normals: Some(normals),
        uvs: Some(uvs),
        triangles,
    }
}

/// A pbrt disk: radius `radius` in the plane `z = height`, facing +z,
/// pbrt's radial parameterization (`v = 1` at the center).
fn disk_mesh(radius: f32, height: f32) -> MeshSource {
    const SEGMENTS: u32 = 64;
    let mut positions = vec![[0.0, 0.0, height]];
    let mut uvs = vec![[0.0, 1.0]];
    for segment in 0..=SEGMENTS {
        let u = segment as f32 / SEGMENTS as f32;
        let phi = u * std::f32::consts::TAU;
        positions.push([radius * phi.cos(), radius * phi.sin(), height]);
        uvs.push([u, 0.0]);
    }
    let triangles = (0..SEGMENTS)
        .map(|segment| [0, segment + 1, segment + 2])
        .collect();
    MeshSource::Inline {
        positions,
        normals: Some(vec![[0.0, 0.0, 1.0]; SEGMENTS as usize + 2]),
        uvs: Some(uvs),
        triangles,
    }
}

/// A blackbody's chromaticity as linear `Rec.709`, normalized to
/// luminance 1 — matching pbrt, which normalizes blackbody emitters to
/// 1 nit before its photometric scale (trap 1's blackbody half).
/// Krystek's Planckian-locus approximation in CIE 1960 UCS, accurate to
/// ~1e-3 in chromaticity over 1000–15000 K.
#[expect(
    clippy::many_single_char_names,
    reason = "the CIE variables are named what colorimetry names them"
)]
fn blackbody_rec709(kelvin: f32) -> [f32; 3] {
    let t = f64::from(kelvin).clamp(1000.0, 15000.0);
    let u = (0.860_117_757 + 1.541_182_54e-4 * t + 1.286_412_12e-7 * t * t)
        / (1.0 + 8.424_202_35e-4 * t + 7.081_451_63e-7 * t * t);
    let v = (0.317_398_726 + 4.228_062_45e-5 * t + 4.204_816_91e-8 * t * t)
        / (1.0 - 2.897_418_16e-5 * t + 1.614_560_53e-7 * t * t);
    // CIE 1960 uv → xy → XYZ at Y = 1 → linear Rec.709.
    let x = 3.0 * u / (2.0 * u - 8.0 * v + 4.0);
    let y = 2.0 * v / (2.0 * u - 8.0 * v + 4.0);
    let xyz = [x / y, 1.0, (1.0 - x - y) / y];
    let rgb = [
        3.240_454_2 * xyz[0] - 1.537_138_5 * xyz[1] - 0.498_531_4 * xyz[2],
        -0.969_266_0 * xyz[0] + 1.876_010_8 * xyz[1] + 0.041_556_0 * xyz[2],
        0.055_643_4 * xyz[0] - 0.204_025_9 * xyz[1] + 1.057_225_2 * xyz[2],
    ];
    // Warm temperatures fall outside the Rec.709 gamut: clamp, then
    // restore unit luminance.
    let rgb = rgb.map(|channel| channel.max(0.0) as f32);
    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    rgb.map(|channel| channel / luminance)
}

#[cfg(test)]
mod tests {
    use cenote::scene::description::SceneDescription;

    use super::*;

    /// Write `files` into a fresh fixture directory, import the first,
    /// and return the change-set with its warnings. Generated assets go
    /// to a `generated/` subdirectory that outlives the call only long
    /// enough for `apply` to see them — callers that apply do so inside.
    fn import_files<T>(
        test: &str,
        files: &[(&str, &str)],
        inspect: impl FnOnce(&ChangeSet, &[String]) -> T,
    ) -> T {
        let dir = std::env::temp_dir().join(format!("cenote-map-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        for (name, source) in files {
            std::fs::write(dir.join(name), source).expect("write fixture");
        }
        let imported = crate::import(&dir.join(files[0].0), &dir.join("generated"));
        let result = match &imported {
            Ok(import) => inspect(&import.set, &import.warnings),
            Err(error) => panic!("import failed: {error}"),
        };
        std::fs::remove_dir_all(&dir).ok();
        result
    }

    fn import_world<T>(
        test: &str,
        world: &str,
        inspect: impl FnOnce(&ChangeSet, &[String]) -> T,
    ) -> T {
        let source = format!("WorldBegin\n{world}\n");
        import_files(test, &[("scene.pbrt", &source)], inspect)
    }

    const TRIANGLE: &str = r#"Shape "trianglemesh"
        "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]"#;

    fn material<'a>(set: &'a ChangeSet, name: &str) -> &'a MaterialPatch {
        set.ops
            .iter()
            .find_map(|op| match op {
                Op::Material(patch) if patch.name == name => Some(&**patch),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no material \"{name}\" in {:?}", names(set)))
    }

    fn instances(set: &ChangeSet) -> Vec<&InstancePatch> {
        set.ops
            .iter()
            .filter_map(|op| match op {
                Op::Instance(patch) => Some(patch),
                _ => None,
            })
            .collect()
    }

    fn names(set: &ChangeSet) -> Vec<(&'static str, String)> {
        set.ops
            .iter()
            .map(|op| {
                let (kind, name) = op.target();
                (
                    match kind {
                        cenote::scene::changeset::Kind::Mesh => "mesh",
                        cenote::scene::changeset::Kind::Instance => "instance",
                        cenote::scene::changeset::Kind::Material => "material",
                        cenote::scene::changeset::Kind::Light => "light",
                        cenote::scene::changeset::Kind::Camera => "camera",
                        cenote::scene::changeset::Kind::Environment => "environment",
                        cenote::scene::changeset::Kind::Settings => "settings",
                    },
                    name.to_owned(),
                )
            })
            .collect()
    }

    fn camera(set: &ChangeSet) -> &CameraPatch {
        set.ops
            .iter()
            .find_map(|op| match op {
                Op::Camera(patch) => Some(patch),
                _ => None,
            })
            .expect("a camera op")
    }

    /// Trap 1, RGB half: light values are nit-valued verbatim — pbrt's
    /// photometric division sees only the illuminant, never the RGB
    /// multiplier — with `scale` riding the luminance slot.
    #[test]
    fn rgb_light_values_import_verbatim_as_nits() {
        let world =
            format!("AreaLightSource \"diffuse\" \"rgb L\" [4 2 1] \"float scale\" 3\n{TRIANGLE}");
        import_world("photometric-rgb", &world, |set, _| {
            let glow = material(set, "pbrt-default-glow-0");
            assert_eq!(
                glow.emission_color,
                Some(Texturable::Constant([4.0, 2.0, 1.0]))
            );
            assert_eq!(glow.emission_luminance, Some(3.0));
        });
    }

    /// Trap 1, blackbody half: pbrt *does* normalize blackbody emitters
    /// to 1 nit, so the imported color is a luminance-1 chromaticity —
    /// warm at 3000 K — and `scale` is the luminance.
    #[test]
    fn blackbody_lights_import_luminance_normalized() {
        let world = format!(
            "AreaLightSource \"diffuse\" \"blackbody L\" 3000 \"float scale\" 5\n{TRIANGLE}"
        );
        import_world("photometric-blackbody", &world, |set, _| {
            let glow = material(set, "pbrt-default-glow-0");
            let Some(Texturable::Constant([r, g, b])) = glow.emission_color else {
                panic!("expected a constant emission color");
            };
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            assert!((luminance - 1.0).abs() < 1e-4, "luminance {luminance}");
            assert!(r > g && g > b, "3000 K must be warm: ({r}, {g}, {b})");
            assert_eq!(glow.emission_luminance, Some(5.0));
        });
    }

    /// Trap 1, delta-light corner: "power" on a point light spreads over
    /// the full sphere, exactly pbrt's `φ_v / 4π`.
    #[test]
    fn point_light_power_spreads_over_the_sphere() {
        let tau2 = 4.0 * std::f32::consts::PI;
        let world =
            format!("LightSource \"point\" \"rgb I\" [1 1 1] \"float power\" {tau2}\n{TRIANGLE}");
        import_world("photometric-power", &world, |set, _| {
            let light = set
                .ops
                .iter()
                .find_map(|op| match op {
                    Op::Light(patch) => patch.light.as_ref(),
                    _ => None,
                })
                .expect("a light op");
            let Light::Point { intensity, .. } = light else {
                panic!("expected a point light");
            };
            assert!(
                intensity
                    .iter()
                    .all(|&channel| (channel - 1.0).abs() < 1e-5),
                "{intensity:?}"
            );
        });
    }

    /// Trap 2: pbrt's remapped roughness (α = √r) lands in `OpenPBR`'s
    /// slug (α = r²) as the fourth root; `remaproughness false` means
    /// the value already *is* α, so it imports as the square root.
    #[test]
    fn roughness_remaps_through_the_alpha_conventions() {
        let world = format!(
            "Material \"conductor\" \"float roughness\" 0.0625\n{TRIANGLE}\n\
             Material \"conductor\" \"float roughness\" 0.0625 \"bool remaproughness\" false\n\
             {TRIANGLE}"
        );
        import_world("roughness", &world, |set, _| {
            assert_eq!(
                material(set, "conductor-0").specular_roughness,
                Some(Texturable::Constant(0.5))
            );
            assert_eq!(
                material(set, "conductor-1").specular_roughness,
                Some(Texturable::Constant(0.25))
            );
        });
    }

    /// Trap 3: `fov` spans the shorter image axis. Landscape frames use
    /// it as the vertical fov directly; portrait frames convert.
    #[test]
    fn fov_names_the_shorter_axis() {
        let source = |x: u32, y: u32| {
            format!(
                "Camera \"perspective\" \"float fov\" 60\n\
                 Film \"rgb\" \"integer xresolution\" {x} \"integer yresolution\" {y}\n\
                 WorldBegin\n{TRIANGLE}\n"
            )
        };
        import_files(
            "fov-landscape",
            &[("scene.pbrt", &source(800, 400))],
            |set, _| {
                assert_eq!(camera(set).vfov_degrees, Some(60.0));
            },
        );
        import_files(
            "fov-portrait",
            &[("scene.pbrt", &source(400, 800))],
            |set, _| {
                let vfov = camera(set).vfov_degrees.expect("set");
                let wanted = 2.0
                    * ((60.0f32.to_radians() / 2.0).tan() * 2.0)
                        .atan()
                        .to_degrees();
                assert!((vfov - wanted).abs() < 1e-3, "vfov {vfov}, wanted {wanted}");
            },
        );
    }

    /// Trap 4: pbrt's left-handed `LookAt` imports so that a pbrt-world +x
    /// object lands on the *same side of the screen* under cenote's
    /// right-handed camera basis (`right = forward × up`).
    #[test]
    fn handedness_conjugation_keeps_screen_sides() {
        let source =
            format!("LookAt 0 0 0  0 0 1  0 1 0\nWorldBegin\nTranslate 1 0 0\n{TRIANGLE}\n");
        import_files("handedness", &[("scene.pbrt", &source)], |set, _| {
            let camera = camera(set);
            let position = Vec3::from(camera.position.expect("set"));
            let look_at = Vec3::from(camera.look_at.expect("set"));
            let up = Vec3::from(camera.up.expect("set"));
            // pbrt's +z view direction lands on cenote's −z.
            assert!((look_at - position).abs_diff_eq(-Vec3::Z, 1e-6));

            let transform = instances(set)[0].transform.clone().expect("set");
            let object = transform.to_mat4().transform_point3(Vec3::ZERO);
            // In pbrt, right = up × dir = +x: the object shows on the
            // right of the image. cenote's right = forward × up must
            // agree.
            let forward = (look_at - position).normalize();
            let right = forward.cross(up).normalize();
            assert!(right.dot(object - position) > 0.5, "object at {object}");
        });
    }

    /// Trap 4's other half: a *reflective* camera transform (how
    /// Tungsten-converted scenes encode their handedness fix) must NOT
    /// get the `FLIP_Z` conjugation — pbrt puts camera-space +x on screen
    /// right either way, and the vendored cornell box catches this as a
    /// mirrored image if it regresses.
    #[test]
    fn reflective_camera_transforms_skip_the_conjugation() {
        // Bitterli-style world-to-camera: x kept, z negated (det −1),
        // camera at pbrt-world (0, 0, 5) looking toward −z.
        let source = format!(
            "Transform [1 0 0 0  0 1 0 0  0 0 -1 0  0 0 5 1]
             Camera \"perspective\"
WorldBegin
Translate 1 0 0
{TRIANGLE}
"
        );
        import_files("reflective-camera", &[("scene.pbrt", &source)], |set, _| {
            let camera = camera(set);
            let position = Vec3::from(camera.position.expect("set"));
            let look_at = Vec3::from(camera.look_at.expect("set"));
            let up = Vec3::from(camera.up.expect("set"));
            let transform = instances(set)[0].transform.clone().expect("set");
            let object = transform.to_mat4().transform_point3(Vec3::ZERO);
            // pbrt renders camera-space +x = world +x on screen right;
            // the object at world +1 must land on cenote's right too.
            let forward = (look_at - position).normalize();
            let right = forward.cross(up).normalize();
            assert!(right.dot(object - position) > 0.5, "object at {object}");
        });
    }

    /// Trap 4's XOR: `ReverseOrientation` flips authored normals and
    /// winding; a mirroring transform flips them back.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "authored normal components must copy through bit-exact"
    )]
    fn reverse_orientation_xors_with_mirroring_transforms() {
        let plate = r#"Shape "trianglemesh"
            "point3 P" [0 0 0  1 0 0  0 1 0]
            "normal N" [0 0 1  0 0 1  0 0 1]
            "integer indices" [0 1 2]"#;
        let world = format!(
            "{plate}\n\
             ReverseOrientation\n{plate}\n\
             Scale -1 1 1\n{plate}\n"
        );
        import_world("reverse-orientation", &world, |set, _| {
            let mesh = |name: &str| {
                set.ops
                    .iter()
                    .find_map(|op| match op {
                        Op::Mesh(patch) if patch.name == name => patch.source.as_ref(),
                        _ => None,
                    })
                    .expect("mesh source")
            };
            let normal_z = |source: &MeshSource| match source {
                MeshSource::Inline {
                    normals, triangles, ..
                } => (
                    normals.as_ref().expect("authored normals")[0][2],
                    triangles[0],
                ),
                MeshSource::Ply { .. } => unreachable!(),
            };
            let (plain, plain_winding) = normal_z(mesh("trianglemesh-0"));
            let (reversed, reversed_winding) = normal_z(mesh("trianglemesh-1"));
            let (cancelled, cancelled_winding) = normal_z(mesh("trianglemesh-2"));
            assert_eq!(plain, 1.0);
            assert_eq!(plain_winding, [0, 1, 2]);
            // ReverseOrientation alone flips…
            assert_eq!(reversed, -1.0);
            assert_eq!(reversed_winding, [0, 2, 1]);
            // …and a handedness-swapping transform cancels it.
            assert_eq!(cancelled, 1.0);
            assert_eq!(cancelled_winding, [0, 1, 2]);
        });
    }

    /// Trap 5's integration seam: a constant infinite light becomes a
    /// generated sky EXR carrying scale × L, referenced by the
    /// environment op. (The octahedral resample itself is pinned in
    /// `crate::env`.)
    #[test]
    fn a_constant_infinite_light_becomes_a_sky_file() {
        let world =
            format!("LightSource \"infinite\" \"rgb L\" [2 2 2] \"float scale\" 0.5\n{TRIANGLE}");
        import_world("infinite-constant", &world, |set, _| {
            let path = set
                .ops
                .iter()
                .find_map(|op| match op {
                    Op::Environment(patch) => patch.path.as_ref(),
                    _ => None,
                })
                .expect("an environment op");
            let (_, _, pixels) = cenote::output::read_exr(path).expect("sky reads");
            assert!((pixels[0] - 1.0).abs() < 1e-6, "expected 2 × 0.5 = 1");
        });
    }

    /// Object instancing: pbrt composes the instance-time CTM on top of
    /// each recorded shape's full declaration-time CTM.
    #[test]
    fn object_instances_compose_transforms() {
        let world = format!(
            "ObjectBegin \"tree\"\nTranslate 0 5 0\n{TRIANGLE}\nObjectEnd\n\
             Translate 3 0 0\nObjectInstance \"tree\"\n\
             Translate 0 0 7\nObjectInstance \"tree\"\n"
        );
        import_world("instancing", &world, |set, _| {
            let placed = instances(set);
            assert_eq!(placed.len(), 2);
            let origin = |index: usize| {
                placed[index]
                    .transform
                    .clone()
                    .expect("set")
                    .to_mat4()
                    .transform_point3(Vec3::ZERO)
            };
            assert!(origin(0).abs_diff_eq(Vec3::new(3.0, 5.0, 0.0), 1e-5));
            // The second use composes both translations; pbrt +z lands
            // on cenote −z.
            assert!(origin(1).abs_diff_eq(Vec3::new(3.0, 5.0, -7.0), 1e-5));
            // Both instances share one mesh.
            assert_eq!(placed[0].mesh, placed[1].mesh);
        });
    }

    /// The attribute stack scopes materials and transforms; named
    /// materials and textures resolve; the shared default material is
    /// emitted once; and unsupported tokens surface as warnings.
    #[test]
    fn graphics_state_scopes_and_names_resolve() {
        let world = format!(
            "Texture \"wood\" \"spectrum\" \"imagemap\" \"string filename\" \"wood.png\"\n\
             MakeNamedMaterial \"planks\" \"string type\" \"diffuse\" \
             \"texture reflectance\" \"wood\"\n\
             AttributeBegin\n\
             NamedMaterial \"planks\"\n{TRIANGLE}\n\
             AttributeEnd\n\
             {TRIANGLE}\n\
             Shape \"hyperboloid\"\n"
        );
        import_files(
            "state",
            &[
                ("scene.pbrt", &world_scene(&world)),
                ("wood.png", "not-a-real-png"),
            ],
            |set, warnings| {
                let placed = instances(set);
                assert_eq!(placed.len(), 2);
                assert_eq!(placed[0].material.as_deref(), Some("planks"));
                // Outside the attribute block, back to the default.
                assert_eq!(placed[1].material.as_deref(), Some("pbrt-default"));
                let planks = material(set, "planks");
                match &planks.base_color {
                    Some(Texturable::Texture(reference)) => {
                        assert!(reference.path.is_absolute());
                        assert!(reference.path.ends_with("wood.png"));
                    }
                    other => panic!("expected a texture, got {other:?}"),
                }
                assert!(
                    warnings
                        .iter()
                        .any(|warning| warning.contains("hyperboloid")),
                    "{warnings:?}"
                );
            },
        );
    }

    fn world_scene(world: &str) -> String {
        format!("WorldBegin\n{world}\n")
    }

    /// Spheres and disks tessellate at import: analytic normals, sane
    /// bounds, disks sitting at their height.
    #[test]
    fn spheres_and_disks_tessellate() {
        let world = "Shape \"sphere\" \"float radius\" 2\n\
                     Shape \"disk\" \"float height\" -1 \"float radius\" 3\n";
        import_world("quadrics", world, |set, _| {
            let sources: Vec<&MeshSource> = set
                .ops
                .iter()
                .filter_map(|op| match op {
                    Op::Mesh(patch) => patch.source.as_ref(),
                    _ => None,
                })
                .collect();
            let MeshSource::Inline {
                positions, normals, ..
            } = sources[0]
            else {
                panic!("sphere is inline");
            };
            assert!(
                positions
                    .iter()
                    .all(|position| { (Vec3::from(*position).length() - 2.0).abs() < 1e-4 })
            );
            let authored = normals.as_ref().expect("analytic normals");
            assert!(
                authored
                    .iter()
                    .all(|normal| { (Vec3::from(*normal).length() - 1.0).abs() < 1e-4 })
            );
            let MeshSource::Inline { positions, .. } = sources[1] else {
                panic!("disk is inline");
            };
            assert!(
                positions
                    .iter()
                    .all(|position| (position[2] + 1.0).abs() < 1e-6)
            );
        });
    }

    /// The end-to-end contract: an imported scene *applies* — every
    /// reference resolves, every path is absolute and exists, and the
    /// singletons (camera, settings) are in place.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "camera parameters must copy through bit-exact — no arithmetic is involved"
    )]
    fn an_imported_scene_applies_cleanly() {
        let source = "\
            LookAt 3 4 1.5  .5 .5 0  0 0 1\n\
            Camera \"perspective\" \"float fov\" 45 \
            \"float lensradius\" 0.05 \"float focaldistance\" 4\n\
            Film \"rgb\" \"integer xresolution\" 320 \"integer yresolution\" 180\n\
            Sampler \"zsobol\" \"integer pixelsamples\" 32\n\
            Integrator \"volpath\" \"integer maxdepth\" 7\n\
            WorldBegin\n\
            LightSource \"infinite\" \"rgb L\" [0.4 0.45 0.5]\n\
            LightSource \"distant\" \"rgb L\" [3 3 3] \"point3 from\" [0 0 10]\n\
            Material \"coateddiffuse\" \"rgb reflectance\" [0.7 0.2 0.1] \
            \"float roughness\" 0.1\n\
            Shape \"sphere\" \"float radius\" 1\n\
            AreaLightSource \"diffuse\" \"rgb L\" [8 7 6]\n\
            Translate 0 0 5\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]\n";
        import_files("applies", &[("scene.pbrt", source)], |set, warnings| {
            let mut description = SceneDescription::new();
            description.apply(set).expect("the imported set applies");
            assert_eq!(description.cameras().len(), 1);
            assert_eq!(description.settings()["main"].resolution, [320, 180]);
            assert_eq!(description.settings()["main"].spp, 32);
            assert_eq!(description.settings()["main"].max_bounces, 7);
            assert_eq!(description.instances().len(), 2);
            assert_eq!(description.lights().len(), 1);
            assert_eq!(description.environments().len(), 1);
            let camera = &description.cameras()["main"];
            assert_eq!(camera.aperture_radius, 0.05);
            assert_eq!(camera.focus_distance, Some(4.0));
            // One warning expected: the one-sided area light summary.
            assert!(
                warnings.iter().any(|warning| warning.contains("one-sided")),
                "{warnings:?}"
            );
        });
    }
}
