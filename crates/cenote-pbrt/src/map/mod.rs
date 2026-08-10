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
//!    `CTM` flips authored normals and winding, per pbrt's rule. cenote
//!    emitters are one-sided like pbrt's default, so the honest divergence
//!    is the other way: a pbrt `twosided` area light is warned, once, with
//!    a count.
//! 5. **Octahedral skies** resample to cenote's equirect at import
//!    ([`crate::env`]), orientation and photometric scale baked in.
//!
//! Everything outside the supported subset warns **by token name** —
//! every directive, shape, material, texture class, or parameter this
//! importer drops is named in the warning list, so silence always means
//! "handled".

mod color;
mod shape;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cenote::scene::changeset::{
    CameraPatch, ChangeSet, EnvironmentPatch, InstancePatch, LightPatch, MaterialPatch, MeshPatch,
    Op, SettingsPatch,
};
use cenote::scene::description::{
    Channel, ColorSpace, Light, MeshSource, Texturable, TextureRef, Transform, UvTransform,
};
use cenote::scene::multiple_scatter_color;
use cenote::{Error, Result};
use glam::{Mat3, Mat4, Vec3};

use color::{blackbody_rec709, conductor_f0, named_metal_f0};
use shape::{bilinearmesh, disk_mesh, sphere_mesh, trianglemesh};

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
        /// pbrt `scale` folded onto the texture — carried onto the
        /// schema's per-reference multiplier (emission folds it into the
        /// luminance instead, trap 1).
        scale: f32,
        /// pbrt's uscale/vscale/udelta/vdelta, in pbrt's v-up convention;
        /// [`Mapper::texture_ref`] flips the v leg into storage space.
        uv: Option<UvAffine>,
    },
    Constant([f32; 3]),
}

/// pbrt keeps `float` and `spectrum` textures in two disjoint namespaces,
/// so one name may denote a `float` bump in one file and a `spectrum`
/// reflectance in another — sanmiguel's `Map #483` does exactly this, a
/// name collision between two independently authored include files that
/// only renders correctly because pbrt never conflates the two. We key the
/// named-texture map by kind so they never collide; a lookup prefers its
/// slot's kind and falls back to the other only when the preferred
/// namespace has no such name (pbrt's coercion when a slot borrows across).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum TexKind {
    Float,
    Spectrum,
}

impl TexKind {
    /// pbrt's texture type word: `float` is scalar, everything else
    /// (`spectrum`, the legacy `color`) is spectral.
    fn from_type(word: &str) -> Self {
        if word == "float" {
            Self::Float
        } else {
            Self::Spectrum
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Float => Self::Spectrum,
            Self::Spectrum => Self::Float,
        }
    }
}

/// An imagemap's affine UV remap as pbrt spells it (`st' = st * scale +
/// delta` with `t` up); cenote stores v down, so the conversion lives at
/// the same boundary the UV flip itself does.
#[derive(Clone, Copy)]
struct UvAffine {
    scale: [f32; 2],
    delta: [f32; 2],
}

/// The schema reference for an image texture definition. The value
/// multiplier rides along verbatim; the affine's v leg converts from
/// pbrt's v-up into storage order — with `v_c = 1 − v_p` on both sides of
/// the remap, `v_c' = 1 − (v_p·s + d)` lands at scale `s`, offset
/// `1 − s − d` (the same boundary flip `shape.rs` applies to authored
/// UVs, here applied to the transform instead of the coordinates).
fn texture_ref(
    path: PathBuf,
    color_space: Option<ColorSpace>,
    scale: f32,
    uv: Option<UvAffine>,
) -> TextureRef {
    TextureRef {
        path,
        color_space,
        channel: None,
        scale: ((scale - 1.0).abs() > f32::EPSILON).then_some(scale),
        uv: uv.map(|affine| UvTransform {
            scale: affine.scale,
            offset: [
                affine.delta[0],
                1.0 - affine.scale[1] - affine.delta[1],
            ],
        }),
    }
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
    named_textures: BTreeMap<(TexKind, String), TextureDef>,
    named_materials: BTreeMap<String, MaterialPatch>,
    /// Finished `ObjectBegin` recordings, by object name.
    objects: BTreeMap<String, Vec<RecordedShape>>,
    /// The recording in progress, if any.
    active_object: Option<(String, Vec<RecordedShape>)>,
    /// Materials forked to carry a shape `alpha` cutout, keyed by base
    /// material name × mask identity — shapes pairing the same material
    /// with the same mask share one fork. The patch rides along so a
    /// pending area light can fork the fork.
    cutout_forks: BTreeMap<(String, String), (String, MaterialPatch)>,
    counters: BTreeMap<String, u32>,
    /// The world-space conjugation every emitted transform passes
    /// through: [`FLIP_Z`], or the identity for reflective-camera scenes
    /// (see [`FLIP_Z`]'s doc). Chosen at `WorldBegin`.
    conjugation: Mat4,
    environment_emitted: bool,
    two_sided_lights: u32,
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
            cutout_forks: BTreeMap::new(),
            counters: BTreeMap::new(),
            conjugation: FLIP_Z,
            environment_emitted: false,
            two_sided_lights: 0,
            default_material_emitted: false,
        }
    }

    /// Walk the whole directive stream and close out the change-set.
    pub fn run(mut self) -> Result<(ChangeSet, Vec<String>)> {
        // pbrt-v4 creates every named texture and material only once the
        // world block is fully parsed, so a material may legally name a
        // texture declared *later* in the file — watercolor does exactly
        // this. We honor that deferral by buffering the directive stream
        // and resolving every `Texture` in a first pass, before any
        // material or shape can look one up. The buffer is bounded by the
        // scene's source text; the heavy mesh data lives in external PLY
        // files referenced by path, so holding the directives costs little.
        let mut directives = Vec::new();
        while let Some(directive) = self.parser.next_directive()? {
            directives.push(directive);
        }
        for directive in &directives {
            if directive.keyword == "Texture" {
                self.define_texture(directive)?;
            }
        }
        for directive in &directives {
            self.dispatch(directive)?;
        }
        if !self.in_world {
            return Err(Error::SceneFormat(
                "the scene never reaches WorldBegin — not a pbrt scene?".into(),
            ));
        }
        if self.two_sided_lights > 0 {
            self.warnings.push(format!(
                "{} area light(s) are two-sided in pbrt; cenote emitters are one-sided, so \
                 their back faces stay dark",
                self.two_sided_lights
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
            // Resolved in the pre-pass (see `run`); here only the block
            // rule is enforced, in source order with the rest of the scene.
            "Texture" => self.verify_block(directive, true)?,
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
        // Trap 4's per-scene half (see FLIP_Z for the why): a proper camera
        // rotation gets the conjugation, a reflective one — already
        // handedness-fixed — gets the identity.
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

/// What one of pbrt's interior descriptions yields: the color light walks
/// to — absent where it was textured, and the schema's default stands —
/// with the mean free path and its per-channel shape.
type Interior = (Option<[f32; 3]>, f32, [f32; 3]);

/// A per-channel mean free path onto `OpenPBR`'s scalar
/// `subsurface_radius` and its per-channel scale: the longest channel takes
/// the radius, so the shape lands in [0, 1] and a channel that travels
/// nowhere keeps an exact zero. `None` where no channel travels at all, or
/// where one is not a finite length — there is no interior to describe, and
/// the caller warns rather than authoring a degenerate one.
fn mean_free_path(mfp: [f32; 3]) -> Option<(f32, [f32; 3])> {
    let radius = mfp.iter().copied().fold(0.0f32, f32::max);
    (radius > 0.0 && mfp.iter().all(|channel| channel.is_finite()))
        .then(|| (radius, mfp.map(|channel| (channel / radius).clamp(0.0, 1.0))))
}

/// Whether a transform mirrors — half of the trap-4 XOR.
fn swaps_handedness(matrix: Mat4) -> bool {
    Mat3::from_mat4(matrix).determinant() < 0.0
}

/// Whether a mask image carries an alpha channel, by a bounded header
/// sniff: a PNG's IHDR color type (4 and 6 have alpha; palette images
/// can smuggle alpha through a `tRNS` chunk this never sees, but cutout
/// masks aren't authored that way), or a TGA's pixel depth (32-bit
/// carries alpha; TGA has no magic bytes, so like the decoder it is
/// extension-hinted). `None` when the file can't be read or the format
/// can't be told.
fn mask_has_alpha(path: &Path) -> Option<bool> {
    use std::io::Read;
    const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let mut header = [0u8; 26];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .ok()?;
    match extension.as_str() {
        "png" if header[..8] == PNG_SIGNATURE && &header[12..16] == b"IHDR" => {
            Some(matches!(header[25], 4 | 6))
        }
        "tga" => Some(header[16] == 32),
        _ => None,
    }
}

impl Mapper {
    fn shape_directive(&mut self, directive: &Directive) -> Result<()> {
        self.verify_block(directive, true)?;
        let ty = directive.names[0].as_str();
        // Trap 4, resolved on the renderer's terms: pbrt XORs
        // ReverseOrientation with the CTM's handedness because it bakes
        // world-space vertices and must cancel the mirror out of their
        // winding. cenote keeps object-space geometry and det-corrects the
        // emission side under the instance transform itself (both the hit
        // side's inverse-transpose normal and the light records' winding
        // sign), so a mirroring CTM needs no compensation here — only
        // ReverseOrientation genuinely flips the authored orientation.
        let flip = self.state.reverse_orientation;
        let (source, mesh_prefix) = match ty {
            "trianglemesh" => (trianglemesh(directive, flip)?, "trianglemesh".to_owned()),
            "bilinearmesh" => match bilinearmesh(directive, flip)? {
                Some(source) => (source, "bilinearmesh".to_owned()),
                None => {
                    self.warn(format!(
                        "{}: only single-patch bilinearmeshes are supported — skipped",
                        directive.location
                    ));
                    return Ok(());
                }
            },
            "plymesh" => {
                let file = directive.params.string("filename")?.ok_or_else(|| {
                    Error::SceneFormat(format!(
                        "{}: plymesh has no \"string filename\"",
                        directive.location
                    ))
                })?;
                let path = self.parser.resolve(file);
                // Under two-sided shading the flip is invisible for a
                // reflective mesh, so it only costs anything when the mesh
                // emits: emission is one-sided (winding-front), and a PLY's
                // winding can't be reversed at import. Warn only there.
                if flip && self.state.area_light.is_some() {
                    self.warn(format!(
                        "{}: ReverseOrientation on an emissive plymesh is ignored — it \
                         emits from the winding-front face, which may be the wrong side",
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
        let material = self.shape_material(directive)?;
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
                transforms: Some(vec![matrix_transform(self.conjugation * self.state.ctm)]),
                ..InstancePatch::new(mesh)
            }));
        }
        Ok(())
    }

    /// The material this shape wears: the current one, pbrt's implicit
    /// default diffuse, or a fork of it — first for a shape `alpha`
    /// cutout, then — under a pending area light — for the emission
    /// (each shared across shapes that repeat the combination).
    fn shape_material(&mut self, directive: &Directive) -> Result<String> {
        let location = &directive.location;
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
        let (base_name, base_patch) = self.cutout_fork(directive, base_name, base_patch)?;
        let Some(mut area) = self.state.area_light.clone() else {
            return Ok(base_name);
        };
        if self.active_object.is_some() {
            // pbrt's own limitation, kept: instanced emitters would need
            // per-instance light-table entries it (and this) can't make.
            self.warn(format!(
                "{location}: area lights are not supported with object instancing — \
                 the shape imports unlit"
            ));
            return Ok(base_name);
        }
        if let Some(fork) = area.forks.get(&base_name) {
            return Ok(fork.clone());
        }
        let fork = self.fresh(&format!("{base_name}-glow"));
        let mut patch = base_patch;
        patch.name.clone_from(&fork);
        patch.emission_color = Some(area.color.clone());
        patch.emission_luminance = Some(area.luminance);
        self.ops.push(Op::Material(Box::new(patch)));
        if area.two_sided {
            self.two_sided_lights += 1;
        }
        area.forks.insert(base_name, fork.clone());
        self.state.area_light = Some(area);
        Ok(fork)
    }

    /// A shape's `alpha` — pbrt puts the cutout mask on the shape, the
    /// schema puts opacity on the material — lands as a fork of the
    /// shape's material with the mask on `geometry_opacity`, where the
    /// traversal kernels test it per crossing. A fully-opaque alpha (the
    /// default, spelled out) forks nothing.
    fn cutout_fork(
        &mut self,
        directive: &Directive,
        base_name: String,
        base_patch: MaterialPatch,
    ) -> Result<(String, MaterialPatch)> {
        let Some(param) = directive.params.take("alpha", &["float", "texture"])? else {
            return Ok((base_name, base_patch));
        };
        let (mask, opacity) = if param.ty == "float" {
            let value = param.as_scalar()?;
            if value >= 1.0 {
                return Ok((base_name, base_patch));
            }
            (format!("={value}"), Texturable::Constant(value))
        } else {
            let texture = param.as_string()?;
            match self.texture_lookup(texture, TexKind::Float, &param.location)? {
                TextureDef::Constant(value) if value[0] >= 1.0 => {
                    return Ok((base_name, base_patch));
                }
                TextureDef::Constant(value) => (texture.to_owned(), Texturable::Constant(value[0])),
                TextureDef::Image {
                    path,
                    color_space,
                    scale,
                    uv,
                } => {
                    let channel = self.mask_channel(&path, &param.location);
                    let mut reference = texture_ref(path, color_space, scale, uv);
                    reference.channel = channel;
                    (texture.to_owned(), Texturable::Texture(reference))
                }
            }
        };
        let key = (base_name.clone(), mask);
        if let Some((name, patch)) = self.cutout_forks.get(&key) {
            return Ok((name.clone(), patch.clone()));
        }
        let fork = self.fresh(&format!("{base_name}-cutout"));
        let mut patch = base_patch;
        patch.name.clone_from(&fork);
        patch.geometry_opacity = Some(opacity);
        self.ops.push(Op::Material(Box::new(patch.clone())));
        self.cutout_forks.insert(key, (fork.clone(), patch.clone()));
        Ok((fork, patch))
    }

    /// The channel a mask image feeds a scalar slot from. pbrt reads a
    /// float imagemap through the image's alpha channel when it has a
    /// meaningful one and averages RGB otherwise; the schema's scalar
    /// slots read a single channel, so an alpha-carrying PNG maps to
    /// `A` and everything else to the red default — exact for the
    /// grayscale masks the average case is in practice.
    fn mask_channel(&mut self, path: &Path, location: &str) -> Option<Channel> {
        match mask_has_alpha(path) {
            Some(true) => Some(Channel::A),
            Some(false) => None,
            None => {
                self.warn(format!(
                    "{location}: \"{}\" could not be probed for an alpha channel — \
                     the mask reads the red channel",
                    path.display()
                ));
                None
            }
        }
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
                    channel: None,
                    scale: None,
                    uv: None,
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
            "rgb" | "color" => param.as_rgb()?,
            "blackbody" => blackbody_rec709(param.as_scalar()?),
            "float" => [param.as_scalar()?; 3],
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
            path: Some(Some(out)),
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
                transforms: Some(vec![matrix_transform(
                    self.conjugation * self.state.ctm * shape.ctm,
                )]),
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
        let name = directive.names[0].clone();
        if let Some(patch) = self.named_materials.get(&name).cloned() {
            self.state.material = Some((name, patch));
        } else {
            // pbrt treats a reference to an undefined named material as a
            // warning, not a fatal error, and falls back to the default
            // material — watercolor names "BG", whose `MakeNamedMaterial`
            // is commented out. Match that leniency: drop to the default
            // (`state.material = None`) so the shapes in scope still render.
            self.warn(format!(
                "{}: NamedMaterial \"{name}\" was never made — the default material is used",
                directive.location
            ));
            self.state.material = None;
        }
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
                patch.coat_ior = Some(self.dielectric_eta(directive, "", 1.5)?);
                patch.coat_roughness = Some(self.coat_roughness(directive, "")?);
            }
            // A dielectric coat over a conductor base — the same clear coat
            // `coateddiffuse` wears, this time over the metal lobe
            // (`metalness = 1`). pbrt names the two interfaces apart: the
            // conductor reads `conductor.*`, the coat `interface.*`. The
            // coat's scattering medium (`thickness`, `albedo`, `g`) has no
            // analogue in OpenPBR's clear coat, so it falls to the
            // unused-parameter backstop.
            "coatedconductor" => {
                patch.base_metalness = Some(Texturable::Constant(1.0));
                patch.base_color = Some(self.conductor_color(directive, "conductor.")?);
                patch.specular_roughness = Some(self.roughness_slot(directive, "conductor.")?);
                patch.coat_weight = Some(1.0);
                patch.coat_ior = Some(self.dielectric_eta(directive, "interface.", 1.5)?);
                patch.coat_roughness = Some(self.coat_roughness(directive, "interface.")?);
            }
            // `metal` is pbrt-v3's name for what v4 calls `conductor`, with
            // the same `eta`/`k`/`roughness` parameters — import it as one.
            "conductor" | "metal" => {
                patch.base_metalness = Some(Texturable::Constant(1.0));
                patch.base_color = Some(self.conductor_color(directive, "")?);
                patch.specular_roughness = Some(self.roughness_slot(directive, "")?);
            }
            "dielectric" | "thindielectric" => {
                patch.transmission_weight = Some(1.0);
                patch.specular_ior = Some(self.dielectric_eta(directive, "", 1.5)?);
                if ty == "thindielectric" {
                    patch.geometry_thin_walled = Some(true);
                } else {
                    patch.specular_roughness = Some(self.roughness_slot(directive, "")?);
                }
            }
            // Lambertian reflection *and* transmission. OpenPBR's only
            // transmission is specular glass, so import the reflective half
            // as an ordinary diffuse — the dominant look for the fabric and
            // paper that use it — and drop the transmitted half with a
            // warning.
            "diffusetransmission" => {
                patch.base_color = Some(
                    self.color_slot(directive, "reflectance")?
                        .unwrap_or(Texturable::Constant([0.25; 3])),
                );
                patch.specular_weight = Some(0.0);
                if directive
                    .params
                    .take(
                        "transmittance",
                        &["rgb", "color", "float", "spectrum", "texture"],
                    )?
                    .is_some()
                {
                    self.warn(format!(
                        "{}: diffuse transmission has no OpenPBR lobe — imported as \
                         opaque diffuse, the transmittance dropped",
                        directive.location
                    ));
                }
            }
            // A BSSRDF under a dielectric interface — `OpenPBR`'s
            // subsurface lobe under its specular one, weight 1. The
            // interior itself is described three mutually exclusive ways,
            // and pbrt picks between them in that order; see
            // [`Self::subsurface_interior`].
            "subsurface" => {
                patch.subsurface_weight = Some(1.0);
                patch.specular_ior = Some(self.dielectric_eta(directive, "", 1.33)?);
                patch.specular_roughness = Some(self.roughness_slot(directive, "")?);
                let g = params.float("g")?.unwrap_or(0.0);
                patch.subsurface_scatter_anisotropy = Some(g);
                if let Some((color, radius, scale)) = self.subsurface_interior(directive, g)? {
                    patch.subsurface_color = color;
                    patch.subsurface_radius = Some(radius);
                    patch.subsurface_radius_scale = Some(scale);
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
                channel: None,
                scale: None,
                uv: None,
            }));
        }
        Ok(patch)
    }

    /// A conductor's F0: `reflectance` verbatim, `eta`/`k` through the
    /// normal-incidence formula (RGB values) or the named-metal table,
    /// copper — pbrt's own default — when nothing usable is given.
    fn conductor_color(
        &mut self,
        directive: &Directive,
        prefix: &str,
    ) -> Result<Texturable<[f32; 3]>> {
        let params = &directive.params;
        // pbrt namespaces a coated conductor's `eta`/`k` as `conductor.*`,
        // but `reflectance` stays bare in both the plain and coated forms.
        let eta_name = format!("{prefix}eta");
        let k_name = format!("{prefix}k");
        if let Some(reflectance) = self.color_slot(directive, "reflectance")? {
            // pbrt refuses reflectance and eta/k together.
            if params
                .take(&eta_name, &["rgb", "color", "spectrum", "float", "texture"])?
                .is_some()
                || params
                    .take(&k_name, &["rgb", "color", "spectrum", "float", "texture"])?
                    .is_some()
            {
                return Err(Error::SceneFormat(format!(
                    "{}: conductor has both \"reflectance\" and \"eta\"/\"k\"",
                    directive.location
                )));
            }
            return Ok(reflectance);
        }
        let eta = params.take(&eta_name, &["rgb", "color", "spectrum", "float", "texture"])?;
        let k = params.take(&k_name, &["rgb", "color", "spectrum", "float", "texture"])?;
        let rgb_of = |param: Option<&crate::parse::Param>| -> Option<[f32; 3]> {
            let param = param?;
            if !matches!(param.ty.as_str(), "rgb" | "color") {
                return None;
            }
            param.as_rgb().ok()
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

    /// pbrt's three mutually exclusive descriptions of a subsurface
    /// interior, tried in pbrt's own order, onto `OpenPBR`'s color, radius,
    /// and radius scale. `None` leaves the schema defaults standing; a
    /// `None` *color* inside a `Some` keeps the mean free path while
    /// leaving the color alone.
    ///
    /// pbrt names an interior by its coefficients; `OpenPBR` names one by
    /// the color light walks to and the distance it walks between events.
    /// Those are the two ends of the same fit, so the conversion is closed
    /// form: `α` = `σ_s`/`σ_t` back through [`multiple_scatter_color`] for
    /// the color, 1/`σ_t` for the distance. `reflectance` and `mfp` are
    /// already stated at cenote's end and need no conversion at all.
    ///
    /// Two things do not convert, and both warn by token name:
    ///
    /// * **`name`** reads a table of measured media compiled into pbrt.
    ///   cenote carries no such table, and inventing coefficients for
    ///   "skin1" would be a different medium wearing the name.
    /// * **a textured `reflectance`** — `OpenPBR`'s subsurface color is a
    ///   constant slot, so the mean free path imports and the texture is
    ///   named as dropped.
    ///
    /// The divergence worth knowing about is in the `reflectance` path:
    /// pbrt inverts a tabulated photon-beam-diffusion solution to reach the
    /// stated reflectance, cenote inverts van de Hulst's fit. Same
    /// quantity, different curve — so the color imports verbatim and the
    /// two renderers still disagree slightly on what it means.
    fn subsurface_interior(
        &mut self,
        directive: &Directive,
        g: f32,
    ) -> Result<Option<Interior>> {
        let params = &directive.params;
        // pbrt's `scale` multiplies whichever description wins — the
        // coefficients directly, the mean free path as a length.
        let scale = params.float("scale")?.unwrap_or(1.0);
        if let Some(name) = params.string("name")? {
            self.warn(format!(
                "{}: subsurface \"name\" \"{name}\" names one of pbrt's measured media, \
                 a table cenote does not carry — the interior imports at the defaults",
                directive.location
            ));
            return Ok(None);
        }
        if let Some(reflectance) = self.color_slot(directive, "reflectance")? {
            let color = match reflectance {
                Texturable::Constant(color) => Some(color),
                Texturable::Texture(_) => {
                    self.warn(format!(
                        "{}: subsurface \"reflectance\" cannot be textured — the mean free \
                         path imports and the texture is dropped",
                        directive.location
                    ));
                    None
                }
            };
            let mfp = match self.color_slot(directive, "mfp")? {
                Some(Texturable::Constant(mfp)) => mfp,
                Some(Texturable::Texture(_)) => {
                    self.warn(format!(
                        "{}: subsurface \"mfp\" cannot be textured — pbrt's default 1 used",
                        directive.location
                    ));
                    [1.0; 3]
                }
                None => [1.0; 3],
            };
            let Some((radius, shape)) = mean_free_path(mfp.map(|channel| channel * scale)) else {
                self.warn(format!(
                    "{}: subsurface \"mfp\" {mfp:?} is not a length — the interior imports \
                     at the defaults",
                    directive.location
                ));
                return Ok(None);
            };
            return Ok(Some((color, radius, shape)));
        }
        // pbrt's own fallback, its defaults included: a marble, in mm⁻¹.
        let sigma_a = self.coefficients(directive, "sigma_a", [0.0011, 0.0024, 0.014])?;
        let sigma_s = self.coefficients(directive, "sigma_s", [2.55, 3.21, 3.77])?;
        let mut sigma_t = [0.0f32; 3];
        let mut color = [0.0f32; 3];
        for channel in 0..3 {
            sigma_t[channel] = scale * (sigma_a[channel] + sigma_s[channel]);
            // `scale` cancels out of the albedo, so it is read off the
            // authored pair — the accurate side of the same quotient.
            let total = sigma_a[channel] + sigma_s[channel];
            color[channel] = multiple_scatter_color(sigma_s[channel] / total, g);
        }
        let Some((radius, shape)) = mean_free_path(sigma_t.map(|sigma| 1.0 / sigma)) else {
            self.warn(format!(
                "{}: subsurface \"sigma_a\" + \"sigma_s\" extinguishes nothing — the \
                 interior imports at the defaults",
                directive.location
            ));
            return Ok(None);
        };
        Ok(Some((Some(color), radius, shape)))
    }

    /// A per-channel extinction or scattering coefficient: RGB, a
    /// broadcast float, or pbrt's default where the slot is absent. A
    /// named or tabulated spectrum warns and takes the default — the
    /// coefficients are the one place where guessing a curve would change
    /// the medium rather than the shading.
    fn coefficients(
        &mut self,
        directive: &Directive,
        name: &str,
        default: [f32; 3],
    ) -> Result<[f32; 3]> {
        let Some(param) = directive
            .params
            .take(name, &["rgb", "color", "float", "spectrum"])?
        else {
            return Ok(default);
        };
        if matches!(param.ty.as_str(), "rgb" | "color" | "float") {
            return param.as_rgb_broadcast();
        }
        self.warn(format!(
            "{}: a spectral \"{name}\" is not supported — pbrt's default {default:?} used",
            param.location
        ));
        Ok(default)
    }

    /// The dielectric IOR: a float (or float-typed spectrum degenerates
    /// with a warning). pbrt's parameter name is `eta`; its default is the
    /// material's own — 1.5 for glass and the coats, 1.33 for `subsurface`,
    /// which pbrt defaults to skin.
    fn dielectric_eta(&mut self, directive: &Directive, prefix: &str, default: f32) -> Result<f32> {
        let Some(param) = directive
            .params
            .take(&format!("{prefix}eta"), &["float", "spectrum", "rgb", "color"])?
        else {
            return Ok(default);
        };
        if param.ty == "float"
            && let [eta] = param.as_floats()?
        {
            return Ok(*eta as f32);
        }
        self.warn(format!(
            "{}: a spectral IOR (dispersion) is not supported — {default} used",
            param.location
        ));
        Ok(default)
    }

    /// Trap 2: pbrt roughness → the `OpenPBR` slug. Under the default
    /// `remaproughness`, pbrt's α is `√roughness` and `OpenPBR`'s is
    /// `roughness²`, so the value imports as the fourth root (square
    /// root when remapping is off). The curve can't ride a texture, so
    /// textured roughness imports as-is with a warning.
    fn roughness_slot(&mut self, directive: &Directive, prefix: &str) -> Result<Texturable<f32>> {
        let params = &directive.params;
        let remap = params.boolean("remaproughness")?.unwrap_or(true);
        let mut roughness = self.float_slot(directive, &format!("{prefix}roughness"))?;
        let anisotropic: Vec<Texturable<f32>> =
            [format!("{prefix}uroughness"), format!("{prefix}vroughness")]
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

    /// The coat's GGX roughness as a scalar. `OpenPBR`'s coat lobe can't
    /// carry a texture, so a textured slot imports smooth with a warning.
    /// `prefix` picks the parameter namespace: `""` for `coateddiffuse`'s
    /// bare `roughness`, `"interface."` for `coatedconductor`'s coat.
    fn coat_roughness(&mut self, directive: &Directive, prefix: &str) -> Result<f32> {
        Ok(match self.roughness_slot(directive, prefix)? {
            Texturable::Constant(roughness) => roughness,
            Texturable::Texture(_) => {
                self.warn(format!(
                    "{}: coat roughness cannot be textured — the coat imports smooth",
                    directive.location
                ));
                0.0
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
            "rgb" | "color" => Texturable::Constant(param.as_rgb()?),
            "float" => Texturable::Constant([param.as_scalar()?; 3]),
            "texture" => match self.texture_lookup(param.as_string()?, TexKind::Spectrum, &param.location)? {
                TextureDef::Constant(value) => Texturable::Constant(value),
                TextureDef::Image {
                    path,
                    color_space,
                    scale,
                    uv,
                } => Texturable::Texture(texture_ref(path, color_space, scale, uv)),
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

    /// A scalar material slot: float constant or texture.
    fn float_slot(&mut self, directive: &Directive, name: &str) -> Result<Option<Texturable<f32>>> {
        let Some(param) = directive.params.take(name, &["float", "texture"])? else {
            return Ok(None);
        };
        Ok(Some(match param.ty.as_str() {
            "float" => Texturable::Constant(param.as_scalar()?),
            _ => match self.texture_lookup(param.as_string()?, TexKind::Float, &param.location)? {
                TextureDef::Constant(value) => Texturable::Constant(value[0]),
                TextureDef::Image {
                    path,
                    color_space,
                    scale,
                    uv,
                } => Texturable::Texture(texture_ref(path, color_space, scale, uv)),
            },
        }))
    }

    /// Resolve a texture reference from the slot's expected `kind`,
    /// falling back to the other namespace when the preferred one has no
    /// such name (a slot may legitimately borrow across, e.g. a float
    /// bump referenced where the scene declared only a spectrum twin).
    fn texture_lookup(&self, name: &str, kind: TexKind, location: &str) -> Result<TextureDef> {
        self.named_textures
            .get(&(kind, name.to_owned()))
            .or_else(|| self.named_textures.get(&(kind.other(), name.to_owned())))
            .cloned()
            .ok_or_else(|| {
                Error::SceneFormat(format!("{location}: texture \"{name}\" was never declared"))
            })
    }

    /// An `imagemap` texture: the filename resolves scene-relative, the
    /// `encoding` override maps onto the schema's color-space field, the
    /// value scale and affine UV remap ride onto the reference, and what
    /// cenote can't express (inversion) warns.
    #[expect(
        clippy::similar_names,
        reason = "uscale/vscale/udelta/vdelta are pbrt's own parameter names"
    )]
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
        // All four are read unconditionally so a mistyped one errors like
        // every other slot and none is left to spuriously warn as unused.
        let uscale = params.float("uscale")?.unwrap_or(1.0);
        let vscale = params.float("vscale")?.unwrap_or(1.0);
        let udelta = params.float("udelta")?.unwrap_or(0.0);
        let vdelta = params.float("vdelta")?.unwrap_or(0.0);
        let identity = [
            (uscale, 1.0),
            (vscale, 1.0),
            (udelta, 0.0),
            (vdelta, 0.0),
        ]
        .iter()
        .all(|(value, identity)| (value - identity).abs() <= f32::EPSILON);
        let uv = (!identity).then_some(UvAffine {
            scale: [uscale, vscale],
            delta: [udelta, vdelta],
        });
        Ok(TextureDef::Image {
            path,
            color_space,
            scale,
            uv,
        })
    }

    /// Resolve one `Texture` directive into the flat named-texture map.
    /// Called from the pre-pass in [`run`], where the world block hasn't
    /// been entered yet — the block rule is enforced separately, when
    /// `dispatch` reaches the directive in source order.
    fn define_texture(&mut self, directive: &Directive) -> Result<()> {
        let name = directive.names[0].clone();
        let kind = TexKind::from_type(&directive.names[1]);
        let class = directive.names[2].clone();
        let params = &directive.params;
        let def = match class.as_str() {
            "imagemap" => self.imagemap_texture(directive, &name)?,
            "constant" => {
                let value = match params.take("value", &["float", "rgb", "color"])? {
                    Some(param) => param.as_rgb_broadcast()?,
                    None => [1.0; 3],
                };
                TextureDef::Constant(value)
            }
            "scale" => {
                let inner = match params.take("tex", &["texture", "float", "rgb", "color"])? {
                    Some(param) if param.ty == "texture" => {
                        self.texture_lookup(param.as_string()?, kind, &param.location)?
                    }
                    Some(param) => TextureDef::Constant(param.as_rgb_broadcast()?),
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
                        uv,
                    } => TextureDef::Image {
                        path,
                        color_space,
                        scale: scale * factor,
                        uv,
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
        if self
            .named_textures
            .insert((kind, name.clone()), def)
            .is_some()
        {
            // A same-kind redefinition: pbrt warns and keeps the last, and
            // since it creates every material only after the full parse,
            // every reference resolves to that last definition — our
            // flattened pre-pass agrees. (A cross-kind reuse is not a
            // redefinition; the two live in separate namespaces.)
            self.warn(format!(
                "{}: texture \"{name}\" redefined; the last definition wins",
                directive.location
            ));
        }
        Ok(())
    }
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
        files: &[(&str, impl AsRef<[u8]>)],
        inspect: impl FnOnce(&ChangeSet, &[String]) -> T,
    ) -> T {
        let dir = std::env::temp_dir().join(format!("cenote-map-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("fixture dir");
        for (name, source) in files {
            std::fs::write(dir.join(name), source.as_ref()).expect("write fixture");
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
                        cenote::scene::changeset::Kind::Medium => "medium",
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

            let transform = instances(set)[0].transforms.clone().expect("set")[0].clone();
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
            let transform = instances(set)[0].transforms.clone().expect("set")[0].clone();
            let object = transform.to_mat4().transform_point3(Vec3::ZERO);
            // pbrt renders camera-space +x = world +x on screen right;
            // the object at world +1 must land on cenote's right too.
            let forward = (look_at - position).normalize();
            let right = forward.cross(up).normalize();
            assert!(right.dot(object - position) > 0.5, "object at {object}");
        });
    }

    /// pbrt inverts `t` at every image-texture lookup; cenote samples `v`
    /// as stored. Authored UVs must land flipped, so both renderers fetch
    /// the same texel for the same hit.
    #[test]
    fn authored_uvs_flip_v_to_pbrt_lookup_convention() {
        let world = r#"Shape "trianglemesh"
            "point3 P" [0 0 0  1 0 0  0 1 0]
            "integer indices" [0 1 2]
            "point2 uv" [0 0  1 0  1 0.25]"#;
        import_world("uv-flip", world, |set, _| {
            let uvs = set
                .ops
                .iter()
                .find_map(|op| match op {
                    Op::Mesh(patch) => patch.source.as_ref(),
                    _ => None,
                })
                .and_then(|source| match source {
                    MeshSource::Inline { uvs, .. } => uvs.clone(),
                    MeshSource::Ply { .. } => unreachable!(),
                })
                .expect("authored uvs");
            assert_eq!(uvs, [[0.0, 1.0], [1.0, 1.0], [1.0, 0.75]]);
        });
    }

    /// `ReverseOrientation` alone flips authored normals and winding. A
    /// mirroring transform does NOT flip them back — pbrt's trap-4 XOR
    /// compensates for its own world-space vertex bake, but cenote keeps
    /// object-space winding and det-corrects the emission side under the
    /// instance transform, so the mirror must stay out of the decision.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "authored normal components must copy through bit-exact"
    )]
    fn reverse_orientation_flips_regardless_of_mirroring() {
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
            let (mirrored, mirrored_winding) = normal_z(mesh("trianglemesh-2"));
            assert_eq!(plain, 1.0);
            assert_eq!(plain_winding, [0, 1, 2]);
            // ReverseOrientation alone flips…
            assert_eq!(reversed, -1.0);
            assert_eq!(reversed_winding, [0, 2, 1]);
            // …and the mirroring transform leaves the (still-reversed)
            // orientation alone: the renderer det-corrects the mirror.
            assert_eq!(mirrored, -1.0);
            assert_eq!(mirrored_winding, [0, 2, 1]);
        });
    }

    /// A PLY's winding can't be reversed at import, so `ReverseOrientation`
    /// is dropped either way — but under two-sided shading that's invisible
    /// for a reflective mesh, and only the emitter (one-sided) loses a face.
    /// Warn there and stay silent on the reflective mesh.
    #[test]
    fn reverse_orientation_on_a_plymesh_warns_only_when_it_emits() {
        let world = "\
            AttributeBegin\n\
              ReverseOrientation\n\
              AreaLightSource \"diffuse\" \"rgb L\" [1 1 1]\n\
              Shape \"plymesh\" \"string filename\" \"emitter.ply\"\n\
            AttributeEnd\n\
            AttributeBegin\n\
              ReverseOrientation\n\
              Shape \"plymesh\" \"string filename\" \"plain.ply\"\n\
            AttributeEnd\n";
        import_world("plymesh-reverse", world, |_, warnings| {
            let hits = warnings
                .iter()
                .filter(|warning| warning.contains("emissive plymesh is ignored"))
                .count();
            assert_eq!(hits, 1, "warnings: {warnings:?}");
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
                    Op::Environment(patch) => patch.path.as_ref()?.as_ref(),
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
                    .transforms
                    .clone()
                    .expect("set")[0]
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
                ("scene.pbrt", world_scene(&world).as_str()),
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

    /// pbrt-v4 defers texture creation to after the world block is parsed,
    /// so a material may name a texture declared *later* in the file —
    /// watercolor does. The pre-pass in [`Mapper::run`] resolves every
    /// texture before any material looks one up, so the forward reference
    /// lands on the real image rather than failing "was never declared".
    #[test]
    fn a_material_may_name_a_texture_declared_later() {
        let world = format!(
            "MakeNamedMaterial \"planks\" \"string type\" \"diffuse\" \
             \"texture reflectance\" \"wood\"\n\
             NamedMaterial \"planks\"\n{TRIANGLE}\n\
             Texture \"wood\" \"spectrum\" \"imagemap\" \"string filename\" \"wood.png\"\n"
        );
        import_files(
            "forward-texture",
            &[
                ("scene.pbrt", world_scene(&world).as_str()),
                ("wood.png", "not-a-real-png"),
            ],
            |set, _| {
                let planks = material(set, "planks");
                match &planks.base_color {
                    Some(Texturable::Texture(reference)) => {
                        assert!(reference.path.ends_with("wood.png"), "{reference:?}");
                    }
                    other => panic!("expected the later-declared texture, got {other:?}"),
                }
            },
        );
    }

    /// pbrt treats a reference to an undefined named material as a warning
    /// and falls back to the default material, not a fatal error —
    /// watercolor names "BG", whose `MakeNamedMaterial` is commented out.
    /// The import must survive it, and the shape lands on `pbrt-default`.
    #[test]
    fn an_undefined_named_material_falls_back_to_the_default() {
        let world = format!("NamedMaterial \"ghost\"\n{TRIANGLE}\n");
        import_world("dangling-material", &world, |set, warnings| {
            let placed = instances(set);
            assert_eq!(placed.len(), 1);
            assert_eq!(placed[0].material.as_deref(), Some("pbrt-default"));
            assert!(
                warnings
                    .iter()
                    .any(|warning| warning.contains("ghost") && warning.contains("never made")),
                "{warnings:?}"
            );
        });
    }

    /// A `coatedconductor` maps to a metal base (`metalness = 1`, F0 from
    /// the named-metal table) under `OpenPBR`'s clear coat (`coat_weight =
    /// 1`) — the same coat `coateddiffuse` wears — rather than falling to
    /// the gray default. The conductor reads `conductor.*`, the coat
    /// `interface.*`.
    #[test]
    fn coated_conductor_wears_a_clear_coat_over_metal() {
        let world = format!(
            "MakeNamedMaterial \"gold\" \"string type\" \"coatedconductor\" \
             \"spectrum conductor.eta\" \"metal-Au-eta\" \
             \"spectrum conductor.k\" \"metal-Au-k\" \
             \"float conductor.roughness\" [ 0.1 ] \
             \"float interface.roughness\" [ 0.0 ]\n\
             NamedMaterial \"gold\"\n{TRIANGLE}\n"
        );
        import_world("coated-conductor", &world, |set, _| {
            let gold = material(set, "gold");
            assert_eq!(gold.base_metalness, Some(Texturable::Constant(1.0)));
            assert_eq!(gold.coat_weight, Some(1.0));
            match gold.base_color {
                Some(Texturable::Constant(f0)) => {
                    let expected = named_metal_f0("metal-Au-eta").expect("gold in the table");
                    assert!(
                        f0.iter().zip(expected).all(|(&a, b)| (a - b).abs() < 1e-6),
                        "expected the gold F0 {expected:?}, got {f0:?} (not the gray default)"
                    );
                }
                ref other => panic!("expected the gold F0, got {other:?}"),
            }
        });
    }

    /// `OpenPBR` has no diffuse-transmission lobe, so `diffusetransmission`
    /// imports as an opaque diffuse carrying its `reflectance`; the
    /// transmitted half is dropped with a warning.
    #[test]
    fn diffuse_transmission_imports_as_opaque_diffuse() {
        let world = format!(
            "MakeNamedMaterial \"leaf\" \"string type\" \"diffusetransmission\" \
             \"rgb reflectance\" [ 0.8 0.6 0.2 ] \"rgb transmittance\" [ 0.1 0.1 0.1 ]\n\
             NamedMaterial \"leaf\"\n{TRIANGLE}\n"
        );
        import_world("diffuse-transmission", &world, |set, warnings| {
            let leaf = material(set, "leaf");
            assert_eq!(leaf.base_color, Some(Texturable::Constant([0.8, 0.6, 0.2])));
            assert_eq!(leaf.specular_weight, Some(0.0));
            assert!(
                warnings
                    .iter()
                    .any(|warning| warning.contains("diffuse transmission")
                        && warning.contains("dropped")),
                "{warnings:?}"
            );
        });
    }

    /// pbrt describes a subsurface interior by its coefficients; cenote
    /// describes one by the color light walks to and how far it walks
    /// between events. The importer converts through the same fit prep
    /// inverts, so the test asserts the *round trip*: the color it authors
    /// must walk back to the σ_s/σ_t it was given, and the radius must be
    /// the scaled mean free path.
    #[test]
    fn subsurface_coefficients_invert_into_a_walked_color() {
        let world = format!(
            "MakeNamedMaterial \"jade\" \"string type\" \"subsurface\" \
             \"rgb sigma_s\" [ 2 2 2 ] \"rgb sigma_a\" [ 0 1 3 ] \
             \"float scale\" [ 2 ] \"float g\" [ 0.25 ]\n\
             NamedMaterial \"jade\"\n{TRIANGLE}\n"
        );
        import_world("subsurface-sigma", &world, |set, warnings| {
            let jade = material(set, "jade");
            assert_eq!(jade.subsurface_weight, Some(1.0));
            assert_eq!(jade.subsurface_scatter_anisotropy, Some(0.25));
            // eta is absent, so pbrt's subsurface default stands — 1.33,
            // not the 1.5 the glass materials default to.
            assert_eq!(jade.specular_ior, Some(1.33));
            // σ_t = scale · (σ_a + σ_s) = (4, 6, 10); the longest mean free
            // path takes the radius and the rest ride as its shape.
            let (radius, shape) = (
                jade.subsurface_radius.expect("a radius"),
                jade.subsurface_radius_scale.expect("a shape"),
            );
            assert!((radius - 0.25).abs() < 1e-6, "{radius}");
            for (channel, expected) in shape.iter().zip([1.0, 4.0 / 6.0, 0.4]) {
                assert!((channel - expected).abs() < 1e-6, "{shape:?}");
            }
            // And the colors walk back to the albedos they were made from.
            let color = jade.subsurface_color.expect("a color");
            for (channel, expected) in color.iter().zip([1.0, 2.0 / 3.0, 0.4]) {
                let walked = cenote::scene::single_scatter_albedo(*channel, 0.25);
                assert!(
                    (walked - expected).abs() < 1e-4,
                    "color {channel} walks to α = {walked}, not {expected}"
                );
            }
            assert!(warnings.is_empty(), "{warnings:?}");
        });
    }

    /// The other numeric form needs no conversion at all: `reflectance` is
    /// already the walked-to color and `mfp` already the distance, `scale`
    /// riding the latter as a length. Anything pbrt reads from a table it
    /// compiled in — a measured `name`, a texture cenote's constant slot
    /// cannot hold — is named in the warnings instead of guessed at.
    #[test]
    fn subsurface_reflectance_and_mfp_import_as_authored() {
        let world = format!(
            "MakeNamedMaterial \"skin\" \"string type\" \"subsurface\" \
             \"rgb reflectance\" [ 0.63 0.44 0.35 ] \
             \"rgb mfp\" [ 0.004 0.002 0.001 ] \"float scale\" [ 0.5 ]\n\
             NamedMaterial \"skin\"\n{TRIANGLE}\n"
        );
        import_world("subsurface-reflectance", &world, |set, warnings| {
            let skin = material(set, "skin");
            assert_eq!(skin.subsurface_color, Some([0.63, 0.44, 0.35]));
            assert_eq!(skin.subsurface_radius, Some(0.002));
            assert_eq!(skin.subsurface_radius_scale, Some([1.0, 0.5, 0.25]));
            assert!(warnings.is_empty(), "{warnings:?}");
        });

        // A measured medium: the table lives in pbrt, so the name is the
        // warning and the interior imports at the defaults.
        let world = format!(
            "MakeNamedMaterial \"marble\" \"string type\" \"subsurface\" \
             \"string name\" [ \"skin1\" ]\n\
             NamedMaterial \"marble\"\n{TRIANGLE}\n"
        );
        import_world("subsurface-name", &world, |set, warnings| {
            let marble = material(set, "marble");
            assert_eq!(marble.subsurface_weight, Some(1.0));
            assert_eq!(
                (marble.subsurface_color, marble.subsurface_radius),
                (None, None),
                "an interior cenote cannot read must not be invented"
            );
            assert!(
                warnings.iter().any(|warning| warning.contains("skin1")),
                "{warnings:?}"
            );
        });

        // A textured albedo: the mean free path still lands — it is the
        // half of the description that *is* constant — and the texture is
        // named as dropped.
        let world = format!(
            "Texture \"albedo\" \"spectrum\" \"imagemap\" \"string filename\" \"albedo.png\"\n\
             MakeNamedMaterial \"head\" \"string type\" \"subsurface\" \
             \"texture reflectance\" [ \"albedo\" ] \"rgb mfp\" [ 0.003 0.002 0.001 ]\n\
             NamedMaterial \"head\"\n{TRIANGLE}\n"
        );
        import_files(
            "subsurface-textured",
            &[
                ("scene.pbrt", format!("WorldBegin\n{world}\n").as_bytes()),
                ("albedo.png", &png_header(2)),
            ],
            |set, warnings| {
                let head = material(set, "head");
                assert_eq!(head.subsurface_color, None);
                assert_eq!(head.subsurface_radius, Some(0.003));
                assert!(
                    warnings
                        .iter()
                        .any(|warning| warning.contains("reflectance")
                            && warning.contains("textured")),
                    "{warnings:?}"
                );
            },
        );
    }

    /// pbrt-v3's `metal` is v4's `conductor` under an older name; it
    /// imports as a full metal (`metalness = 1`, F0 from its `eta`/`k`),
    /// not the gray default.
    #[test]
    fn v3_metal_imports_as_a_conductor() {
        let world = format!(
            "MakeNamedMaterial \"chrome\" \"string type\" \"metal\" \
             \"rgb eta\" [ 1.65 0.88 0.52 ] \"rgb k\" [ 9.2 6.26 4.83 ] \
             \"float roughness\" [ 0.005 ]\n\
             NamedMaterial \"chrome\"\n{TRIANGLE}\n"
        );
        import_world("v3-metal", &world, |set, _| {
            let chrome = material(set, "chrome");
            assert_eq!(chrome.base_metalness, Some(Texturable::Constant(1.0)));
            match chrome.base_color {
                Some(Texturable::Constant(f0)) => {
                    let expected = conductor_f0([1.65, 0.88, 0.52], [9.2, 6.26, 4.83]);
                    assert!(
                        f0.iter().zip(expected).all(|(&a, b)| (a - b).abs() < 1e-6),
                        "expected the eta/k conductor F0 {expected:?}, got {f0:?}"
                    );
                }
                ref other => panic!("expected a conductor F0, got {other:?}"),
            }
        });
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
            AreaLightSource \"diffuse\" \"rgb L\" [8 7 6] \"bool twosided\" true\n\
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
            // One warning expected: the two-sided area light summary.
            assert!(
                warnings.iter().any(|warning| warning.contains("two-sided")),
                "{warnings:?}"
            );
        });
    }

    /// The first 26 bytes of a PNG — signature and IHDR through the
    /// color type, everything the alpha probe reads. Not decodable.
    fn png_header(color_type: u8) -> [u8; 26] {
        let mut header = [0u8; 26];
        header[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        header[8..12].copy_from_slice(&13u32.to_be_bytes());
        header[12..16].copy_from_slice(b"IHDR");
        header[24] = 8;
        header[25] = color_type;
        header
    }

    /// A shape `alpha` texture forks the material into a cutout whose
    /// mask feeds `geometry_opacity` — through the alpha channel when
    /// the PNG carries one. Shapes pairing the same material and mask
    /// share the fork; a maskless shape keeps the unforked base.
    #[test]
    fn shape_alpha_forks_a_cutout_material() {
        let source = "WorldBegin\n\
            Texture \"mask\" \"float\" \"imagemap\" \"string filename\" \"mask.png\"\n\
            Material \"diffuse\" \"rgb reflectance\" [0.8 0.2 0.2]\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"mask\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"mask\"\n";
        let files: &[(&str, &[u8])] = &[
            ("scene.pbrt", source.as_bytes()),
            ("mask.png", &png_header(6)),
        ];
        import_files("alpha-cutout", files, |set, warnings| {
            let shapes = instances(set);
            let cutout = shapes[0].material.as_deref().expect("a material");
            assert_eq!(cutout, "diffuse-0-cutout-0");
            assert_eq!(shapes[1].material.as_deref(), Some("diffuse-0"));
            assert_eq!(shapes[2].material.as_deref(), Some(cutout));
            let patch = material(set, cutout);
            let Some(Texturable::Texture(reference)) = &patch.geometry_opacity else {
                panic!("no opacity mask: {patch:?}");
            };
            assert!(reference.path.ends_with("mask.png"));
            assert_eq!(reference.channel, Some(Channel::A));
            // The base color rode along into the fork.
            assert_eq!(
                patch.base_color,
                Some(Texturable::Constant([0.8, 0.2, 0.2]))
            );
            assert_eq!(material(set, "diffuse-0").geometry_opacity, None);
            assert!(warnings.is_empty(), "{warnings:?}");
        });
    }

    /// The mask channel follows the image: an alpha-less PNG or a
    /// 24-bit TGA reads the red default (pbrt averages RGB — identical
    /// for the gray masks that case is in practice), and a float
    /// `alpha` imports as a constant — unless it is the fully-opaque
    /// default, which forks nothing.
    #[test]
    fn mask_channels_and_float_alphas_follow_the_source() {
        // A 24-bit TGA header: pixel depth (byte 16) is all the sniff
        // reads — TGA has no magic bytes.
        let mut tga = [0u8; 26];
        tga[2] = 2;
        tga[16] = 24;
        let source = "WorldBegin\n\
            Texture \"mask\" \"float\" \"imagemap\" \"string filename\" \"mask.png\"\n\
            Texture \"rug\" \"float\" \"imagemap\" \"string filename\" \"rug.tga\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"mask\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"float alpha\" 0.25\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"float alpha\" 1\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"rug\"\n";
        let files: &[(&str, &[u8])] = &[
            ("scene.pbrt", source.as_bytes()),
            ("mask.png", &png_header(0)),
            ("rug.tga", &tga),
        ];
        import_files("alpha-channels", files, |set, warnings| {
            let shapes = instances(set);
            let gray = material(set, shapes[0].material.as_deref().expect("a material"));
            let Some(Texturable::Texture(reference)) = &gray.geometry_opacity else {
                panic!("no opacity mask: {gray:?}");
            };
            assert_eq!(reference.channel, None);
            let faded = material(set, shapes[1].material.as_deref().expect("a material"));
            assert_eq!(faded.geometry_opacity, Some(Texturable::Constant(0.25)));
            assert_eq!(shapes[2].material.as_deref(), Some("pbrt-default"));
            let rug = material(set, shapes[3].material.as_deref().expect("a material"));
            let Some(Texturable::Texture(reference)) = &rug.geometry_opacity else {
                panic!("no opacity mask: {rug:?}");
            };
            assert_eq!(reference.channel, None);
            assert!(warnings.is_empty(), "{warnings:?}");
        });
    }

    /// An alpha'd emitter wears both forks: the cutout comes first, so
    /// the area light's glow forks the cutout material and the shape
    /// ends up with mask and emission together.
    #[test]
    fn an_alpha_cutout_composes_with_an_area_light() {
        let source = "WorldBegin\n\
            Texture \"mask\" \"float\" \"imagemap\" \"string filename\" \"mask.png\"\n\
            AreaLightSource \"diffuse\" \"rgb L\" [4 2 1]\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"mask\"\n";
        let files: &[(&str, &[u8])] = &[
            ("scene.pbrt", source.as_bytes()),
            ("mask.png", &png_header(6)),
        ];
        import_files("alpha-glow", files, |set, _| {
            let shapes = instances(set);
            let name = shapes[0].material.as_deref().expect("a material");
            assert_eq!(name, "pbrt-default-cutout-0-glow-0");
            let patch = material(set, name);
            assert_eq!(
                patch.emission_color,
                Some(Texturable::Constant([4.0, 2.0, 1.0]))
            );
            assert!(
                matches!(patch.geometry_opacity, Some(Texturable::Texture(_))),
                "{patch:?}"
            );
        });
    }

    /// pbrt's affine UV parameters land on the reference with the v leg
    /// flipped into storage order — `offset_v = 1 − vscale − vdelta`, the
    /// transform-space image of the `1 − v` the UVs themselves get — and
    /// an imagemap `scale` rides the reference's multiplier. Both were
    /// warned-and-dropped before this landed.
    #[test]
    fn uv_affines_flip_into_storage_order_and_the_scale_rides() {
        let world = "Texture \"art\" \"spectrum\" \"imagemap\" \
            \"string filename\" \"art.png\" \"float scale\" 3 \
            \"float uscale\" 2 \"float vscale\" 0.5 \
            \"float udelta\" 0.25 \"float vdelta\" 0.1\n\
            MakeNamedMaterial \"Art\" \"string type\" \"diffuse\" \
            \"texture reflectance\" \"art\"\n\
            NamedMaterial \"Art\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]";
        import_world("uv-affine", world, |set, warnings| {
            let patch = material(set, "Art");
            let Some(Texturable::Texture(reference)) = &patch.base_color else {
                panic!("expected a textured base color: {patch:?}");
            };
            assert_eq!(reference.scale, Some(3.0));
            let uv = reference.uv.expect("an affine remap");
            assert_eq!(uv.scale, [2.0, 0.5]);
            assert_eq!(uv.offset[0], 0.25);
            assert!((uv.offset[1] - (1.0 - 0.5 - 0.1)).abs() < 1e-6, "{uv:?}");
            assert!(
                warnings.iter().all(|warning| !warning.contains("UV")),
                "{warnings:?}"
            );
        });
    }

    /// Identity parameters stay off the reference — an untransformed
    /// imagemap serializes exactly as before the feature.
    #[test]
    fn identity_uv_parameters_leave_the_reference_bare() {
        let world = "Texture \"art\" \"spectrum\" \"imagemap\" \
            \"string filename\" \"art.png\" \"float uscale\" 1 \"float vdelta\" 0\n\
            MakeNamedMaterial \"Art\" \"string type\" \"diffuse\" \
            \"texture reflectance\" \"art\"\n\
            NamedMaterial \"Art\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]";
        import_world("uv-identity", world, |set, _| {
            let patch = material(set, "Art");
            let Some(Texturable::Texture(reference)) = &patch.base_color else {
                panic!("expected a textured base color: {patch:?}");
            };
            assert_eq!(reference.scale, None);
            assert_eq!(reference.uv, None);
        });
    }

    /// A `scale` texture over an imagemap composes its factor onto the
    /// reference's multiplier instead of dropping it.
    #[test]
    fn a_scale_texture_composes_onto_the_reference() {
        let world = "Texture \"wood\" \"spectrum\" \"imagemap\" \
            \"string filename\" \"wood.png\" \"float scale\" 2\n\
            Texture \"dim\" \"spectrum\" \"scale\" \
            \"texture tex\" \"wood\" \"float scale\" 0.25\n\
            MakeNamedMaterial \"Dim\" \"string type\" \"diffuse\" \
            \"texture reflectance\" \"dim\"\n\
            NamedMaterial \"Dim\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]";
        import_world("scale-compose", world, |set, _| {
            let patch = material(set, "Dim");
            let Some(Texturable::Texture(reference)) = &patch.base_color else {
                panic!("expected a textured base color: {patch:?}");
            };
            assert_eq!(reference.scale, Some(0.5));
        });
    }

    /// pbrt keeps `float` and `spectrum` textures in disjoint namespaces,
    /// so one name may denote a `float` mask in one include file and a
    /// `spectrum` reflectance in another — sanmiguel's `Map #483` is
    /// exactly this collision. Each slot resolves against its own kind: the
    /// reflectance (spectrum) takes the color image, the alpha (float) the
    /// mask, though both name `"Map"`.
    #[test]
    fn float_and_spectrum_textures_share_a_name_without_colliding() {
        let source = "WorldBegin\n\
            Texture \"Map\" \"spectrum\" \"imagemap\" \"string filename\" \"color.png\"\n\
            Texture \"Map\" \"float\" \"imagemap\" \"string filename\" \"mask.png\"\n\
            Material \"diffuse\" \"texture reflectance\" \"Map\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2] \"texture alpha\" \"Map\"\n";
        let files: &[(&str, &[u8])] = &[
            ("scene.pbrt", source.as_bytes()),
            ("color.png", &png_header(0)),
            ("mask.png", &png_header(6)),
        ];
        import_files("tex-namespaces", files, |set, warnings| {
            let shapes = instances(set);
            let patch = material(set, shapes[0].material.as_deref().expect("a material"));
            let Some(Texturable::Texture(color)) = &patch.base_color else {
                panic!("expected a textured base color: {patch:?}");
            };
            assert!(color.path.ends_with("color.png"), "{:?}", color.path);
            let Some(Texturable::Texture(mask)) = &patch.geometry_opacity else {
                panic!("expected an opacity mask: {patch:?}");
            };
            assert!(mask.path.ends_with("mask.png"), "{:?}", mask.path);
            assert!(
                !warnings.iter().any(|warning| warning.contains("redefined")),
                "a cross-kind reuse is not a redefinition: {warnings:?}"
            );
        });
    }

    /// A genuine same-kind redefinition is pbrt's last-wins with a warning:
    /// pbrt creates every material only after the full parse, so all
    /// references see the final definition — the flattened pre-pass agrees.
    #[test]
    fn a_same_kind_texture_redefinition_takes_the_last_and_warns() {
        let world = "Texture \"wall\" \"spectrum\" \"imagemap\" \
            \"string filename\" \"first.png\"\n\
            Texture \"wall\" \"spectrum\" \"imagemap\" \
            \"string filename\" \"second.png\"\n\
            MakeNamedMaterial \"Wall\" \"string type\" \"diffuse\" \
            \"texture reflectance\" \"wall\"\n\
            NamedMaterial \"Wall\"\n\
            Shape \"trianglemesh\" \"point3 P\" [0 0 0  1 0 0  0 1 0] \
            \"integer indices\" [0 1 2]";
        import_world("tex-redefine", world, |set, warnings| {
            let patch = material(set, "Wall");
            let Some(Texturable::Texture(reference)) = &patch.base_color else {
                panic!("expected a textured base color: {patch:?}");
            };
            assert!(
                reference.path.ends_with("second.png"),
                "the last definition wins: {:?}",
                reference.path
            );
            assert!(
                warnings.iter().any(|warning| warning.contains("redefined")),
                "{warnings:?}"
            );
        });
    }

    /// A single-patch `bilinearmesh` imports as two triangles on the
    /// patch's `dpdu × dpdv` side, its authored UVs v-flipped like every
    /// authored stream; a patch mesh with explicit indices stays skipped.
    #[test]
    fn a_single_patch_bilinearmesh_imports_as_two_triangles() {
        let world = "Shape \"bilinearmesh\" \
            \"point3 P\" [0 0 0  1 0 0  0 1 0  1 1 0] \
            \"point2 uv\" [0 1  1 1  0 0  1 0]\n\
            Shape \"bilinearmesh\" \
            \"point3 P\" [0 0 0  1 0 0  0 1 0  1 1 0] \
            \"integer indices\" [0 1 2 3]";
        import_world("bilinear", world, |set, warnings| {
            let sources: Vec<_> = set
                .ops
                .iter()
                .filter_map(|op| match op {
                    Op::Mesh(patch) => patch.source.as_ref(),
                    _ => None,
                })
                .collect();
            assert_eq!(sources.len(), 1, "the indexed mesh must skip");
            let MeshSource::Inline {
                positions,
                uvs,
                triangles,
                ..
            } = sources[0]
            else {
                panic!("expected an inline mesh");
            };
            assert_eq!(positions.len(), 4);
            // (p00, p10, p11) and (p00, p11, p01): +z winding for a
            // +x/+y patch, pbrt's dpdu × dpdv.
            assert_eq!(triangles, &vec![[0, 1, 3], [0, 3, 2]]);
            // Authored (0,1) at p00 flips to storage (0,0).
            assert_eq!(
                uvs.as_ref().expect("authored uvs")[..2],
                [[0.0, 0.0], [1.0, 0.0]]
            );
            assert!(
                warnings
                    .iter()
                    .any(|warning| warning.contains("single-patch")),
                "{warnings:?}"
            );
        });
    }
}
