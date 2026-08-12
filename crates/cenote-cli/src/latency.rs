//! `edit-latency`: how long the renderer takes to *show* an edit.
//!
//! One scene, one session, one pass down a fixed vocabulary of edits — a
//! material tweak, an instance moved, a texture, a new mesh — each issued
//! against a settled image and timed to the first pixel that carries it.
//! Nothing here holds a stopwatch: the renderer measures its own latency
//! ([`cenote::stats::Phases`]) and hands the breakdown back on the published
//! frame, so this harness starts the clock by issuing a verb and reads the
//! result off the frame that answers it. A number in the sidecar and the
//! same number in the viewer's overlay therefore cannot disagree.
//!
//! `--drag` swaps the walk for the other half of interactivity: not one verb
//! against a settled image but a sustained one, the view moving as fast as
//! the renderer can answer it. See [`drag`].
//!
//! The edits are synthesized from whatever the scene already holds, with
//! targets picked in name order — the determinism rule the rest of the
//! renderer runs on, applied to choosing what to edit, so the same scene
//! walks the same edits every run. A scene with nothing to target (no
//! environment image, no textures anywhere) skips that edit and records the
//! reason; the sidecar carries the skip rather than a zero.
//!
//! This is benchmark scaffolding and it lives here rather than in the
//! library. The core's own edit walk — `scene::prep`'s tests — is
//! hand-authored against the demo and checks *correctness*; it shares this
//! vocabulary and none of its code.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use cenote::render::{Frame, Session};
use cenote::scene::Camera;
use cenote::scene::changeset::{
    ChangeSet, EnvironmentPatch, InstancePatch, Kind, LightPatch, MaterialPatch, MeshPatch, Op,
    SettingsPatch,
};
use cenote::scene::description::{
    Channel, Light, SceneDescription, Texturable, TextureRef, Transform,
};
use cenote::stats::{Phases, Stats};
use serde::Serialize;

/// How long the harness waits for a frame before calling it a hang. Not a
/// budget — a big scene's first settle is legitimately seconds — just the
/// line past which something has gone wrong and printing so beats waiting.
const WAIT_LIMIT: Duration = Duration::from_mins(5);

/// How often the harness looks for a published frame. Below the render
/// thread's own idle nap, so waiting adds nothing to the number.
const POLL: Duration = Duration::from_millis(1);

/// The objects [`EditKind::Topology`] adds and [`EditKind::Removal`] takes
/// away. Named after the harness so a scene file that somehow got one
/// written into it says where it came from.
const PROBE: &str = "cenote-latency-probe";

/// Degrees [`EditKind::EnvironmentPlacement`] turns the sky, big enough to
/// see in the frame it lands in.
const TURN_DEGREES: f32 = 37.0;

/// Width of the console's first column — wide enough for the longest edit
/// name, so every row's numbers line up under the header's.
const LABEL_WIDTH: usize = 22;

#[derive(clap::Args)]
pub struct LatencyArgs {
    /// Scene file, `.ron` or `.pbrt`.
    scene: PathBuf,

    /// Render width in pixels. Defaults to the scene's settings — latency
    /// is resolution-bound, so this is not a detail.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    width: Option<u32>,

    /// Render height in pixels. Defaults to the scene's settings.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    height: Option<u32>,

    /// Samples to accumulate before each edit. The render parks at this
    /// count, so every edit lands on a settled image — which is the case a
    /// person editing lookdev actually hits, and it makes the walk
    /// reproducible.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
    settle: u32,

    /// Drag the view instead of walking the edits: this many camera moves
    /// back to back, each issued as soon as a frame carrying the last one
    /// arrives, so the render never settles. Reports the cadence during the
    /// motion — the number a person feels while orbiting.
    #[arg(long, value_name = "MOVES", value_parser = clap::value_parser!(u32).range(2..))]
    drag: Option<u32>,

    /// Let the render drop resolution while the picture is changing, at the
    /// session's interactive target. Applied to both measurements: it
    /// shortens the frame time through the drag, and the wait for the first
    /// frame after each edit of the walk. Watch the `size` column — a row
    /// that answers sooner because it holds fewer pixels says so there.
    #[arg(long)]
    preview: bool,

    /// Denoise the published frames, as the viewer does by default. Applied
    /// to both measurements, and it is not free: the filter runs over every
    /// frame the drag publishes, so `--preview` divides a bigger number and
    /// may pick a smaller rectangle than it does without this.
    #[arg(long)]
    denoise: bool,

    /// Walk only these edits, comma separated (`--only material,transform`).
    /// Note that later edits in the walk target what earlier ones left
    /// behind: `removal` takes away `topology`'s mesh.
    #[arg(long, value_delimiter = ',')]
    only: Option<Vec<EditKind>>,

    /// Delete the DDS texture caches beside the scene's images first, so
    /// the run measures a cold decode-and-compress instead of a cache read.
    /// They regenerate on use; the first run after this one is slow on
    /// purpose.
    #[arg(long)]
    cold_textures: bool,

    /// Sidecar path.
    #[arg(long, default_value = "latency.ron")]
    out: PathBuf,
}

/// The edit vocabulary: one variant per re-prep path the scene model has,
/// named after the path it exercises rather than after the field it sets.
///
/// Declaration order is walk order, and the walk is a sequence rather than a
/// set — the texture edit's bindless slot is what the texture removal
/// retires, the topology edit's mesh is what the removal removes, and the
/// environment edit drops the sky image the two edits before it were
/// turning and tinting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum EditKind {
    /// A scalar material parameter: one material record and nothing else.
    Material,
    /// A material starts emitting — the emitter tables rebuild.
    Emission,
    /// Coat plus fractional opacity: the material records *and* the TLAS
    /// opacity flags.
    Closure,
    /// The view moves. The one verb here that is not a change-set: it is
    /// `Session::set_camera`, the path the viewer's orbit control takes.
    Camera,
    /// A delta light brightens.
    Light,
    /// An instance leaves camera rays — TLAS instance masks.
    Visibility,
    /// An instance moves — the TLAS rebuilds over it.
    Transform,
    /// The sky's tint: scene-table constants, image kept resident.
    EnvironmentTint,
    /// The sky turns: same path, no re-decode.
    EnvironmentPlacement,
    /// The sky's image is dropped, retiring its residency and its sampling
    /// tables.
    Environment,
    /// A material grows a texture the scene has not decoded before: a real
    /// decode, a real block compress, a new bindless slot.
    Texture,
    /// ...and loses it again, retiring the slot.
    TextureRemoval,
    /// New geometry — a second copy of the scene's first mesh, which on a
    /// PLY scene is a real file read and a real BLAS build.
    Topology,
    /// ...removed, retiring its residency.
    Removal,
}

/// What the harness does to the session to provoke a restart. Two verbs,
/// because the interactive camera path is not a change-set and pretending
/// it is would measure the wrong thing.
enum Verb {
    /// Queue a change-set ([`Session::apply`]).
    Edit(ChangeSet),
    /// Move the view ([`Session::set_camera`]).
    Move(Camera),
}

impl EditKind {
    /// The verb this edit is against `description` — or why this scene has
    /// nothing for it to target. `view` is the camera the session is
    /// currently rendering, which is the one the camera edit moves.
    #[expect(
        clippy::too_many_lines,
        reason = "a flat list of edits, one per re-prep path — the same shape as the \
                  core's own edit walk, and splitting it would hide the vocabulary"
    )]
    fn synthesize(self, description: &SceneDescription, view: Camera) -> Result<Verb, String> {
        match self {
            Self::Material => {
                let (name, material) = first(description.materials(), "material")?;
                // Diffuse roughness because it is the one material field
                // that is a plain scalar on every scene: no texture can be
                // hiding in it, so this edit writes a material record and
                // touches no residency.
                Ok(material_edit(MaterialPatch {
                    base_diffuse_roughness: Some(wrapped(material.base_diffuse_roughness)),
                    ..MaterialPatch::new(name)
                }))
            }
            Self::Emission => {
                let (name, material) = first(description.materials(), "material")?;
                Ok(material_edit(MaterialPatch {
                    // Upward, so a material that was dark becomes a light
                    // and the emitter tables have something new to hold.
                    emission_luminance: Some(material.emission_luminance + 10.0),
                    ..MaterialPatch::new(name)
                }))
            }
            Self::Closure => {
                let (name, material) = first(description.materials(), "material")?;
                Ok(material_edit(MaterialPatch {
                    coat_weight: Some(wrapped(material.coat_weight)),
                    // Fractional opacity flips the instance's TLAS opacity
                    // flag — the material edit that also rebuilds the TLAS.
                    geometry_opacity: Some(Texturable::Constant(0.5)),
                    ..MaterialPatch::new(name)
                }))
            }
            Self::Camera => {
                let mut moved = view;
                moved.position.x = stepped(moved.position.x);
                Ok(Verb::Move(moved))
            }
            Self::Light => {
                let (name, light) = first(description.lights(), "light")?;
                let brighter = match *light {
                    Light::Distant {
                        direction,
                        irradiance,
                    } => Light::Distant {
                        direction,
                        irradiance: irradiance.map(stepped),
                    },
                    Light::Point {
                        position,
                        intensity,
                    } => Light::Point {
                        position,
                        intensity: intensity.map(stepped),
                    },
                };
                Ok(edit(Op::Light(LightPatch {
                    light: Some(brighter),
                    ..LightPatch::new(name)
                })))
            }
            Self::Visibility => {
                let (name, instance) = first(description.instances(), "instance")?;
                Ok(edit(Op::Instance(InstancePatch {
                    camera_visible: Some(!instance.camera_visible),
                    ..InstancePatch::new(name)
                })))
            }
            Self::Transform => {
                let (name, instance) = first(description.instances(), "instance")?;
                let mut placements = instance.transforms.clone();
                let placement = placements
                    .first_mut()
                    .ok_or_else(|| format!("instance \"{name}\" places nothing"))?;
                *placement = moved(placement);
                Ok(edit(Op::Instance(InstancePatch {
                    transforms: Some(placements),
                    ..InstancePatch::new(name)
                })))
            }
            Self::EnvironmentTint => {
                let (name, environment) = first(description.environments(), "environment")?;
                Ok(edit(Op::Environment(EnvironmentPatch {
                    tint: Some(environment.tint.map(stepped)),
                    ..EnvironmentPatch::new(name)
                })))
            }
            Self::EnvironmentPlacement => {
                let (name, environment) = first(description.environments(), "environment")?;
                Ok(edit(Op::Environment(EnvironmentPatch {
                    transform: Some(turned(&environment.transform)?),
                    ..EnvironmentPatch::new(name)
                })))
            }
            Self::Environment => {
                let (name, environment) = first(description.environments(), "environment")?;
                if environment.path.is_none() {
                    return Err("the environment carries no image to retire".to_owned());
                }
                Ok(edit(Op::Environment(EnvironmentPatch {
                    path: Some(None),
                    ..EnvironmentPatch::new(name)
                })))
            }
            Self::Texture => {
                let image = images(description)
                    .into_iter()
                    .next()
                    .ok_or_else(|| "no material in the scene references an image".to_owned())?;
                let (name, _) = first(description.materials(), "material")?;
                Ok(material_edit(MaterialPatch {
                    // The green channel of an image the scene already
                    // carries. A scalar bake of one channel caches and
                    // resides separately from the same file's color bake,
                    // so this is a genuine decode, block compress, and new
                    // bindless slot — on any textured scene, without the
                    // harness shipping an image of its own.
                    specular_roughness: Some(Texturable::Texture(TextureRef {
                        path: image,
                        color_space: None,
                        channel: Some(Channel::G),
                        scale: None,
                        uv: None,
                    })),
                    ..MaterialPatch::new(name)
                }))
            }
            Self::TextureRemoval => {
                let (name, _) = first(description.materials(), "material")?;
                Ok(material_edit(MaterialPatch {
                    specular_roughness: Some(Texturable::Constant(0.3)),
                    ..MaterialPatch::new(name)
                }))
            }
            Self::Topology => {
                let (_, mesh) = first(description.meshes(), "mesh")?;
                let (material, _) = first(description.materials(), "material")?;
                Ok(Verb::Edit(ChangeSet {
                    ops: vec![
                        // The payload is copied, not referenced: a new mesh
                        // name is new residency, so a PLY source is read
                        // again and built into its own BLAS.
                        Op::Mesh(MeshPatch {
                            source: Some(mesh.source.clone()),
                            ..MeshPatch::new(PROBE)
                        }),
                        Op::Instance(InstancePatch {
                            mesh: Some(PROBE.to_owned()),
                            material: Some(material.to_owned()),
                            transforms: Some(vec![Transform::default()]),
                            ..InstancePatch::new(PROBE)
                        }),
                    ],
                }))
            }
            Self::Removal => {
                if !description.instances().contains_key(PROBE) {
                    return Err(format!(
                        "\"{PROBE}\" is not in the scene — the topology edit adds it, \
                         and this edit follows it in the walk"
                    ));
                }
                Ok(Verb::Edit(ChangeSet {
                    ops: vec![
                        Op::Remove(Kind::Instance, PROBE.to_owned()),
                        Op::Remove(Kind::Mesh, PROBE.to_owned()),
                    ],
                }))
            }
        }
    }

    /// The one name this edit has: the console's, the sidecar's, and the
    /// one `--only` takes. Taken from the value enum rather than spelled
    /// again, so reading a row tells you what to type to re-run it.
    fn label(self) -> String {
        <Self as clap::ValueEnum>::to_possible_value(&self)
            .expect("every kind is selectable")
            .get_name()
            .to_owned()
    }
}

/// The first object of a kind, by name — the determinism rule (name order
/// everywhere) applied to picking an edit's target, so the same scene walks
/// the same edits on every machine and every run.
fn first<'a, T>(objects: &'a BTreeMap<String, T>, kind: &str) -> Result<(&'a str, &'a T), String> {
    objects
        .iter()
        .next()
        .map(|(name, object)| (name.as_str(), object))
        .ok_or_else(|| format!("the scene has no {kind}"))
}

/// One op as a verb.
fn edit(op: Op) -> Verb {
    Verb::Edit(ChangeSet { ops: vec![op] })
}

/// One material patch as a verb — the shape most of the vocabulary takes.
fn material_edit(patch: MaterialPatch) -> Verb {
    edit(Op::Material(Box::new(patch)))
}

/// A value in the unit range this one certainly is not: a fixed step,
/// wrapped. Wrapping at this step has no fixed point, and that matters —
/// an edit whose values are already in place dirties nothing (the
/// change-set contract), and a row of the walk would come back a no-op
/// instead of a measurement.
fn wrapped(value: f32) -> f32 {
    (value + 0.37).fract().abs()
}

/// A value this one certainly is not, for quantities with no range:
/// relative, so it survives f32 precision at any scene scale, and small, so
/// a nudged camera still looks at the same thing.
fn stepped(value: f32) -> f32 {
    value + value.abs().max(1.0) * 0.001
}

/// One placement, moved a step along X — in whichever spelling the scene
/// used. Small, because the TLAS rebuild is the point and not the picture.
fn moved(placement: &Transform) -> Transform {
    match placement {
        Transform::Trs {
            translate,
            rotate_degrees,
            scale,
        } => Transform::Trs {
            translate: [stepped(translate[0]), translate[1], translate[2]],
            rotate_degrees: *rotate_degrees,
            scale: *scale,
        },
        Transform::Matrix(rows) => {
            let mut rows = *rows;
            // Translation lives in the last column.
            rows[0][3] = stepped(rows[0][3]);
            Transform::Matrix(rows)
        }
    }
}

/// The sky, turned about Y from wherever it was. A matrix sky is left
/// alone rather than re-spelled as a `Trs`, which would throw away the
/// placement it already had.
fn turned(placement: &Transform) -> Result<Transform, String> {
    match placement {
        Transform::Trs {
            translate,
            rotate_degrees,
            scale,
        } => Ok(Transform::Trs {
            translate: *translate,
            rotate_degrees: [
                rotate_degrees[0],
                rotate_degrees[1] + TURN_DEGREES,
                rotate_degrees[2],
            ],
            scale: *scale,
        }),
        Transform::Matrix(_) => {
            Err("the sky is placed by matrix; the harness only turns a `Trs` sky".to_owned())
        }
    }
}

/// Every image the scene references, in path order and without repeats.
///
/// Three questions read this one list: what the texture-cache state is,
/// what `--cold-textures` clears, and — its first entry — which image
/// [`EditKind::Texture`] re-reads. That last one wants any image this
/// machine can certainly decode, and one the scene decodes already is the
/// best evidence available.
fn images(description: &SceneDescription) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = description
        .materials()
        .values()
        .flat_map(|material| {
            [
                material.base_color.texture(),
                material.base_metalness.texture(),
                material.specular_roughness.texture(),
                material.emission_color.texture(),
                material.geometry_opacity.texture(),
                material.geometry_normal.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|reference| reference.path.clone())
            .collect::<Vec<_>>()
        })
        .collect();
    paths.sort_unstable();
    paths.dedup();
    paths
}

/// Whether the block-compressed texture caches were sitting beside the
/// scene's images when the run started.
///
/// A cold run pays seconds of decode and compress in an edit's `lower`
/// phase that a warm run does not, so a latency sidecar that does not say
/// which one it was is not comparable to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum TextureCache {
    /// At least one cache file was there.
    Warm,
    /// None were — every texture this run touches is decoded and
    /// compressed from source.
    Cold,
    /// The scene references no images at all, so neither word applies and
    /// claiming one of them would read as a measurement.
    Untextured,
}

impl TextureCache {
    /// Which state this run will measure under, clearing the caches first
    /// if `--cold-textures` asked for it. Reading and setting the state are
    /// one step because they are one decision: a run that clears is cold,
    /// and a run that does not is whatever it found.
    fn of(images: &[PathBuf], clear: bool) -> anyhow::Result<Self> {
        if images.is_empty() {
            return Ok(Self::Untextured);
        }
        if clear {
            clear_texture_caches(images)?;
            return Ok(Self::Cold);
        }
        Ok(if images.iter().any(|image| !caches_of(image).is_empty()) {
            Self::Warm
        } else {
            Self::Cold
        })
    }

    /// The word the console header uses.
    fn label(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Untextured => "absent",
        }
    }
}

/// A cache file sits beside its source, named after it: `wood.png` becomes
/// `wood.png.color.dds`, its green channel `wood.png.scalar.g.dds`. That
/// prefix rule is the whole contract this harness relies on — it does not
/// try to reconstruct the rest of the name, only to recognize the family.
fn is_cache_of(entry: &Path, image: &Path) -> bool {
    let (Some(entry), Some(image)) = (entry.file_name(), image.file_name()) else {
        return false;
    };
    let (entry, image) = (entry.to_string_lossy(), image.to_string_lossy());
    entry.starts_with(&format!("{image}.")) && entry.ends_with(".dds")
}

/// The cache files beside `image`.
fn caches_of(image: &Path) -> Vec<PathBuf> {
    let Some(directory) = image.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|entry| is_cache_of(entry, image))
        .collect()
}

/// Delete the texture caches beside the scene's images, so the run measures
/// a decode rather than a file read. Reports what it removed: this deletes
/// files in the scene's own directory, and doing that quietly would be
/// worse than doing it at all.
fn clear_texture_caches(images: &[PathBuf]) -> anyhow::Result<()> {
    let mut removed = 0;
    for image in images {
        for cache in caches_of(image) {
            std::fs::remove_file(&cache)
                .with_context(|| format!("removing the texture cache {}", cache.display()))?;
            println!("  removed {}", cache.display());
            removed += 1;
        }
    }
    println!("  {removed} texture cache(s) cleared");
    Ok(())
}

/// One edit's outcome. A skip is a result too — a scene with no textures
/// has no texture latency, and saying so beats a zero that reads like a
/// fast edit.
#[derive(Debug, Serialize)]
enum Outcome {
    /// It landed, and this is where the time went.
    Measured {
        /// Verb to first pixel, milliseconds — the same interval
        /// `phases` accounts for, so the two cannot disagree.
        millis: f64,
        /// The breakdown the renderer handed back.
        phases: Phases,
        /// What the answering frame rendered at. A frame that arrives sooner
        /// because it holds fewer pixels is not the same result as one that
        /// arrives sooner, and the total alone cannot tell them apart — so
        /// the two travel together, as the drag's `sizes` does.
        size: (u32, u32),
        /// The same verb to a readable full-resolution image, milliseconds,
        /// or `None` when the mark did not land — a settle count short of
        /// [`cenote::stats::READABLE_SAMPLES`] parks the render before it.
        /// The counter-metric to `millis`: anything that buys a first pixel
        /// by rendering less of the image pays for it here.
        readable_millis: Option<f64>,
    },
    /// The scene had nothing for this edit to target.
    Skipped(String),
    /// The renderer took the edit and restarted nothing: every value it set
    /// was already in place, so no re-prep ran and there was no latency to
    /// measure.
    NoOp,
    /// The renderer rejected it.
    Rejected(String),
}

/// The last value the walk saw of each interaction mark, carried between
/// edits. A mark that did not move is how the harness tells a frame
/// answering *this* verb from one answering the last — see [`walk_one`].
struct Marks {
    first_pixel: Option<Duration>,
    readable: Option<Duration>,
}

/// One row of the walk.
#[derive(Debug, Serialize)]
struct Measurement {
    kind: EditKind,
    outcome: Outcome,
}

/// What the renderer did while the view kept moving — see [`drag`].
#[derive(Debug, Serialize)]
struct DragReport {
    moves: u32,
    /// Whether the render was allowed to drop resolution during the motion —
    /// a condition of the measurement, like the texture-cache state.
    preview: bool,
    /// Median cost of the filter over the drag's frames, milliseconds, or
    /// `None` if nothing was denoised. Part of `frame_millis` rather than
    /// beside it: with denoising on the filter runs inside every frame the
    /// drag publishes, and this says how much of the cadence it is.
    denoise_millis: Option<f64>,
    /// Median interval between published frames, milliseconds. The cadence,
    /// and the one number a resolution change has to move.
    frame_millis: f64,
    /// Median verb-to-first-pixel across the moves, milliseconds.
    first_pixel_millis: f64,
    /// Every move's verb-to-first-pixel, in order. The sequence and not just
    /// its middle, because a drag whose marks are all *identical* is one
    /// where the renderer stopped re-arming them between waves.
    first_pixels: Vec<f64>,
    /// Every size a frame came back at. One entry unless the renderer
    /// changed resolution during the drag.
    sizes: BTreeSet<(u32, u32)>,
}

/// The sidecar: what was measured, on what, under which conditions.
#[derive(Debug, Serialize)]
struct LatencyReport {
    device: String,
    scene: String,
    size: (u32, u32),
    /// Samples accumulated before each edit — every measurement here is an
    /// edit to a settled image.
    settle: u32,
    /// Whether the session denoised what it published — a condition of every
    /// number below, since the filter runs inside the frames they time.
    denoise: bool,
    texture_cache: TextureCache,
    /// Process start to the first ray, milliseconds, and where it went.
    load_millis: f64,
    load: Phases,
    /// The edit walk, or empty under `--drag`: a run measures one or the
    /// other, never both, because the drag leaves the view somewhere the
    /// walk did not put it.
    edits: Vec<Measurement>,
    drag: Option<DragReport>,
}

/// Walk `args.scene` through the edit vocabulary, printing each result and
/// leaving a RON sidecar.
pub fn run(args: &LatencyArgs) -> anyhow::Result<()> {
    let gpu = Arc::new(cenote::gpu::Context::new()?);
    let set = crate::load_scene(&args.scene)?;
    let mut description = SceneDescription::new();
    description.apply(&set).context("scene rejected")?;
    // The harness's own copy of the scene, applied the same sets in the same
    // order as the render thread's. It is where edits find their targets —
    // the session's description is out of reach behind the thread — so a
    // walk stays a sequence, each edit seeing what the last one left.
    let mut replica = SceneDescription::new();
    replica.apply(&set).context("scene rejected")?;

    let texture_cache = TextureCache::of(&images(&replica), args.cold_textures)?;

    // The harness owns the stopping rule, so it authors one: the render
    // parks at the settle count, and every edit below lands on an image that
    // has stopped moving — including the wait for the parked thread to wake,
    // which is part of what the edit costs. Authored into the scene rather
    // than handed to the session beside it, since the description is the
    // session's only authority on how long to render.
    //
    // The early stop is turned off explicitly, not left to the file: the
    // walk waits for an exact sample count, and a scene that asked to stop
    // at some noise level would settle below it and hang the harness.
    //
    // Denoising rides along for the same reason: it is a condition of the
    // measurement, and the file's own answer would otherwise decide it.
    //
    // The name is the scene's own, since prep allows exactly one settings
    // object and a second would be rejected.
    let settle = ChangeSet {
        ops: vec![Op::Settings(SettingsPatch {
            spp: Some(args.settle),
            noise_threshold: Some(None),
            denoise: Some(args.denoise),
            ..SettingsPatch::new(
                description
                    .settings()
                    .keys()
                    .next()
                    .map_or("settings", String::as_str),
            )
        })],
    };
    description
        .apply(&settle)
        .context("authoring the settle count")?;
    replica
        .apply(&settle)
        .context("authoring the settle count")?;

    let settings = description
        .settings()
        .values()
        .next()
        .cloned()
        .unwrap_or_default();
    let width = args.width.unwrap_or(settings.resolution[0]);
    let height = args.height.unwrap_or(settings.resolution[1]);

    let (scene, load) =
        cenote::scene::Scene::prep_timed(&gpu, &mut description).context("preparing the scene")?;
    let camera = *scene.camera();
    let renderer = cenote::render::Renderer::with_max_bounces(&gpu, settings.max_bounces)?;
    let mut session = Session::new(
        Arc::clone(&gpu),
        description,
        scene,
        renderer,
        camera,
        width,
        height,
        load,
    );
    // Applied here rather than inside `drag`, because the walk needs it too:
    // a reduced frame now follows an edit as well as a camera move.
    if args.preview {
        session.set_preview_target(Some(Session::PREVIEW_TARGET));
    }

    println!(
        "\n  {} — {}×{}, settling at {} spp, textures {}, denoise {}",
        args.scene.display(),
        width,
        height,
        args.settle,
        texture_cache.label(),
        if args.denoise { "on" } else { "off" },
    );
    // Wait out the first accumulation: the render parks at its cap, so the
    // first edit below lands on a settled image like every one after it.
    let (settled, settled_size) = wait_for(&mut session, "a settled frame", |frame| {
        frame.samples() >= args.settle
    })?;
    let first_ray = settled
        .interactivity
        .to_first_ray
        .context("the first frame carried no time-to-first-ray")?;
    // The same breakdown the session was handed, read back closed: the
    // sample and the unnamed remainder are only known once the mark lands.
    let load = settled
        .interactivity
        .load
        .context("the first frame carried no load breakdown")?;
    print_header();
    print_row(
        "first ray",
        millis(first_ray),
        &load,
        settled_size,
        settled.interactivity.to_readable.map(millis),
    );

    // Filled by whichever measurement this run is: the walk or the drag.
    let mut report = LatencyReport {
        device: gpu.device_summary().to_owned(),
        scene: args.scene.display().to_string(),
        size: (width, height),
        settle: args.settle,
        denoise: args.denoise,
        texture_cache,
        load_millis: millis(first_ray),
        load,
        edits: Vec::new(),
        drag: None,
    };

    if let Some(moves) = args.drag {
        let dragged = drag(&mut session, camera, moves, args.preview)?;
        print_drag(&dragged);
        report.drag = Some(dragged);
        return write_report(args, &report);
    }

    report.edits = walk(&mut session, &mut replica, camera, args, &settled)?;
    write_report(args, &report)
}

/// One pass down the edit vocabulary, printing each row as it lands.
///
/// `settled` is the frame the walk starts from: its marks are what the first
/// edit has to move.
fn walk(
    session: &mut Session,
    replica: &mut SceneDescription,
    camera: Camera,
    args: &LatencyArgs,
    settled: &Stats,
) -> anyhow::Result<Vec<Measurement>> {
    let kinds = args
        .only
        .clone()
        .unwrap_or_else(|| <EditKind as clap::ValueEnum>::value_variants().to_vec());
    // The marks each edit has to move for its frame to be its own.
    let mut marks = Marks {
        first_pixel: settled.interactivity.to_first_pixel,
        readable: settled.interactivity.to_readable,
    };
    let mut edits = Vec::new();
    for kind in kinds {
        let outcome = walk_one(session, replica, camera, kind, args.settle, &mut marks)?;
        let label = kind.label();
        match &outcome {
            Outcome::Measured {
                millis,
                phases,
                size,
                readable_millis,
            } => print_row(&label, *millis, phases, *size, *readable_millis),
            Outcome::Skipped(why) => println!("  {label:<LABEL_WIDTH$}skipped — {why}"),
            Outcome::NoOp => {
                println!("  {label:<LABEL_WIDTH$}no-op — every value it set was already in place");
            }
            Outcome::Rejected(why) => println!("  {label:<LABEL_WIDTH$}rejected — {why}"),
        }
        edits.push(Measurement { kind, outcome });
        // Between edits, so a failure in the render thread surfaces as
        // itself rather than as the next wait timing out.
        session.check()?;
    }
    Ok(edits)
}

/// Leave the RON sidecar and say where it went.
fn write_report(args: &LatencyArgs, report: &LatencyReport) -> anyhow::Result<()> {
    let text = ron::ser::to_string_pretty(report, ron::ser::PrettyConfig::default())
        .context("serializing the latency report")?;
    std::fs::write(&args.out, text).with_context(|| format!("writing {}", args.out.display()))?;
    println!("\nwrote {}\n", args.out.display());
    Ok(())
}

/// Drag the view: `moves` camera changes issued back to back, each as soon
/// as a frame carrying the last one arrives, so the render never settles and
/// never naps.
///
/// Deliberately not a row of the walk. The walk times one verb against a
/// *parked* render, where the idle nap is on the critical path and a change
/// in what a sample costs arrives diluted by it — the wrong instrument for
/// anything that makes samples cheaper. Sustained motion is what a person
/// orbiting does, and where the resolution divisor has to show itself.
///
/// Medians, not means: the first move pays for waking the parked render this
/// drag starts from, and that outlier should not move the figure.
///
/// `preview` is recorded as a condition of the measurement, like the
/// texture-cache state; the caller applies it, because the walk needs it too.
fn drag(
    session: &mut Session,
    view: Camera,
    moves: u32,
    preview: bool,
) -> anyhow::Result<DragReport> {
    let mut camera = view;
    let mut intervals = Vec::new();
    let mut first_pixels = Vec::new();
    let mut sizes = BTreeSet::new();
    let mut filters = Vec::new();
    let mut last_frame = Instant::now();
    for _ in 0..moves {
        camera.position.x = stepped(camera.position.x);
        session.set_camera(camera);
        let target = session.epoch();
        // Per move, as the walk allows per edit: a long drag on a heavy
        // scene is slow, not stuck.
        let deadline = Instant::now() + WAIT_LIMIT;
        loop {
            session.check()?;
            if let Some(frame) = session.take_frame() {
                // Every published frame is a tick of the cadence, including
                // the ones still carrying the previous move: the consumer
                // paints those too.
                intervals.push(last_frame.elapsed());
                last_frame = Instant::now();
                sizes.insert((frame.width(), frame.height()));
                filters.extend(frame.stats().denoise);
                if frame.epoch() >= target {
                    first_pixels.extend(frame.stats().interactivity.to_first_pixel);
                    break;
                }
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "gave up waiting {} s for one move of the drag to answer",
                WAIT_LIMIT.as_secs()
            );
            std::thread::sleep(POLL);
        }
    }
    let marks = first_pixels.iter().copied().map(millis).collect();
    Ok(DragReport {
        moves,
        preview,
        denoise_millis: (!filters.is_empty()).then(|| median(&mut filters)),
        frame_millis: median(&mut intervals),
        first_pixel_millis: median(&mut first_pixels),
        first_pixels: marks,
        sizes,
    })
}

/// The middle of `values`, in milliseconds — zero if there are none.
fn median(values: &mut [Duration]) -> f64 {
    values.sort_unstable();
    values.get(values.len() / 2).copied().map_or(0.0, millis)
}

/// Issue one edit and read back what it cost. `mark` is the last
/// time-to-first-pixel this walk saw, and it is how the harness knows the
/// frame it read is answering *this* verb.
///
/// That check is the load-bearing part. An edit whose values are already in
/// place restarts nothing (the change-set contract), and the renderer still
/// republishes the settled image under the new epoch — a frame that answers
/// the verb while carrying the *previous* interaction's breakdown. But a
/// frame stamped at or past the verb's epoch was published by a wave that
/// had already drained the verb (the loop reads the epoch before it drains,
/// so it can undercount a queued verb, never overcount one), which leaves an
/// unchanged mark meaning exactly one thing: the verb restarted nothing.
///
/// Asking the renderer beats predicting it. The harness could apply the set
/// to its replica and look for dirt, but then two implementations of "did
/// this change anything" have to agree, and the moment they don't the
/// harness reports a number belonging to a different edit.
fn walk_one(
    session: &mut Session,
    replica: &mut SceneDescription,
    view: Camera,
    kind: EditKind,
    settle: u32,
    marks: &mut Marks,
) -> anyhow::Result<Outcome> {
    let verb = match kind.synthesize(replica, view) {
        Ok(verb) => verb,
        Err(why) => return Ok(Outcome::Skipped(why)),
    };
    // Keep the replica in step with the render thread's description, so
    // later edits target what earlier ones left behind — and so a set the
    // renderer would refuse is refused here first, before it is timed.
    if let Verb::Edit(set) = &verb
        && let Err(error) = replica.apply(set)
    {
        return Ok(Outcome::Rejected(error.to_string()));
    }
    match verb {
        Verb::Edit(set) => session.apply(set),
        Verb::Move(camera) => session.set_camera(camera),
    }
    let target = session.epoch();
    let (stats, size) = wait_for(session, "the edited frame", |frame| frame.epoch() >= target)?;
    if let Some(error) = session.take_edit_error() {
        return Ok(Outcome::Rejected(error.to_string()));
    }
    if stats.interactivity.to_first_pixel == marks.first_pixel {
        return Ok(Outcome::NoOp);
    }
    marks.first_pixel = stats.interactivity.to_first_pixel;
    let phases = stats
        .interactivity
        .interaction
        .context("the edited frame carried no breakdown")?;
    // Then wait out the accumulation this edit restarted. Two things need
    // it: the readable mark below is not this edit's until the image it
    // measures has arrived, and the *next* edit only lands on a settled
    // image — the claim `--settle` makes — if this one finished first.
    // Skipped for the outcomes above, where nothing restarted and the render
    // is still parked with no further frame to wait for.
    let (settled, _) = wait_for(session, "the edit to settle", |frame| {
        frame.samples() >= settle
    })?;
    // A mark that did not move did not land: below `READABLE_SAMPLES` the
    // render parks before it, and reporting the previous edit's number would
    // be a lie in the shape of a measurement.
    let readable = settled.interactivity.to_readable;
    let landed = if readable == marks.readable {
        None
    } else {
        readable
    };
    marks.readable = readable;
    Ok(Outcome::Measured {
        millis: millis(phases.total()),
        phases,
        size,
        readable_millis: landed.map(millis),
    })
}

/// Poll published frames until one satisfies `accept`, and hand back its
/// statistics and the size it rendered at. Frames are dropped as they are
/// read: the render thread has two publish slots and holding one stalls it —
/// which is also why the size is copied out here rather than the frame
/// handed back.
fn wait_for(
    session: &mut Session,
    what: &str,
    accept: impl Fn(&Frame) -> bool,
) -> anyhow::Result<(Stats, (u32, u32))> {
    let deadline = Instant::now() + WAIT_LIMIT;
    loop {
        session.check()?;
        if let Some(frame) = session.take_frame()
            && accept(&frame)
        {
            return Ok((frame.stats().clone(), (frame.width(), frame.height())));
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "gave up waiting {} s for {what}",
            WAIT_LIMIT.as_secs()
        );
        std::thread::sleep(POLL);
    }
}

/// Milliseconds, the one unit this harness reports in.
fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// The column headers, taken from the breakdown itself so a phase added to
/// [`Phases`] appears here without anyone remembering to add it.
fn print_header() {
    print!("\n  {:<LABEL_WIDTH$}{:>9}", "edit", "total");
    for (name, _) in Phases::default().named() {
        print!("{name:>9}");
    }
    println!("{:>12}{:>11}", "size", "readable");
}

/// The drag's three lines, in the same column as the walk's numbers.
fn print_drag(report: &DragReport) {
    let sizes: Vec<_> = report
        .sizes
        .iter()
        .map(|(width, height)| format!("{width}×{height}"))
        .collect();
    println!(
        "\n  {} moves, never settling, preview resolution {}",
        report.moves,
        if report.preview { "on" } else { "off" },
    );
    println!(
        "  {:<LABEL_WIDTH$}{:>9.2}",
        "frame interval", report.frame_millis
    );
    println!(
        "  {:<LABEL_WIDTH$}{:>9.2}",
        "to first pixel", report.first_pixel_millis
    );
    if let Some(denoise) = report.denoise_millis {
        println!("  {:<LABEL_WIDTH$}{denoise:>9.2}", "of which denoise");
    }
    println!("  {:<LABEL_WIDTH$}{:>9}", "frame size", sizes.join(", "));
}

/// One row, in the same order the header came out in. The two trailing
/// columns are the conditions the total has to be read against: what it
/// rendered, and what waiting for the readable image after it cost.
fn print_row(label: &str, total: f64, phases: &Phases, size: (u32, u32), readable: Option<f64>) {
    print!("  {label:<LABEL_WIDTH$}{total:>9.2}");
    for (_, value) in phases.named() {
        print!("{:>9.2}", millis(value));
    }
    print!("{:>12}", format!("{}×{}", size.0, size.1));
    match readable {
        Some(readable) => println!("{readable:>11.2}"),
        None => println!("{:>11}", "—"),
    }
}
