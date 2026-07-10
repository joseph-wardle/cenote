//! The `.ron` scene-file boundary: a scene file is one serialized
//! [`ChangeSet`] behind a format-version field, applied against the empty
//! description at load — file, wire, and edit are the same value (the
//! schema itself lives in [`crate::scene::description`] and
//! [`crate::scene::changeset`]; serde derives *are* the parser, so schema
//! and format cannot drift).
//!
//! The version field leads the file and is probed before the full parse,
//! so a file from a different format generation fails with "version 3, this
//! build reads 1" instead of a field-level parse error. Typos inside a
//! recognized version do fail loudly: unknown fields are rejected, never
//! skipped — a misspelled parameter silently ignored would be a wrong
//! render with no error, the worst outcome a scene format can produce.
//!
//! Serialization is plain RON — explicit `Some`, externally tagged enums,
//! no extensions. (`implicit_some` would collapse the `Some(None)` patches
//! that clear an optional field, and RON's documented weak spots —
//! `untagged`, `flatten` — stay out of the schema entirely.)

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::scene::changeset::{ChangeSet, Op};

/// The scene-file format generation this build reads and writes. Bumped
/// only for incompatible schema changes; compatible growth (a new optional
/// field with a default) is not a bump.
pub const FORMAT_VERSION: u32 = 1;

/// The file schema: the version, then the ops. Split into owned/borrowed
/// twins so writing never clones a mesh payload.
#[derive(Serialize)]
struct SceneFileOut<'a> {
    version: u32,
    ops: &'a [Op],
}

/// See [`SceneFileOut`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SceneFileIn {
    #[expect(
        dead_code,
        reason = "the probe reads it; declared so deny_unknown_fields accepts the field"
    )]
    version: u32,
    ops: Vec<Op>,
}

/// The version field alone, parsed leniently — readable even when the ops
/// schema has drifted, so version mismatches diagnose themselves.
#[derive(Deserialize)]
struct VersionProbe {
    version: u32,
}

/// Serialize a change-set as scene-file text.
///
/// # Errors
///
/// [`Error::SceneFormat`] if serialization fails (a path that is not
/// valid UTF-8 is the realistic case).
pub fn to_ron(set: &ChangeSet) -> Result<String> {
    let file = SceneFileOut {
        version: FORMAT_VERSION,
        ops: &set.ops,
    };
    ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default())
        .map_err(|error| Error::SceneFormat(format!("serialization failed: {error}")))
}

/// Parse scene-file text into a change-set. Paths come through verbatim —
/// [`load`] is the entry point that also rebases them.
///
/// # Errors
///
/// [`Error::SceneFormat`] if the text is not a scene file, is from a
/// different format version, or fails to parse (including unknown fields —
/// see the module doc).
pub fn from_ron(text: &str) -> Result<ChangeSet> {
    let probe: VersionProbe = ron::from_str(text).map_err(|error| {
        Error::SceneFormat(format!(
            "not a cenote scene file (expected `(version: {FORMAT_VERSION}, ops: [...])`): {error}"
        ))
    })?;
    if probe.version != FORMAT_VERSION {
        return Err(Error::SceneFormat(format!(
            "scene file is format version {}, this build reads {FORMAT_VERSION}",
            probe.version
        )));
    }
    let file: SceneFileIn = ron::from_str(text)
        .map_err(|error| Error::SceneFormat(format!("scene file parse failed: {error}")))?;
    Ok(ChangeSet { ops: file.ops })
}

/// Read a scene file and rebase its relative paths against the file's own
/// directory — the only place relative paths gain a meaning, which is why
/// apply refuses them.
///
/// # Errors
///
/// [`Error::Io`] if the file cannot be read, or anything [`from_ron`]
/// returns.
pub fn load(path: &Path) -> Result<ChangeSet> {
    let text = std::fs::read_to_string(path)?;
    let mut set = from_ron(&text)?;
    let base = std::path::absolute(path)?;
    // An absolute path to a readable file always has a parent; the
    // fallback is unreachable but cheaper than a documented panic.
    set.rebase_paths(base.parent().unwrap_or(Path::new("/")));
    Ok(set)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::scene::changeset::{
        CameraPatch, Kind, LightPatch, MaterialPatch, MeshPatch, SettingsPatch,
    };
    use crate::scene::description::{Light, MeshSource, Texturable, TextureRef};

    /// One of everything the serializer must not mangle: every op shape,
    /// a texture, both double-`Option` states, a `Remove`.
    fn kitchen_sink() -> ChangeSet {
        ChangeSet {
            ops: vec![
                Op::Settings(SettingsPatch {
                    resolution: Some([640, 480]),
                    ..SettingsPatch::new("main")
                }),
                Op::Camera(CameraPatch {
                    // `Some(None)`: an explicit "clear the focus distance".
                    focus_distance: Some(None),
                    ..CameraPatch::new("main")
                }),
                Op::Camera(CameraPatch {
                    focus_distance: Some(Some(2.5)),
                    aperture_radius: Some(0.05),
                    ..CameraPatch::new("dof")
                }),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Inline {
                        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        normals: None,
                        uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                        triangles: vec![[0, 1, 2]],
                    }),
                    ..MeshPatch::new("tri")
                }),
                Op::Mesh(MeshPatch {
                    source: Some(MeshSource::Ply {
                        path: "geo/bunny.ply".into(),
                    }),
                    ..MeshPatch::new("bunny")
                }),
                Op::Material(Box::new(MaterialPatch {
                    base_color: Some(Texturable::Texture(TextureRef {
                        path: "wood.png".into(),
                        color_space: None,
                    })),
                    geometry_normal: Some(Some(TextureRef {
                        path: "wood_n.png".into(),
                        color_space: None,
                    })),
                    ..MaterialPatch::new("wood")
                })),
                Op::Material(Box::new(MaterialPatch {
                    geometry_normal: Some(None),
                    ..MaterialPatch::new("wood")
                })),
                Op::Light(LightPatch {
                    light: Some(Light::Distant {
                        direction: [0.0, -1.0, 0.0],
                        irradiance: [2.0; 3],
                    }),
                    ..LightPatch::new("sun")
                }),
                Op::Remove(Kind::Material, "old".into()),
            ],
        }
    }

    #[test]
    fn everything_round_trips() {
        let set = kitchen_sink();
        let text = to_ron(&set).expect("serializes");
        let parsed = from_ron(&text).expect("parses");
        assert_eq!(parsed, set);
    }

    #[test]
    fn the_version_field_leads_the_file() {
        let text = to_ron(&ChangeSet::default()).expect("serializes");
        let body = text.trim_start_matches(['(', '\n', ' ']);
        assert!(body.starts_with("version: 1"), "{text}");
    }

    #[test]
    fn a_newer_version_is_refused_by_number() {
        let error = from_ron("(version: 999, ops: [])").unwrap_err();
        assert!(error.to_string().contains("999"), "{error}");
        assert!(error.to_string().contains("reads 1"), "{error}");
    }

    #[test]
    fn non_scene_text_is_refused() {
        let error = from_ron("hello").unwrap_err();
        assert!(error.to_string().contains("not a cenote scene"), "{error}");
    }

    #[test]
    fn unknown_fields_are_refused_not_skipped() {
        // A typo at the file level…
        assert!(from_ron("(version: 1, ops: [], extra: 5)").is_err());
        // …and inside a patch: `base_colour` must not silently no-op.
        let error = from_ron(
            "(version: 1, ops: [Material((name: \"m\", base_colour: Constant((1.0, 0.0, 0.0))))])",
        )
        .unwrap_err();
        assert!(error.to_string().contains("base_colour"), "{error}");
    }

    #[test]
    fn load_rebases_relative_paths_against_the_file() {
        let dir = std::env::temp_dir().join(format!("cenote-format-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let scene = dir.join("scene.ron");
        std::fs::write(
            &scene,
            "(version: 1, ops: [Mesh((name: \"m\", source: Some(Ply(path: \"geo/mesh.ply\"))))])",
        )
        .expect("write scene");
        let set = load(&scene).expect("loads");
        let Op::Mesh(mesh) = &set.ops[0] else {
            panic!("unexpected op");
        };
        assert_eq!(
            mesh.source,
            Some(MeshSource::Ply {
                path: dir.join("geo/mesh.ply")
            })
        );
        std::fs::remove_file(&scene).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn a_hand_written_scene_parses() {
        // The shape a stranger would write after reading one example —
        // sparse patches, defaults everywhere else.
        let text = r#"(
            version: 1,
            ops: [
                Settings((name: "main", spp: Some(16))),
                Camera((name: "main", position: Some((0.0, 1.0, 4.0)))),
                Mesh((name: "tri", source: Some(Inline(
                    positions: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    triangles: [(0, 1, 2)],
                )))),
                Material((name: "clay", base_color: Some(Constant((0.7, 0.3, 0.2))))),
                Instance((name: "thing", mesh: Some("tri"), material: Some("clay"))),
            ],
        )"#;
        let set = from_ron(text).expect("parses");
        assert_eq!(set.ops.len(), 5);
        let mut description = crate::scene::description::SceneDescription::new();
        description.apply(&set).expect("applies");
        assert_eq!(description.settings()["main"].spp, 16);
        // Fields the file never mentioned hold their documented defaults.
        assert_eq!(description.settings()["main"].resolution, [1280, 720]);
    }

    /// The repo's example scene stays in step with the schema: it must
    /// load, apply, and reference files that exist — the file the viewer's
    /// live-edit walkthrough opens can never silently rot.
    #[test]
    fn the_example_scene_loads_and_applies() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenes/example.ron");
        let set = load(&path).expect("example scene loads");
        let mut description = crate::scene::description::SceneDescription::new();
        description.apply(&set).expect("example scene applies");
        assert!(!description.instances().is_empty());
        assert_eq!(description.cameras().len(), 1);
    }

    #[test]
    fn paths_survive_serialization() {
        let set = ChangeSet {
            ops: vec![Op::Mesh(MeshPatch {
                source: Some(MeshSource::Ply {
                    path: PathBuf::from("/abs/with spaces/mesh.ply"),
                }),
                ..MeshPatch::new("m")
            })],
        };
        let text = to_ron(&set).expect("serializes");
        assert_eq!(from_ron(&text).expect("parses"), set);
    }
}
