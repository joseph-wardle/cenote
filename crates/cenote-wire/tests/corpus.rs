//! The byte-exact drift guard: the golden corpus that pins the
//! wire's encoded bytes across languages, with Rust as the authority.
//!
//! Every case here encodes to a checked-in golden in `tests/golden/`. The
//! C++ encoder (`hydra/wire/`) must reproduce each
//! golden **byte for byte** — that agreement is the compiler-substitute
//! this project chose in place of gRPC's codegen. Two assertions per case
//! keep the corpus honest in both directions:
//!
//! - encoding the case must equal the golden (the encoder didn't drift);
//! - decoding the golden must equal the case (the golden didn't rot — a
//!   stale file fails here rather than silently blessing old bytes).
//!
//! After an **intentional** wire change, regenerate consciously — in the
//! same commit as the change, with the protocol/layout version bumped and
//! the C++ mirror updated:
//!
//! ```sh
//! UPDATE_GOLDENS=1 cargo test -p cenote-wire --test corpus
//! ```
//!
//! Coverage, per the step-0 plan: every `Op` variant; every patch field
//! `Some`; both `MeshSource` and both `Transform` spellings; the instance
//! transforms array empty, single, and multi-element; both `Light`
//! and both `Texturable` variants; the texture channel absent and in all
//! four spellings; the doubly-optional fields in all three states (the
//! material's normal map, the camera's focus, the environment's image,
//! the settings' noise threshold);
//! a `Remove` of every `Kind`; an empty set; unicode in names, paths, and
//! messages; and every `Request`/`Response` variant.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use cenote_wire::protocol::{Camera, FbDesc, PROTOCOL, Request, Response, decode, encode};
use cenote_wire::scene::{
    CameraPatch, ChangeSet, Channel, ColorSpace, CurveBasis, CurveType, CurveWrap, CurvesPatch,
    CurvesSource, EnvironmentPatch, InstancePatch, Kind, Light, LightPatch, MaterialPatch,
    MeshPatch, MeshSource, Op, Reset, SettingsPatch, TextureRef, Texturable, Transform,
    WidthInterpolation, Widths,
};

/// One pinned message — the corpus is heterogeneous, so each case carries
/// its own encode/decode-check pair.
enum Case {
    Request(Request),
    Response(Response),
}

impl Case {
    fn encode(&self) -> Vec<u8> {
        match self {
            Self::Request(message) => encode(message).expect("requests always encode"),
            Self::Response(message) => encode(message).expect("responses always encode"),
        }
    }

    /// Decode `golden` as this case's type and require the original value
    /// back — the rot check.
    fn assert_decodes_from(&self, name: &str, golden: &[u8]) {
        match self {
            Self::Request(message) => {
                let back: Request = decode(golden)
                    .unwrap_or_else(|error| panic!("golden \"{name}\" does not decode: {error}"));
                assert_eq!(&back, message, "golden \"{name}\" decoded to a different value");
            }
            Self::Response(message) => {
                let back: Response = decode(golden)
                    .unwrap_or_else(|error| panic!("golden \"{name}\" does not decode: {error}"));
                assert_eq!(&back, message, "golden \"{name}\" decoded to a different value");
            }
        }
    }
}

/// The corpus: names are the golden file stems, values the pinned
/// messages. Adding a wire field means extending a case here (or adding
/// one) and regenerating — the encoder change alone fails the test.
#[expect(
    clippy::too_many_lines,
    reason = "the corpus is deliberately exhaustive — one literal per covered shape"
)]
fn corpus() -> Vec<(&'static str, Case)> {
    vec![
        (
            "request-hello",
            Case::Request(Request::Hello {
                protocol: PROTOCOL,
                token: "0123456789abcdef0123456789abcdef".into(),
            }),
        ),
        ("request-ping", Case::Request(Request::Ping)),
        (
            "request-resize",
            Case::Request(Request::Resize {
                width: 1920,
                height: 1080,
            }),
        ),
        (
            "request-set-camera-focused",
            Case::Request(Request::SetCamera(Camera {
                position: [1.0, 2.5, -3.0],
                look_at: [0.0, 1.0, 0.0],
                up: [0.0, 1.0, 0.0],
                vfov_degrees: 36.5,
                focus_distance: Some(4.25),
                aperture_radius: 0.05,
            })),
        ),
        (
            "request-set-camera-auto-focus",
            Case::Request(Request::SetCamera(Camera {
                position: [0.0, 0.0, 5.0],
                look_at: [0.0; 3],
                up: [0.0, 1.0, 0.0],
                vfov_degrees: 40.0,
                focus_distance: None,
                aperture_radius: 0.0,
            })),
        ),
        (
            "request-replace-empty",
            Case::Request(Request::Replace(ChangeSet::default())),
        ),
        (
            "request-replace-genesis",
            Case::Request(Request::Replace(genesis())),
        ),
        (
            "request-apply-removes",
            Case::Request(Request::Apply(ChangeSet {
                ops: vec![
                    Op::Remove(Kind::Mesh, "mesh".into()),
                    Op::Remove(Kind::Curves, "curves".into()),
                    Op::Remove(Kind::Instance, "instance".into()),
                    Op::Remove(Kind::Material, "材料/µaterial".into()),
                    Op::Remove(Kind::Light, "light".into()),
                    Op::Remove(Kind::Camera, "camera".into()),
                    Op::Remove(Kind::Environment, "environment".into()),
                    Op::Remove(Kind::Settings, "settings".into()),
                ],
            })),
        ),
        (
            "response-welcome",
            Case::Response(Response::Welcome {
                protocol: PROTOCOL,
                fb: FbDesc {
                    shm_name: "/cenote-12345-1".into(),
                    bytes: 4096 + 2 * (1280 * 720 * 20),
                    width: 1280,
                    height: 720,
                },
            }),
        ),
        (
            "response-ack-clean",
            Case::Response(Response::Ack {
                rejected: vec![],
                epoch: 7,
            }),
        ),
        (
            "response-ack-rejected",
            Case::Response(Response::Ack {
                rejected: vec![
                    "instance \"chair\" references a mesh \"seat\" that does not exist".into(),
                    "environment \"ciel-d'été\" references \"/scènes/небо.exr\", which does not \
                     exist"
                        .into(),
                ],
                // Past u32, deliberately: the epoch is a u64 on the wire,
                // and this pins the width for the C++ mirror.
                epoch: 4_294_967_296,
            }),
        ),
        (
            "response-resized",
            Case::Response(Response::Resized {
                fb: FbDesc {
                    shm_name: "/cenote-12345-2".into(),
                    bytes: 4096 + 2 * (640 * 480 * 20),
                    width: 640,
                    height: 480,
                },
                epoch: 9,
            }),
        ),
    ]
}

/// The kitchen-sink genesis: every op variant, every patch field `Some`,
/// both spellings of every two-spelling type, the doubly-optionals in all
/// three states, and unicode where the wire carries text.
#[expect(
    clippy::too_many_lines,
    reason = "the corpus is deliberately exhaustive — one literal per covered shape"
)]
fn genesis() -> ChangeSet {
    ChangeSet {
        ops: vec![
            Op::Mesh(MeshPatch {
                name: "tri".into(),
                source: Some(MeshSource::Inline {
                    positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    normals: Some(vec![[0.0, 0.0, 1.0]; 3]),
                    uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                    triangles: vec![[0, 1, 2]],
                }),
            }),
            Op::Mesh(MeshPatch {
                name: "статуя".into(),
                source: Some(MeshSource::Ply {
                    path: "/scènes/geo/статуя.ply".into(),
                }),
            }),
            // The curve batch twice over: cells inline with every field
            // authored, and a groom by reference — both `CurvesSource`
            // spellings, and every enum the schema carries.
            Op::Curves(CurvesPatch {
                name: "groom".into(),
                source: Some(CurvesSource::Inline {
                    points: vec![
                        [0.0; 3],
                        [0.0, 1.0, 0.0],
                        [0.5, 2.0, 0.0],
                        [1.0, 3.0, 0.0],
                        [1.5, 4.0, 0.0],
                    ],
                    curve_vertex_counts: vec![5],
                    widths: Some(Widths {
                        values: vec![0.02, 0.015, 0.01, 0.008, 0.004],
                        interpolation: WidthInterpolation::Vertex,
                    }),
                    curve_type: CurveType::Cubic,
                    basis: CurveBasis::CatmullRom,
                    wrap: CurveWrap::Pinned,
                }),
            }),
            Op::Curves(CurvesPatch {
                name: "волосы".into(),
                source: Some(CurvesSource::Hair {
                    path: "/scènes/geo/волосы.hair".into(),
                }),
            }),
            Op::Instance(InstancePatch {
                name: "thing".into(),
                mesh: Some("tri".into()),
                curves: None,
                material: Some("m-set".into()),
                transforms: Some(vec![Transform::Trs {
                    translate: [1.0, 2.0, 3.0],
                    rotate_degrees: [0.0, 90.0, 0.0],
                    scale: [2.0, 2.0, 2.0],
                }]),
                camera_visible: Some(false),
            }),
            // A multi-element transforms array, both spellings in one —
            // the array-instancer form the field's Vec exists for.
            Op::Instance(InstancePatch {
                name: "matrix-thing".into(),
                mesh: Some("статуя".into()),
                curves: None,
                material: Some("m-clear".into()),
                transforms: Some(vec![
                    Transform::Matrix([
                        [1.0, 0.0, 0.0, 4.0],
                        [0.0, 1.0, 0.0, 5.0],
                        [0.0, 0.0, 1.0, 6.0],
                    ]),
                    Transform::Trs {
                        translate: [-4.0, 0.0, 4.0],
                        rotate_degrees: [0.0, 0.0, 45.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                ]),
                camera_visible: Some(true),
            }),
            // The empty array is legal and distinct from an absent field:
            // resident, places nothing (a fully-masked instancer).
            Op::Instance(InstancePatch {
                name: "masked".into(),
                mesh: Some("tri".into()),
                curves: None,
                material: Some("m-leave".into()),
                transforms: Some(vec![]),
                camera_visible: None,
            }),
            // The other geometry reference: an instance that places a
            // groom rather than a mesh.
            Op::Instance(InstancePatch {
                name: "hair".into(),
                mesh: None,
                curves: Some("groom".into()),
                material: Some("m-set".into()),
                transforms: None,
                camera_visible: None,
            }),
            // The material patch three times over: every field `Some`
            // with the normal map set, then the clear, then the
            // leave-alone — the three states of the doubly-optional. The
            // texture references between them cover the source channel
            // absent and in all four spellings.
            Op::Material(Box::new(MaterialPatch {
                name: "m-set".into(),
                base_color: Some(Texturable::Texture(TextureRef {
                    path: "/textures/дерево.png".into(),
                    color_space: Some(ColorSpace::Srgb),
                    channel: None,
                })),
                base_diffuse_roughness: Some(0.25),
                base_metalness: Some(Texturable::Constant(1.0)),
                specular_weight: Some(0.9),
                specular_roughness: Some(Texturable::Texture(TextureRef {
                    path: "/textures/rough.png".into(),
                    color_space: Some(ColorSpace::Linear),
                    channel: Some(Channel::G),
                })),
                specular_ior: Some(1.45),
                transmission_weight: Some(0.5),
                transmission_color: Some([0.9, 0.95, 1.0]),
                transmission_depth: Some(0.1),
                transmission_scatter: Some([0.05, 0.4, 0.7]),
                transmission_scatter_anisotropy: Some(0.6),
                subsurface_weight: Some(Texturable::Constant(0.75)),
                // The subsurface albedo is the slot a real asset textures
                // — a head scan's map — so it carries the reference here
                // and its four neighbours carry constants.
                subsurface_color: Some(Texturable::Texture(TextureRef {
                    path: "/textures/albedo.png".into(),
                    color_space: None,
                    channel: None,
                })),
                subsurface_radius: Some(Texturable::Constant(0.01)),
                subsurface_radius_scale: Some(Texturable::Constant([1.0, 0.5, 0.25])),
                subsurface_scatter_anisotropy: Some(Texturable::Constant(-0.3)),
                coat_weight: Some(1.0),
                coat_color: Some([1.0, 0.9, 0.8]),
                coat_roughness: Some(0.05),
                coat_ior: Some(1.6),
                coat_darkening: Some(0.75),
                fuzz_weight: Some(0.2),
                fuzz_color: Some([1.0, 1.0, 0.9]),
                fuzz_roughness: Some(0.6),
                emission_luminance: Some(1000.0),
                emission_color: Some(Texturable::Constant([1.0, 0.5, 0.25])),
                geometry_opacity: Some(Texturable::Texture(TextureRef {
                    path: "/textures/mask.png".into(),
                    color_space: None,
                    channel: Some(Channel::A),
                })),
                geometry_thin_walled: Some(true),
                // A channel on a normal slot is inert server-side; the
                // mirror is total, so its bytes are pinned regardless.
                geometry_normal: Some(Reset::Set(TextureRef {
                    path: "/textures/normal.png".into(),
                    color_space: None,
                    channel: Some(Channel::B),
                })),
            })),
            Op::Material(Box::new(MaterialPatch {
                name: "m-clear".into(),
                geometry_normal: Some(Reset::Clear),
                ..MaterialPatch::default()
            })),
            Op::Material(Box::new(MaterialPatch {
                name: "m-leave".into(),
                base_metalness: Some(Texturable::Constant(0.0)),
                specular_roughness: Some(Texturable::Texture(TextureRef {
                    path: "/textures/orm.png".into(),
                    color_space: None,
                    channel: Some(Channel::R),
                })),
                geometry_opacity: Some(Texturable::Constant(1.0)),
                ..MaterialPatch::default()
            })),
            Op::Light(LightPatch {
                name: "sun".into(),
                light: Some(Light::Distant {
                    direction: [-0.3, -1.0, -0.2],
                    irradiance: [3.0, 2.9, 2.7],
                }),
            }),
            Op::Light(LightPatch {
                name: "bulb".into(),
                light: Some(Light::Point {
                    position: [0.0, 2.0, 0.0],
                    intensity: [10.0, 9.0, 8.0],
                }),
            }),
            // The camera patch's focus distance is the other
            // doubly-optional: set, clear, leave alone.
            Op::Camera(CameraPatch {
                name: "cam-set".into(),
                position: Some([0.0, 1.0, 5.0]),
                // Every vector distinct, so a field swap in either encoder
                // cannot produce identical bytes.
                look_at: Some([0.0, 1.5, 0.0]),
                up: Some([0.0, 1.0, 0.0]),
                vfov_degrees: Some(45.0),
                focus_distance: Some(Reset::Set(5.0)),
                aperture_radius: Some(0.02),
            }),
            Op::Camera(CameraPatch {
                name: "cam-clear".into(),
                focus_distance: Some(Reset::Clear),
                ..CameraPatch::default()
            }),
            Op::Camera(CameraPatch {
                name: "cam-leave".into(),
                vfov_degrees: Some(30.0),
                ..CameraPatch::default()
            }),
            // The environment patch three times over, like the material:
            // the image set, cleared (back to the constant white sky),
            // and left alone — the path's three doubly-optional states.
            Op::Environment(EnvironmentPatch {
                name: "ciel-d'été".into(),
                path: Some(Reset::Set("/scènes/небо.exr".into())),
                tint: Some([1.0, 0.9, 0.8]),
                transform: Some(Transform::Trs {
                    translate: [0.0; 3],
                    rotate_degrees: [0.0, 45.0, 0.0],
                    scale: [1.0; 3],
                }),
            }),
            Op::Environment(EnvironmentPatch {
                name: "env-clear".into(),
                path: Some(Reset::Clear),
                ..EnvironmentPatch::default()
            }),
            Op::Environment(EnvironmentPatch {
                name: "env-leave".into(),
                tint: Some([0.5, 0.5, 0.5]),
                ..EnvironmentPatch::default()
            }),
            // The settings patch three times over: the threshold set,
            // cleared (spend the whole budget), and left alone — the
            // fourth doubly-optional's three states.
            Op::Settings(SettingsPatch {
                name: "main".into(),
                spp: Some(256),
                noise_threshold: Some(Reset::Set(0.02)),
                max_bounces: Some(8),
                denoise: Some(true),
                seed: Some(7),
            }),
            Op::Settings(SettingsPatch {
                name: "settings-clear".into(),
                noise_threshold: Some(Reset::Clear),
                ..SettingsPatch::default()
            }),
            Op::Settings(SettingsPatch {
                name: "settings-leave".into(),
                spp: Some(64),
                ..SettingsPatch::default()
            }),
            Op::Remove(Kind::Mesh, "outgrown".into()),
        ],
    }
}

fn golden_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden"))
}

/// The C++ mirror's version literals, scraped from its headers — the one
/// wire surface the byte corpus cannot see. A schema change regenerated
/// into the goldens with a `PROTOCOL` bump on only one side of the
/// language seam fails here instead of at a customer's handshake.
#[test]
fn the_cpp_mirror_agrees_on_the_version_literals() {
    let scrape = |file: &str, name: &str| -> u32 {
        let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../hydra/wire/"))
            .join(file);
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let needle = format!("{name} = ");
        text.lines()
            .find_map(|line| {
                let value = line.split(&needle).nth(1)?;
                value.trim_end_matches(';').trim().parse().ok()
            })
            .unwrap_or_else(|| panic!("no `{name} = <int>;` in {}", path.display()))
    };
    assert_eq!(
        scrape("protocol.hpp", "PROTOCOL"),
        cenote_wire::protocol::PROTOCOL
    );
    assert_eq!(
        scrape("fb.hpp", "LAYOUT_VERSION"),
        cenote_wire::fb::LAYOUT_VERSION
    );
}

/// The guard itself. Under `UPDATE_GOLDENS=1` it rewrites the corpus
/// (and removes stale files) instead of asserting.
#[test]
fn the_corpus_is_byte_exact() {
    let dir = golden_dir();
    let update = std::env::var_os("UPDATE_GOLDENS").is_some();
    if update {
        fs::create_dir_all(&dir).expect("golden dir");
    }

    let cases = corpus();
    let mut expected_files = BTreeSet::new();
    for (name, case) in &cases {
        let path = dir.join(format!("{name}.bin"));
        expected_files.insert(path.file_name().expect("named").to_owned());
        let bytes = case.encode();
        if update {
            fs::write(&path, &bytes).expect("write golden");
            continue;
        }
        let golden = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "golden \"{name}\" is missing ({error}) — if this case is new, regenerate with \
                 UPDATE_GOLDENS=1 in the same commit"
            )
        });
        assert_eq!(
            bytes, golden,
            "the encoding of \"{name}\" drifted from its golden; if the wire change is \
             intentional, bump PROTOCOL, update the C++ mirror, and regenerate with \
             UPDATE_GOLDENS=1 in the same commit"
        );
        case.assert_decodes_from(name, &golden);
    }

    // A golden with no case is a lie waiting to be believed — the C++
    // side would faithfully reproduce bytes nothing checks anymore.
    let stale: Vec<String> = fs::read_dir(&dir)
        .expect("golden dir exists")
        .map(|entry| entry.expect("dir entry"))
        .filter(|entry| !expected_files.contains(&entry.file_name()))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    if update {
        for name in &stale {
            fs::remove_file(dir.join(name)).expect("prune stale golden");
        }
    } else {
        assert!(stale.is_empty(), "stale goldens with no case: {stale:?}");
    }
}
