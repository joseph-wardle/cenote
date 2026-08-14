//! Wire → renderer translation, in the one crate where both vocabularies
//! are in scope. Every function here **exhaustively destructures**
//! its wire struct — no `..`, no `_` on a data-bearing field — so a field
//! added to either side is a compile error in this file, never a value
//! silently dropped on the floor. The byte-exact corpus guards the
//! encoding; this file is the same discipline applied to the meaning.
//!
//! The translation is mechanical by design: names copy, `Option`s map,
//! wire strings become `PathBuf`s, and [`wire::Reset`] unfolds back into
//! the renderer's doubly-optional `Option<Option<T>>` (the wire can't
//! spell `Some(None)` — see `cenote_wire::scene`).

use std::convert;
use std::path::PathBuf;

use cenote::scene::changeset as core;
use cenote::scene::description;
use cenote_wire::protocol;
use cenote_wire::scene as wire;
use glam::Vec3;

/// A whole change-set, op for op.
#[must_use]
pub fn change_set(set: wire::ChangeSet) -> core::ChangeSet {
    let wire::ChangeSet { ops } = set;
    core::ChangeSet {
        ops: ops.into_iter().map(op).collect(),
    }
}

/// The active camera off the `SetCamera` lane, lowered the way scene prep
/// lowers a description camera: a positive aperture makes the thin lens,
/// and an unset focus distance means focus-at-`look_at`.
#[must_use]
pub fn camera(camera: protocol::Camera) -> cenote::scene::Camera {
    let protocol::Camera {
        position,
        look_at,
        up,
        vfov_degrees,
        focus_distance,
        aperture_radius,
    } = camera;
    let position = Vec3::from(position);
    let look_at = Vec3::from(look_at);
    cenote::scene::Camera {
        position,
        look_at,
        up: up.into(),
        vfov_degrees,
        lens: (aperture_radius > 0.0).then(|| cenote::scene::Lens {
            aperture_radius,
            focus_distance: focus_distance.unwrap_or_else(|| position.distance(look_at)),
        }),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one arm per op variant — splitting would scatter the exhaustive \
              destructuring this file exists to concentrate"
)]
fn op(op: wire::Op) -> core::Op {
    match op {
        wire::Op::Mesh(patch) => {
            let wire::MeshPatch { name, source } = patch;
            core::Op::Mesh(core::MeshPatch {
                name,
                source: source.map(mesh_source),
            })
        }
        wire::Op::Curves(patch) => {
            let wire::CurvesPatch { name, source } = patch;
            core::Op::Curves(core::CurvesPatch {
                name,
                source: source.map(curves_source),
            })
        }
        wire::Op::Instance(patch) => {
            let wire::InstancePatch {
                name,
                mesh,
                curves,
                material,
                transforms,
                camera_visible,
            } = patch;
            core::Op::Instance(core::InstancePatch {
                name,
                mesh,
                curves,
                material,
                transforms: transforms
                    .map(|list| list.into_iter().map(self::transform).collect()),
                camera_visible,
                // The wire protocol has no volume prims yet — Hydra's
                // `UsdVolVolume` reader is Track A's, and until it lands
                // nothing on this side can bound a medium — nor author the
                // priority that only matters where interiors overlap.
                medium: None,
                interior_priority: None,
            })
        }
        wire::Op::Material(patch) => {
            let wire::MaterialPatch {
                name,
                base_color,
                base_diffuse_roughness,
                base_metalness,
                specular_weight,
                specular_roughness,
                specular_ior,
                transmission_weight,
                transmission_color,
                transmission_depth,
                transmission_scatter,
                transmission_scatter_anisotropy,
                subsurface_weight,
                subsurface_color,
                subsurface_radius,
                subsurface_radius_scale,
                subsurface_scatter_anisotropy,
                coat_weight,
                coat_color,
                coat_roughness,
                coat_ior,
                coat_darkening,
                fuzz_weight,
                fuzz_color,
                fuzz_roughness,
                emission_luminance,
                emission_color,
                geometry_opacity,
                geometry_thin_walled,
                geometry_normal,
            } = *patch;
            core::Op::Material(Box::new(core::MaterialPatch {
                name,
                base_color: base_color.map(texturable),
                base_diffuse_roughness,
                base_metalness: base_metalness.map(texturable),
                specular_weight,
                specular_roughness: specular_roughness.map(texturable),
                specular_ior,
                transmission_weight,
                transmission_color,
                transmission_depth,
                transmission_scatter,
                transmission_scatter_anisotropy,
                subsurface_weight: subsurface_weight.map(texturable),
                subsurface_color: subsurface_color.map(texturable),
                subsurface_radius: subsurface_radius.map(texturable),
                subsurface_radius_scale: subsurface_radius_scale.map(texturable),
                subsurface_scatter_anisotropy: subsurface_scatter_anisotropy.map(texturable),
                coat_weight,
                coat_color,
                coat_roughness,
                coat_ior,
                coat_darkening,
                fuzz_weight,
                fuzz_color,
                fuzz_roughness,
                emission_luminance,
                emission_color: emission_color.map(texturable),
                geometry_opacity: geometry_opacity.map(texturable),
                geometry_thin_walled,
                geometry_normal: geometry_normal.map(|value| reset(value, texture_ref)),
            }))
        }
        wire::Op::Light(patch) => {
            let wire::LightPatch { name, light } = patch;
            core::Op::Light(core::LightPatch {
                name,
                light: light.map(self::light),
            })
        }
        wire::Op::Camera(patch) => {
            let wire::CameraPatch {
                name,
                position,
                look_at,
                up,
                vfov_degrees,
                focus_distance,
                aperture_radius,
            } = patch;
            core::Op::Camera(core::CameraPatch {
                name,
                position,
                look_at,
                up,
                vfov_degrees,
                focus_distance: focus_distance.map(|value| reset(value, |distance| distance)),
                aperture_radius,
            })
        }
        wire::Op::Environment(patch) => {
            let wire::EnvironmentPatch {
                name,
                path,
                tint,
                transform,
            } = patch;
            core::Op::Environment(core::EnvironmentPatch {
                name,
                path: path.map(|value| reset(value, PathBuf::from)),
                tint,
                transform: transform.map(self::transform),
            })
        }
        wire::Op::Settings(patch) => {
            let wire::SettingsPatch {
                name,
                resolution,
                spp,
                noise_threshold,
                max_bounces,
                denoise,
                seed,
            } = patch;
            core::Op::Settings(core::SettingsPatch {
                name,
                resolution,
                spp,
                noise_threshold: noise_threshold.map(|value| reset(value, convert::identity)),
                max_bounces,
                denoise,
                seed,
                // The wire schema has no medium kind to reference yet, so
                // a client cannot name one: leave whatever the scene file
                // set alone rather than clearing it.
                global_medium: None,
            })
        }
        wire::Op::Remove(kind, name) => core::Op::Remove(self::kind(kind), name),
    }
}

fn kind(kind: wire::Kind) -> core::Kind {
    match kind {
        wire::Kind::Mesh => core::Kind::Mesh,
        wire::Kind::Instance => core::Kind::Instance,
        wire::Kind::Material => core::Kind::Material,
        wire::Kind::Curves => core::Kind::Curves,
        wire::Kind::Light => core::Kind::Light,
        wire::Kind::Camera => core::Kind::Camera,
        wire::Kind::Environment => core::Kind::Environment,
        wire::Kind::Settings => core::Kind::Settings,
    }
}

fn mesh_source(source: wire::MeshSource) -> description::MeshSource {
    match source {
        wire::MeshSource::Inline {
            positions,
            normals,
            uvs,
            triangles,
        } => description::MeshSource::Inline {
            positions,
            normals,
            uvs,
            triangles,
        },
        wire::MeshSource::Ply { path } => description::MeshSource::Ply {
            path: PathBuf::from(path),
        },
    }
}

fn curves_source(source: wire::CurvesSource) -> description::CurvesSource {
    match source {
        wire::CurvesSource::Inline {
            points,
            curve_vertex_counts,
            widths,
            curve_type,
            basis,
            wrap,
        } => description::CurvesSource::Inline {
            points,
            curve_vertex_counts,
            widths: widths.map(|widths| {
                let wire::Widths {
                    values,
                    interpolation,
                } = widths;
                description::Widths {
                    values,
                    interpolation: match interpolation {
                        wire::WidthInterpolation::Constant => {
                            description::WidthInterpolation::Constant
                        }
                        wire::WidthInterpolation::Uniform => {
                            description::WidthInterpolation::Uniform
                        }
                        wire::WidthInterpolation::Varying => {
                            description::WidthInterpolation::Varying
                        }
                        wire::WidthInterpolation::Vertex => description::WidthInterpolation::Vertex,
                    },
                }
            }),
            curve_type: match curve_type {
                wire::CurveType::Linear => description::CurveType::Linear,
                wire::CurveType::Cubic => description::CurveType::Cubic,
            },
            basis: match basis {
                wire::CurveBasis::Bezier => description::CurveBasis::Bezier,
                wire::CurveBasis::BSpline => description::CurveBasis::BSpline,
                wire::CurveBasis::CatmullRom => description::CurveBasis::CatmullRom,
            },
            wrap: match wrap {
                wire::CurveWrap::Nonperiodic => description::CurveWrap::Nonperiodic,
                wire::CurveWrap::Pinned => description::CurveWrap::Pinned,
                wire::CurveWrap::Periodic => description::CurveWrap::Periodic,
            },
        },
        wire::CurvesSource::Hair { path } => description::CurvesSource::Hair {
            path: PathBuf::from(path),
        },
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "every translation consumes its wire value; this one is all-Copy inside"
)]
fn transform(transform: wire::Transform) -> description::Transform {
    match transform {
        wire::Transform::Trs {
            translate,
            rotate_degrees,
            scale,
        } => description::Transform::Trs {
            translate,
            rotate_degrees,
            scale,
        },
        wire::Transform::Matrix(rows) => description::Transform::Matrix(rows),
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "every translation consumes its wire value; this one is all-Copy inside"
)]
fn light(light: wire::Light) -> description::Light {
    match light {
        wire::Light::Distant {
            direction,
            irradiance,
        } => description::Light::Distant {
            direction,
            irradiance,
        },
        wire::Light::Point {
            position,
            intensity,
        } => description::Light::Point {
            position,
            intensity,
        },
    }
}

fn texturable<T>(value: wire::Texturable<T>) -> description::Texturable<T> {
    match value {
        wire::Texturable::Constant(constant) => description::Texturable::Constant(constant),
        wire::Texturable::Texture(reference) => {
            description::Texturable::Texture(texture_ref(reference))
        }
    }
}

fn texture_ref(reference: wire::TextureRef) -> description::TextureRef {
    let wire::TextureRef {
        path,
        color_space,
        channel,
    } = reference;
    description::TextureRef {
        path: PathBuf::from(path),
        color_space: color_space.map(|space| match space {
            wire::ColorSpace::Srgb => description::ColorSpace::Srgb,
            wire::ColorSpace::Linear => description::ColorSpace::Linear,
        }),
        channel: channel.map(|channel| match channel {
            wire::Channel::R => description::Channel::R,
            wire::Channel::G => description::Channel::G,
            wire::Channel::B => description::Channel::B,
            wire::Channel::A => description::Channel::A,
        }),
        // The wire schema predates sample-time texture parameters; USD's
        // `st` inputs don't author them today, so the delegate always
        // sends identity.
        scale: None,
        uv: None,
    }
}

/// Unfold the wire's [`wire::Reset`] into the renderer's inner
/// `Option<T>`: `Clear` is `None` (restore the default), `Set` maps the
/// value through. The caller's `.map` supplies the outer `Some`.
fn reset<W, C>(value: wire::Reset<W>, map: impl FnOnce(W) -> C) -> Option<C> {
    match value {
        wire::Reset::Clear => None,
        wire::Reset::Set(inner) => Some(map(inner)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three states of a doubly-optional field land on the renderer's
    /// three values — the semantic half of what the corpus pins as bytes.
    #[test]
    fn reset_unfolds_to_the_doubly_optional() {
        let with = |normal| {
            let set = change_set(wire::ChangeSet {
                ops: vec![wire::Op::Material(Box::new(wire::MaterialPatch {
                    name: "m".into(),
                    geometry_normal: normal,
                    ..wire::MaterialPatch::default()
                }))],
            });
            let core::Op::Material(patch) = &set.ops[0] else {
                panic!("op kind changed in translation");
            };
            patch.geometry_normal.clone()
        };
        assert_eq!(with(None), None);
        assert_eq!(with(Some(wire::Reset::Clear)), Some(None));
        let reference = wire::TextureRef {
            path: "/n.png".into(),
            color_space: Some(wire::ColorSpace::Linear),
            channel: Some(wire::Channel::A),
        };
        assert_eq!(
            with(Some(wire::Reset::Set(reference))),
            Some(Some(description::TextureRef {
                path: "/n.png".into(),
                color_space: Some(description::ColorSpace::Linear),
                channel: Some(description::Channel::A),
                scale: None,
                uv: None,
            }))
        );
    }

    /// The stopping rule survives translation: a client that asks to
    /// stop early gets to, one that clears the threshold spends its whole
    /// budget, and one that says nothing leaves the scene's own answer
    /// standing — a translation that drops the field is indistinguishable
    /// from that third case, which is why all three are pinned.
    #[test]
    fn the_noise_threshold_reaches_the_renderer() {
        let with = |threshold| {
            let set = change_set(wire::ChangeSet {
                ops: vec![wire::Op::Settings(wire::SettingsPatch {
                    name: "settings".into(),
                    noise_threshold: threshold,
                    ..wire::SettingsPatch::default()
                })],
            });
            let core::Op::Settings(patch) = &set.ops[0] else {
                panic!("op kind changed in translation");
            };
            patch.noise_threshold
        };
        assert_eq!(with(None), None);
        assert_eq!(with(Some(wire::Reset::Clear)), Some(None));
        assert_eq!(with(Some(wire::Reset::Set(0.02))), Some(Some(0.02)));
    }

    /// A translated set survives the renderer's own apply — names,
    /// references, and geometry all meaning what they meant on the wire.
    #[test]
    fn a_translated_genesis_applies() {
        let set = change_set(wire::ChangeSet {
            ops: vec![
                wire::Op::Mesh(wire::MeshPatch {
                    name: "tri".into(),
                    source: Some(wire::MeshSource::Inline {
                        positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        normals: None,
                        uvs: None,
                        triangles: vec![[0, 1, 2]],
                    }),
                }),
                wire::Op::Material(Box::new(wire::MaterialPatch {
                    name: "gray".into(),
                    ..wire::MaterialPatch::default()
                })),
                wire::Op::Instance(wire::InstancePatch {
                    name: "thing".into(),
                    mesh: Some("tri".into()),
                    curves: None,
                    material: Some("gray".into()),
                    transforms: Some(vec![wire::Transform::Trs {
                        translate: [0.0; 3],
                        rotate_degrees: [0.0; 3],
                        scale: [1.0; 3],
                    }]),
                    camera_visible: Some(true),
                }),
                wire::Op::Remove(wire::Kind::Material, "gray".into()),
            ],
        });
        let mut description = cenote::scene::description::SceneDescription::new();
        // The removal strands the instance — the *renderer's* validation
        // speaking about wire-borne names proves the meaning crossed.
        let error = description.apply(&set).expect_err("stranded reference");
        assert!(error.to_string().contains("\"gray\""), "{error}");
        let mut set = set;
        set.ops.pop();
        description.apply(&set).expect("the genesis applies");
        assert_eq!(description.instances().len(), 1);
    }

    /// The camera lane lowers like scene prep does: zero aperture is a
    /// pinhole, a positive one focuses at `look_at` unless told otherwise.
    #[test]
    fn the_camera_lane_lowers_the_lens() {
        let base = protocol::Camera {
            position: [0.0, 0.0, 4.0],
            look_at: [0.0; 3],
            up: [0.0, 1.0, 0.0],
            vfov_degrees: 40.0,
            focus_distance: None,
            aperture_radius: 0.0,
        };
        assert!(camera(base).lens.is_none(), "zero aperture is a pinhole");
        let lens = camera(protocol::Camera {
            aperture_radius: 0.1,
            ..base
        })
        .lens
        .expect("a positive aperture makes a lens");
        assert!((lens.focus_distance - 4.0).abs() < 1e-6, "focus at look_at");
        let told = camera(protocol::Camera {
            aperture_radius: 0.1,
            focus_distance: Some(2.5),
            ..base
        })
        .lens
        .expect("lens");
        assert!((told.focus_distance - 2.5).abs() < 1e-6);
    }
}
