//! Turntable orbit camera: spherical coordinates around a fixed target,
//! converted to a core [`Camera`] each frame. Screen-space drag deltas map
//! to yaw/pitch; scroll dollies along the view axis.
//!
//! The turntable spins about the *scene's* up axis, not a hard-coded one.
//! Scenes do not all agree which way is up — sanmiguel is Z-up, the rest of
//! the corpus Y-up — and spinning a Z-up scene about world +Y lays its
//! horizon on its side.

use cenote::scene::{Camera, Lens};
use glam::Vec3;

/// Radians of orbit per pixel of drag — a full-width drag across the
/// default 1280 px window is about one full turn.
const RADIANS_PER_PIXEL: f32 = 0.005;

/// Pitch limit, just shy of the poles: the core camera derives its frame
/// from the up axis and panics on a view axis parallel to it.
const MAX_PITCH: f32 = 88.5 * (std::f32::consts::PI / 180.0);

/// Multiplicative distance change per scroll notch (scrolling up zooms in).
const DOLLY_STEP: f32 = 0.9;

/// Distance clamps: never inside the subject, never lost in the sky.
const DISTANCE_RANGE: (f32, f32) = (0.2, 100.0);

/// The orbit state: a camera on a sphere around `target`, always looking at
/// its center.
pub struct OrbitCamera {
    target: Vec3,
    distance: f32,
    /// The scene's up axis: the turntable's spin axis, screen-up in every
    /// frame it hands out, and the pole pitch is kept away from. Always
    /// cardinal — see [`OrbitCamera::framing`].
    up: Vec3,
    /// Radians around [`OrbitCamera::up`]; 0 puts the camera on the
    /// `ahead` side of the target (see [`frame`]), and increasing yaw
    /// swings it counter-clockwise seen from above.
    yaw: f32,
    /// Radians above the horizon, clamped to ±[`MAX_PITCH`].
    pitch: f32,
    vfov_degrees: f32,
    /// The scene's authored lens, carried through every orbit move so a
    /// depth-of-field scene keeps its look while navigating. Orbiting
    /// around the target holds the subject distance, so the authored
    /// focus stays meaningful; only a dolly walks away from it.
    lens: Option<Lens>,
}

impl OrbitCamera {
    /// Start where `camera` stands: orbit parameters recovered from its pose
    /// so the first frame matches the scene's authored view. Any roll the
    /// source camera carried is dropped — a turntable is level by
    /// construction, so the first drag would cancel it anyway.
    ///
    /// Level *about what* is the scene's business, and the authored up is
    /// the only thing that knows: nothing else in a scene names a world
    /// axis. It is snapped to the nearest cardinal direction, which is what
    /// separates the up axis from the roll — an exporter derives the up it
    /// writes from the world axis (pbrt's `LookAt` orthogonalizes it
    /// against the view direction), so snapping recovers the axis exactly
    /// and drops only the tilt that orthogonalization introduced. A camera
    /// deliberately rolled off the world axis is snapped level, as before.
    ///
    /// The one pose this cannot read is a near-vertical one: a camera
    /// looking straight down carries a horizontal up that says nothing
    /// about which way the world stands, and the turntable inherits the
    /// mistake. A scene-level up axis would settle it; the format has none.
    pub fn framing(camera: &Camera) -> Self {
        let offset = camera.position - camera.look_at;
        let distance = offset.length();
        let up = cardinal(camera.up);
        let (ahead, right) = frame(up);
        Self {
            target: camera.look_at,
            distance,
            up,
            yaw: offset.dot(right).atan2(offset.dot(ahead)),
            pitch: (offset.dot(up) / distance).asin().clamp(-MAX_PITCH, MAX_PITCH),
            vfov_degrees: camera.vfov_degrees,
            lens: camera.lens,
        }
    }

    /// Drag by a screen-space delta (pixels, +y down): the camera orbits in
    /// the drag direction — rightward and upward around the target.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * RADIANS_PER_PIXEL;
        self.pitch = (self.pitch - dy * RADIANS_PER_PIXEL).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Scroll by `notches` (positive zooms in): multiplicative, so each
    /// notch feels equal at every scale, clamped to [`DISTANCE_RANGE`].
    pub fn dolly(&mut self, notches: f32) {
        self.distance =
            (self.distance * DOLLY_STEP.powf(notches)).clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);
    }

    /// The camera at the current orbit position, wearing the scene's lens.
    pub fn camera(&self) -> Camera {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (ahead, right) = frame(self.up);
        let toward_camera = cos_pitch * (cos_yaw * ahead + sin_yaw * right) + sin_pitch * self.up;
        Camera {
            position: self.target + toward_camera * self.distance,
            look_at: self.target,
            up: self.up,
            vfov_degrees: self.vfov_degrees,
            lens: self.lens,
        }
    }
}

/// The cardinal direction `up` points most nearly along — the scene's up
/// axis, read out of the one vector that carries it.
fn cardinal(up: Vec3) -> Vec3 {
    let along = up.abs();
    if along.x >= along.y && along.x >= along.z {
        Vec3::X * up.x.signum()
    } else if along.y >= along.z {
        Vec3::Y * up.y.signum()
    } else {
        Vec3::Z * up.z.signum()
    }
}

/// The two axes yaw sweeps between: `ahead` is where yaw 0 puts the camera,
/// `right` a quarter turn counter-clockwise from it about `up`. Which
/// perpendicular `ahead` lands on only offsets the yaw origin, which
/// nothing outside this file reads; deriving `right` from `up` is what
/// keeps a rightward drag turning the same way whatever is up.
fn frame(up: Vec3) -> (Vec3, Vec3) {
    // Y-up is the common case and wants ahead = +Z, the convention the
    // orbit carried when +Y was the only up it knew.
    let seed = if up.z.abs() < 0.5 { Vec3::Z } else { Vec3::Y };
    let ahead = (seed - up * seed.dot(up)).normalize();
    (ahead, up.cross(ahead))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_pose() -> Camera {
        // Any representative off-axis pose does — these tests exercise the
        // orbit math, not the demo scene's authored camera.
        Camera {
            position: Vec3::new(0.0, 1.8, 5.0),
            look_at: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            vfov_degrees: 40.0,
            lens: None,
        }
    }

    /// `framing` inverts `camera`: adopting a pose and converting back must
    /// land on the same camera, or the first viewer frame won't match the
    /// scene's authored view.
    #[test]
    fn framing_round_trips_the_source_camera() {
        let source = demo_pose();
        let orbit = OrbitCamera::framing(&source);
        let back = orbit.camera();
        assert!(back.position.distance(source.position) < 1e-5);
        assert!(back.look_at.distance(source.look_at) < 1e-5);
        assert_eq!(back.up, Vec3::Y);
        assert!((back.vfov_degrees - source.vfov_degrees).abs() < f32::EPSILON);
    }

    /// sanmiguel's authored pose, the corpus's one Z-up scene, and the one
    /// that rendered on its side. Matching both endpoints is not enough to
    /// catch that: a turntable stuck on +Y reproduces the position and the
    /// target exactly and still rolls the image a quarter turn, so the
    /// assertion is on the *rendering* frame.
    #[test]
    fn framing_round_trips_a_z_up_camera() {
        let source = Camera {
            position: Vec3::new(22.867_598, -12.928_896, 1.947_840_7),
            look_at: Vec3::new(22.211_624, -12.180_727, 2.047_546_6),
            up: Vec3::new(0.065_730_36, -0.074_971_974, 0.995_017),
            vfov_degrees: 83.9744,
            lens: None,
        };
        let back = OrbitCamera::framing(&source).camera();
        assert!(back.position.distance(source.position) < 1e-5);
        assert!(back.look_at.distance(source.look_at) < 1e-5);
        assert_eq!(back.up, Vec3::Z);
        let (authored, adopted) = (source.basis(1.6), back.basis(1.6));
        assert!(authored.up.distance(adopted.up) < 1e-5);
        assert!(authored.right.distance(adopted.right) < 1e-5);
        assert!(authored.forward.distance(adopted.forward) < 1e-5);
    }

    /// The axis outlives the first drag: a Z-up scene must still be Z-up
    /// after orbiting, or the horizon rolls as soon as the view moves.
    #[test]
    fn orbiting_keeps_the_scene_axis_up() {
        let mut orbit = OrbitCamera::framing(&Camera {
            up: Vec3::Z,
            ..demo_pose()
        });
        orbit.orbit(300.0, -120.0);
        let camera = orbit.camera();
        assert_eq!(camera.up, Vec3::Z);
        assert!((camera.position.distance(camera.look_at) - orbit.distance).abs() < 1e-4);
    }

    /// A rightward drag turns the same way whichever axis is up — the
    /// yaw origin may differ, the handedness may not.
    #[test]
    fn a_rightward_drag_turns_the_same_way_about_any_axis() {
        for up in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::Y] {
            let mut orbit = OrbitCamera::framing(&Camera { up, ..demo_pose() });
            // Yaw only moves the part of the offset that lies across `up`,
            // so that is the part whose turn direction means anything.
            let target = orbit.target;
            let across = |camera: Camera| {
                let offset = camera.position - target;
                offset - up * offset.dot(up)
            };
            let before = across(orbit.camera());
            orbit.orbit(RADIANS_PER_PIXEL.recip() * 0.25, 0.0);
            let after = across(orbit.camera());
            // A quarter radian counter-clockwise about `up`: the cross
            // product of before and after points along it, right-hand rule.
            assert!(before.cross(after).normalize().dot(up) > 0.99);
        }
    }

    /// Up axes are read off the authored vector by nearest cardinal, and a
    /// deliberate roll is snapped level rather than tilting the turntable.
    #[test]
    fn the_up_axis_is_the_nearest_cardinal() {
        assert_eq!(cardinal(Vec3::new(0.0, 0.87, 0.0)), Vec3::Y);
        assert_eq!(cardinal(Vec3::new(0.066, -0.075, 0.995)), Vec3::Z);
        assert_eq!(cardinal(Vec3::new(-0.488, 0.873, 0.003)), Vec3::Y);
        assert_eq!(cardinal(Vec3::new(0.0, 0.0, -1.0)), -Vec3::Z);
        assert_eq!(cardinal(Vec3::new(-0.9, 0.1, 0.2)), -Vec3::X);
    }

    /// However hard the user drags, the view axis must stay off the world
    /// vertical — the core camera panics on a degenerate frame, so `basis`
    /// succeeding is the assertion.
    #[test]
    fn pitch_never_reaches_the_poles() {
        let mut orbit = OrbitCamera::framing(&demo_pose());
        for pitch in [-1e6, 1e6] {
            orbit.orbit(0.0, pitch);
            let camera = orbit.camera();
            let forward = (camera.look_at - camera.position).normalize();
            // Short of the pole by a workable margin (the clamp leaves
            // ~1.5°): at the pole itself the view axis is parallel to up
            // and the basis degenerates.
            assert!(
                forward.dot(camera.up).abs() < 0.9999,
                "pitch {pitch} reached the pole: forward {forward:?}"
            );
            let basis = camera.basis(1.0);
            assert!(basis.forward.is_finite() && basis.up.is_finite() && basis.right.is_finite());
        }
    }

    /// Zoom saturates at the clamps instead of hitting zero (a degenerate
    /// camera) or infinity from one aggressive scroll.
    #[test]
    fn dolly_saturates_at_the_distance_clamps() {
        let mut orbit = OrbitCamera::framing(&demo_pose());
        orbit.dolly(1e4);
        assert!((orbit.distance - DISTANCE_RANGE.0).abs() < f32::EPSILON);
        orbit.dolly(-1e4);
        assert!((orbit.distance - DISTANCE_RANGE.1).abs() < f32::EPSILON);
    }
}
