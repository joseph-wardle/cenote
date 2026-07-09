//! Turntable orbit camera: spherical coordinates around a fixed target,
//! converted to a core [`Camera`] each frame. Screen-space drag deltas map
//! to yaw/pitch; scroll dollies along the view axis.

use cenote::scene::Camera;
use glam::Vec3;

/// Radians of orbit per pixel of drag — a full-width drag across the
/// default 1280 px window is about one full turn.
const RADIANS_PER_PIXEL: f32 = 0.005;

/// Pitch limit, just shy of the poles: the core camera derives its frame
/// from world-up and panics on a vertical view axis.
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
    /// Radians around world +Y; 0 puts the camera on the target's +Z side.
    yaw: f32,
    /// Radians above the horizon, clamped to ±[`MAX_PITCH`].
    pitch: f32,
    vfov_degrees: f32,
}

impl OrbitCamera {
    /// Start where `camera` stands: orbit parameters recovered from its pose
    /// so the first frame matches the scene's authored view.
    pub fn framing(camera: &Camera) -> Self {
        let offset = camera.position - camera.look_at;
        let distance = offset.length();
        Self {
            target: camera.look_at,
            distance,
            yaw: offset.x.atan2(offset.z),
            pitch: (offset.y / distance).asin().clamp(-MAX_PITCH, MAX_PITCH),
            vfov_degrees: camera.vfov_degrees,
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

    /// The pinhole camera at the current orbit position.
    pub fn camera(&self) -> Camera {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let toward_camera = Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
        Camera {
            position: self.target + toward_camera * self.distance,
            look_at: self.target,
            vfov_degrees: self.vfov_degrees,
        }
    }
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
            vfov_degrees: 40.0,
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
        assert!((back.vfov_degrees - source.vfov_degrees).abs() < f32::EPSILON);
    }

    /// However hard the user drags, the view axis must stay off the world
    /// vertical — the core camera panics on a degenerate frame, so `basis`
    /// succeeding is the assertion.
    #[test]
    fn pitch_never_reaches_the_poles() {
        let mut orbit = OrbitCamera::framing(&demo_pose());
        orbit.orbit(0.0, -1e6);
        let _ = orbit.camera().basis(1.0);
        orbit.orbit(0.0, 1e6);
        let _ = orbit.camera().basis(1.0);
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
