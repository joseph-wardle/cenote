//! Curve geometry: `BasisCurves` cells in, tube triangles out.
//!
//! The renderer owns every piece of curve mathematics, and it owns it
//! *once*. A description carries the cells USD authored — points, vertex
//! counts, widths, and the type/basis/wrap tokens that say how to read
//! them — and this module turns them into the one canonical form the rest
//! of the renderer knows about:
//!
//! **[`Strands`]: polylines with a radius at every point.** Everything
//! downstream is a backend over that buffer. Today the only backend is
//! [`tessellate`], which sweeps a three-sided tube along each strand and
//! hands the result to the ordinary triangle path — same BLAS, same
//! `GeometryRecord`, same `OpenPBR` shading, and no kernel knows curves
//! exist. Ribbons, AABB-bounded capsule intersection, and NVIDIA's linear
//! swept spheres would each be a second consumer of the same [`Strands`],
//! which is why the canonical form comes first and the backend second.
//!
//! Two shapes of approximation live here, both deliberate and both
//! measured against the strand's own width rather than against the screen:
//!
//! - **Longitudinal.** Linear curves pass through verbatim; cubic spans
//!   flatten by recursive bisection until the chord's deviation falls
//!   under [`FLATNESS`] × the local radius, capped at [`MAX_SPAN_SEGMENTS`]
//!   pieces. An error smaller than the strand's own footprint cannot be
//!   seen at any distance, so the rule needs no camera — which is what
//!   keeps a tessellation bit-stable across frames, edits, and machines.
//! - **Radial.** Three sides, not four or eight: the fewest that never
//!   let a strand vanish (a triangle seen from its worst angle still
//!   spans 87% of its width, where a ribbon would disappear outright),
//!   with the *analytic* radial normal stored at every ring vertex so the
//!   shading is a cylinder's even though the geometry is a prism. It is
//!   the icosphere trick from [`super::shapes`], one dimension down.
//!
//! The strand coordinates the tessellation writes are `u` = root-to-tip
//! arc length and `v` = one random value per strand: the two handles a
//! hair lookdev actually reaches for (a melanin ramp along the fiber, a
//! per-strand variation across the groom) without a second closure or a
//! per-strand ID buffer. The cost is that the parameterization is
//! degenerate in `v` — every vertex of a strand shares one value — so the
//! UV-derived tangent frame in `surface.slang` finds no `v` axis and
//! tangent-space normal maps fall back to the interpolated normal. Hair
//! does not wear normal maps; per-strand variation it cannot do without.

use std::f32::consts::TAU;

use glam::{Vec2, Vec3};

use super::description::{
    CurveBasis, CurveType, CurveWrap, Curves, CurvesSource, WidthInterpolation, Widths,
};
use super::{Mesh, scene_error};
use crate::error::{Error, Result};

/// How far a flattened chord may stray from the curve it replaces, as a
/// fraction of the local strand radius. Half a radius is a deviation the
/// strand's own body covers, at any distance and from any angle.
const FLATNESS: f32 = 0.5;

/// The most pieces one cubic span may flatten into — three levels of
/// bisection. A cap is what keeps a pathological control polygon (a
/// hairpin authored across four points) from turning one span into
/// thousands of triangles; 16 is the escape hatch if a groom is ever
/// measured to need it.
const MAX_SPAN_SEGMENTS: u32 = 8;

/// Sides per tube — see the module docs. Three vertices per ring, six
/// triangles per segment.
const SIDES: usize = 3;

/// The canonical curve representation: strands as polylines, each point
/// carrying the radius of the tube there.
///
/// Flat arrays with a start offset per strand, rather than a vector of
/// vectors — a hero groom is fifty thousand strands and three and a half
/// million points, and one allocation per strand is a cost paid for
/// nothing. `offsets` holds `strands + 1` entries, so strand `i` spans
/// `offsets[i]..offsets[i + 1]` in both `points` and `radii`.
#[derive(Debug)]
pub(crate) struct Strands {
    points: Vec<Vec3>,
    radii: Vec<f32>,
    offsets: Vec<u32>,
}

impl Strands {
    /// An empty set, ready for building.
    pub(crate) fn new() -> Self {
        Self {
            points: Vec::new(),
            radii: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Reserve room for `points` points across `strands` strands — the
    /// exact size is known up front for linear input, and closely
    /// predicted for cubic.
    pub(crate) fn reserve(&mut self, strands: usize, points: usize) {
        self.points.reserve(points);
        self.radii.reserve(points);
        self.offsets.reserve(strands);
    }

    /// Append one point to the strand under construction. Radii clamp at
    /// zero: an interpolating basis can undershoot a width stream the way
    /// it undershoots a position, and a negative radius would turn the
    /// ring inside out.
    pub(crate) fn push_point(&mut self, point: Vec3, radius: f32) {
        self.points.push(point);
        self.radii.push(radius.max(0.0));
    }

    /// Close the strand under construction. A strand of fewer than two
    /// points has no segment to sweep, so it is dropped rather than
    /// carried as an unrenderable stub.
    pub(crate) fn end_strand(&mut self) {
        let start = *self.offsets.last().expect("offsets starts at one entry") as usize;
        if self.points.len() - start < 2 {
            self.points.truncate(start);
            self.radii.truncate(start);
            return;
        }
        self.offsets.push(self.points.len() as u32);
    }

    /// How many strands survived.
    pub(crate) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// How many points they hold in total.
    pub(crate) fn points(&self) -> usize {
        self.points.len()
    }

    /// One point's position, counted across the whole set — the shape a
    /// test's analytic oracle compares against.
    #[cfg(test)]
    pub(crate) fn point(&self, index: usize) -> Vec3 {
        self.points[index]
    }

    /// One point's radius, counted the same way.
    #[cfg(test)]
    pub(crate) fn radius(&self, index: usize) -> f32 {
        self.radii[index]
    }

    /// One strand's points and radii.
    fn strand(&self, index: usize) -> (&[Vec3], &[f32]) {
        let range = self.offsets[index] as usize..self.offsets[index + 1] as usize;
        (&self.points[range.clone()], &self.radii[range])
    }
}

/// Resolve a description's curves onto the host as triangles — the one
/// entry point prep calls, mirroring `resolve_mesh` next door.
///
/// # Errors
///
/// [`Error::Scene`](crate::Error) naming the curves object: a `.hair` file
/// that will not read, or — defensively, since apply validated the cells —
/// a topology that cannot be walked.
pub(crate) fn resolve(name: &str, curves: &Curves) -> Result<Mesh> {
    let named = |error| match error {
        Error::Scene(message) => scene_error(format!("curves \"{name}\": {message}")),
        other => other,
    };
    let strands = match &curves.source {
        CurvesSource::Inline {
            points,
            curve_vertex_counts,
            widths,
            curve_type,
            basis,
            wrap,
        } => evaluate(
            points,
            curve_vertex_counts,
            widths.as_ref(),
            *curve_type,
            *basis,
            *wrap,
        )
        .map_err(named)?,
        CurvesSource::Hair { path } => crate::hair::read(path).map_err(named)?,
    };
    let mesh = tessellate(&strands);
    log::debug!(
        "curves \"{name}\": {} strands, {} points, {} triangles",
        strands.len(),
        strands.points(),
        mesh.triangles.len()
    );
    Ok(mesh)
}

/// Walk the `BasisCurves` cells into strands: linear curves verbatim,
/// cubic ones evaluated through their basis and flattened.
///
/// # Errors
///
/// [`Error::Scene`](crate::Error) for a periodic wrap (refused outright,
/// never silently opened) or a vertex count no segment rule accepts.
pub(crate) fn evaluate(
    points: &[[f32; 3]],
    counts: &[u32],
    widths: Option<&Widths>,
    curve_type: CurveType,
    basis: CurveBasis,
    wrap: CurveWrap,
) -> Result<Strands> {
    if wrap == CurveWrap::Periodic {
        return Err(scene_error(
            "periodic curves are not supported — the renderer sweeps strands from a root, \
             and a closed loop has none"
                .to_owned(),
        ));
    }
    let mut strands = Strands::new();
    strands.reserve(counts.len(), points.len());
    let mut at = Cursor::default();
    for (curve, &count) in counts.iter().enumerate() {
        at.curve = curve;
        at.count = count as usize;
        at.segments = segment_count(at.count, curve_type, basis, wrap)?;
        if at.vertex + at.count > points.len() {
            return Err(scene_error(format!(
                "curve {curve} runs past the end of the points array"
            )));
        }
        let control = &points[at.vertex..at.vertex + at.count];
        let mut resolved = CurveWidths::select(widths, &at, curve_type, basis);
        if pinned(wrap, basis) {
            resolved.pin();
        }
        match curve_type {
            CurveType::Linear => push_linear(&mut strands, control, &resolved),
            CurveType::Cubic => {
                push_cubic(&mut strands, control, &resolved, basis, wrap, at.segments);
            }
        }
        strands.end_strand();
        at.vertex += at.count;
        at.varying += at.segments + 1;
    }
    Ok(strands)
}

/// A linear curve is already a polyline: its vertices *are* the strand,
/// and only the radii have to be resolved onto them.
fn push_linear(strands: &mut Strands, control: &[[f32; 3]], widths: &CurveWidths) {
    // Never read — a linear curve's widths are constant or interpolate
    // linearly across a segment, and neither asks for a basis.
    let matrix = CurveBasis::Bezier.matrix();
    for (index, point) in control.iter().enumerate() {
        let segment = index.min(control.len() - 2);
        let t = (index - segment) as f32;
        let radius = widths.segment(segment).radius(&matrix, t);
        strands.push_point(Vec3::from(*point), radius);
    }
}

/// A cubic curve is a chain of four-point spans, each evaluated through
/// the basis and flattened until its chords lie inside the strand.
///
/// Pinned curves unpack first: USD defines them as an ordinary
/// nonperiodic curve with a phantom point mirrored off each end, which is
/// exactly what makes the approximating bases pass through the authored
/// endpoints. Unpacking here — rather than special-casing every span —
/// keeps the evaluation below to one code path.
fn push_cubic(
    strands: &mut Strands,
    control: &[[f32; 3]],
    widths: &CurveWidths,
    basis: CurveBasis,
    wrap: CurveWrap,
    segments: usize,
) {
    let matrix = basis.matrix();
    let step = basis.vstep();
    let points: Vec<Vec3> = if pinned(wrap, basis) {
        let cells: Vec<Vec3> = control.iter().copied().map(Vec3::from).collect();
        let head = 2.0 * cells[0] - cells[1];
        let tail = 2.0 * cells[cells.len() - 1] - cells[cells.len() - 2];
        std::iter::once(head)
            .chain(cells)
            .chain(std::iter::once(tail))
            .collect()
    } else {
        control.iter().copied().map(Vec3::from).collect()
    };

    for segment in 0..segments {
        let base = segment * step;
        let span = [
            points[base],
            points[base + 1],
            points[base + 2],
            points[base + 3],
        ];
        let radii = widths.segment(segment);
        // The root of the whole strand is emitted once; every span after
        // it starts where the last one ended.
        if segment == 0 {
            strands.push_point(evaluate_span(&matrix, &span, 0.0), radii.radius(&matrix, 0.0));
        }
        flatten(strands, &matrix, &span, &radii, 0.0, 1.0, 0);
    }
}

/// Bisect `[t0, t1]` until its chord lies within [`FLATNESS`] of the span,
/// then emit the interval's far end. Deviation is measured at the quarter,
/// half, and three-quarter points rather than at the midpoint alone: an
/// S-shaped span passes straight through the middle of its own chord, and
/// a midpoint-only test would call it flat.
fn flatten(
    strands: &mut Strands,
    matrix: &[[f32; 4]; 4],
    span: &[Vec3; 4],
    widths: &SegmentWidths,
    t0: f32,
    t1: f32,
    depth: u32,
) {
    if depth < MAX_SPAN_SEGMENTS.ilog2() {
        let a = evaluate_span(matrix, span, t0);
        let b = evaluate_span(matrix, span, t1);
        // The tolerance rides the fatter end: that is where a given
        // deviation is least hidden by the strand's own body, and it keeps
        // a tapered tip from demanding subdivision as its radius goes to
        // zero.
        let radius = widths.radius(matrix, t0).max(widths.radius(matrix, t1));
        let deviation = [0.25_f32, 0.5, 0.75]
            .into_iter()
            .map(|s| evaluate_span(matrix, span, t0 + s * (t1 - t0)).distance(a.lerp(b, s)))
            .fold(0.0_f32, f32::max);
        if deviation > FLATNESS * radius {
            let mid = 0.5 * (t0 + t1);
            flatten(strands, matrix, span, widths, t0, mid, depth + 1);
            flatten(strands, matrix, span, widths, mid, t1, depth + 1);
            return;
        }
    }
    strands.push_point(evaluate_span(matrix, span, t1), widths.radius(matrix, t1));
}

/// One cubic span at parameter `t`: the basis weights are the row vector
/// `[t³ t² t 1]` through the basis matrix, exactly as `UsdGeomBasisCurves`
/// (and `RenderMan`'s `Basis` before it) defines them.
fn evaluate_span(matrix: &[[f32; 4]; 4], span: &[Vec3; 4], t: f32) -> Vec3 {
    let weights = basis_weights(matrix, t);
    weights[0] * span[0] + weights[1] * span[1] + weights[2] * span[2] + weights[3] * span[3]
}

/// The four basis weights at `t` — shared by the positions and by the
/// `vertex`-interpolated width stream, which USD says rides the same
/// basis the centerline does.
fn basis_weights(matrix: &[[f32; 4]; 4], t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    std::array::from_fn(|column| {
        t3 * matrix[0][column] + t2 * matrix[1][column] + t * matrix[2][column] + matrix[3][column]
    })
}

/// Whether a wrap actually pins: `pinned` is defined only for the
/// approximating bases, and a pinned bezier is a nonperiodic bezier —
/// what USD says, and what every DCC writes.
fn pinned(wrap: CurveWrap, basis: CurveBasis) -> bool {
    wrap == CurveWrap::Pinned && basis != CurveBasis::Bezier
}

/// A curve's segment count, per `UsdGeomBasisCurves`' own table. Apply
/// validated this already; the error path is what keeps a future caller
/// (a fixture, a fuzzer) from indexing off the end of `points`.
pub(crate) fn segment_count(
    count: usize,
    curve_type: CurveType,
    basis: CurveBasis,
    wrap: CurveWrap,
) -> Result<usize> {
    let step = basis.vstep();
    match curve_type {
        CurveType::Linear if count >= 2 => Ok(count - 1),
        CurveType::Cubic if pinned(wrap, basis) && count >= 2 => Ok(count - 1),
        CurveType::Cubic if count >= 4 && (count - 4).is_multiple_of(step) => {
            Ok((count - 4) / step + 1)
        }
        _ => Err(scene_error(format!(
            "a {curve_type} {basis} {wrap} curve cannot have {count} vertices"
        ))),
    }
}

/// Where the walk stands in the prim-wide arrays: which curve, and how
/// many vertices and segment ends the ones before it consumed. The three
/// cursors advance separately because the arrays count different things.
#[derive(Default)]
struct Cursor {
    curve: usize,
    vertex: usize,
    varying: usize,
    count: usize,
    segments: usize,
}

/// One curve's slice of the prim-wide width stream, in the shape its
/// segments read.
enum CurveWidths<'a> {
    /// `constant` (whole prim) and `uniform` (per curve) both land here:
    /// one radius, everywhere on this strand.
    Everywhere(f32),
    /// `varying`: one width per segment end, linear across each segment.
    Varying(&'a [f32]),
    /// `vertex` on a linear curve: one width per point, and a linear
    /// curve's points *are* its segment ends.
    Vertex(&'a [f32]),
    /// `vertex` on a cubic curve: one width per control vertex,
    /// interpolated through the basis, so a strand's width bulges the way
    /// its centerline does. Owned because a pinned curve's stream carries
    /// the same mirrored phantoms its points do.
    Cubic { values: Vec<f32>, step: usize },
}

impl<'a> CurveWidths<'a> {
    /// Slice one curve's widths out of the prim-wide array. Lengths are
    /// validated at apply, so the slicing here cannot miss.
    fn select(
        widths: Option<&'a Widths>,
        at: &Cursor,
        curve_type: CurveType,
        basis: CurveBasis,
    ) -> Self {
        // Unauthored widths are 1 — `UsdGeomCurves`' own fallback, and
        // unmistakable rather than invisible when a scene forgets them.
        let Some(widths) = widths else {
            return Self::Everywhere(0.5);
        };
        let values = &widths.values;
        match widths.interpolation {
            WidthInterpolation::Constant => Self::Everywhere(0.5 * values[0]),
            WidthInterpolation::Uniform => Self::Everywhere(0.5 * values[at.curve]),
            WidthInterpolation::Varying => Self::Varying(&values[at.varying..=at.varying + at.segments]),
            WidthInterpolation::Vertex if curve_type == CurveType::Linear => {
                Self::Vertex(&values[at.vertex..at.vertex + at.count])
            }
            WidthInterpolation::Vertex => Self::Cubic {
                values: values[at.vertex..at.vertex + at.count].to_vec(),
                step: basis.vstep(),
            },
        }
    }

    /// Extend the `vertex` width stream with the phantoms a pinned curve's
    /// points grow, so the two streams stay index for index.
    fn pin(&mut self) {
        if let Self::Cubic { values, .. } = self {
            let head = 2.0 * values[0] - values[1];
            let tail = 2.0 * values[values.len() - 1] - values[values.len() - 2];
            values.insert(0, head);
            values.push(tail);
        }
    }

    /// What segment `index` reads.
    fn segment(&self, index: usize) -> SegmentWidths {
        match self {
            Self::Everywhere(radius) => SegmentWidths::Everywhere(*radius),
            Self::Varying(values) | Self::Vertex(values) => {
                SegmentWidths::Ends(0.5 * values[index], 0.5 * values[index + 1])
            }
            Self::Cubic { values, step } => {
                let base = index * step;
                SegmentWidths::Cubic(std::array::from_fn(|offset| 0.5 * values[base + offset]))
            }
        }
    }
}

/// The width rule of one segment, in the form its parameter evaluates.
enum SegmentWidths {
    /// One radius across the whole segment.
    Everywhere(f32),
    /// Radii at the two ends, linear between them.
    Ends(f32, f32),
    /// Four control radii through the curve's basis.
    Cubic([f32; 4]),
}

impl SegmentWidths {
    /// The radius at parameter `t`.
    fn radius(&self, matrix: &[[f32; 4]; 4], t: f32) -> f32 {
        match self {
            Self::Everywhere(radius) => *radius,
            Self::Ends(near, far) => near + t * (far - near),
            Self::Cubic(control) => {
                let weights = basis_weights(matrix, t);
                (0..4).map(|index| weights[index] * control[index]).sum()
            }
        }
    }
}

/// Sweep a three-sided tube along every strand.
///
/// The frame that carries the ring along the strand is rotation-minimizing
/// (Wang et al.'s double reflection): each ring is the previous one
/// transported onto the new tangent with no twist added, so a straight
/// strand has none and a curled one picks up only what its own geometry
/// forces. The alternative — rebuilding an arbitrary frame per point —
/// twists visibly under any texture, and the twist moves when the curve
/// does.
///
/// Two details the images depend on:
///
/// - **Analytic normals.** Every ring vertex stores the radial direction
///   at its own angle, so barycentric interpolation across a face
///   reconstructs the cylinder's normal rather than the prism's facet.
/// - **A random phase per strand.** Rings that all start at the same angle
///   make a groom moiré where neighbouring strands align; the phase comes
///   off a hash of the strand index, so it is stable across every rebuild.
pub(crate) fn tessellate(strands: &Strands) -> Mesh {
    let rings = strands.points();
    let segments = rings.saturating_sub(strands.len());
    let mut mesh = Mesh {
        positions: Vec::with_capacity(rings * SIDES),
        normals: Vec::with_capacity(rings * SIDES),
        uvs: Vec::with_capacity(rings * SIDES),
        triangles: Vec::with_capacity(segments * SIDES * 2),
    };
    for strand in 0..strands.len() {
        let (points, radii) = strands.strand(strand);
        let noise = hash(strand as u32);
        let v = (noise >> 16) as f32 / 65536.0;
        let phase = (noise & 0xffff) as f32 / 65536.0 * (TAU / SIDES as f32);
        let base = mesh.positions.len() as u32;
        let mut tangent = tangent_at(points, 0);
        let mut across = orthogonal(tangent);
        let mut travelled = 0.0;
        let length: f32 = points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum();
        for index in 0..points.len() {
            if index > 0 {
                let next = tangent_at(points, index);
                across = transport(across, points[index - 1], tangent, points[index], next);
                tangent = next;
                travelled += points[index].distance(points[index - 1]);
            }
            let up = tangent.cross(across);
            let u = if length > 0.0 { travelled / length } else { 0.0 };
            for side in 0..SIDES {
                let angle = phase + TAU * side as f32 / SIDES as f32;
                let radial = angle.cos() * across + angle.sin() * up;
                mesh.positions.push(points[index] + radii[index] * radial);
                mesh.normals.push(radial);
                mesh.uvs.push(Vec2::new(u, v));
            }
        }
        for segment in 0..points.len() as u32 - 1 {
            let near = base + segment * SIDES as u32;
            let far = near + SIDES as u32;
            for side in 0..SIDES as u32 {
                let next = (side + 1) % SIDES as u32;
                // Wound so the face normal agrees with the radial normal
                // stored above: counter-clockwise seen from outside.
                mesh.triangles.push([near + side, near + next, far + side]);
                mesh.triangles.push([near + next, far + next, far + side]);
            }
        }
    }
    mesh
}

/// The unit tangent at a point: the central difference where there are
/// neighbours on both sides, the one-sided difference at the ends. A
/// coincident pair (a groom with a duplicated point) falls back to the
/// nearest distinct neighbour, and a strand with no extent at all to +Y —
/// its tube is a degenerate sliver either way, but the frame stays finite.
fn tangent_at(points: &[Vec3], index: usize) -> Vec3 {
    let previous = index.saturating_sub(1);
    let next = (index + 1).min(points.len() - 1);
    if let Some(tangent) = (points[next] - points[previous]).try_normalize() {
        return tangent;
    }
    for other in 0..points.len() {
        if let Some(tangent) = (points[other] - points[index]).try_normalize() {
            return if other > index { tangent } else { -tangent };
        }
    }
    Vec3::Y
}

/// Any unit vector perpendicular to `axis` — the first ring's reference
/// direction, before the per-strand phase turns it.
fn orthogonal(axis: Vec3) -> Vec3 {
    axis.any_orthonormal_vector()
}

/// Carry `across` from one point's frame to the next with no twist added:
/// reflect it through the plane of the step, then through the plane that
/// takes the reflected tangent onto the new one. Two reflections compose
/// into the rotation that moves the frame the least — the standard
/// double-reflection rotation-minimizing frame, and cheaper than the
/// trigonometric spelling of the same rotation.
fn transport(across: Vec3, from: Vec3, tangent: Vec3, to: Vec3, next: Vec3) -> Vec3 {
    let step = to - from;
    let square = step.length_squared();
    let (mut carried, reflected) = if square > 0.0 {
        (
            across - (2.0 / square) * step.dot(across) * step,
            tangent - (2.0 / square) * step.dot(tangent) * step,
        )
    } else {
        (across, tangent)
    };
    let turn = next - reflected;
    let square = turn.length_squared();
    if square > 0.0 {
        carried -= (2.0 / square) * turn.dot(carried) * turn;
    }
    // Re-orthogonalize against the tangent the ring is actually built on:
    // the reflections are exact in theory and drift in float.
    (carried - next * next.dot(carried))
        .try_normalize()
        .unwrap_or_else(|| orthogonal(next))
}

/// A stable integer hash — the per-strand phase and `v` come off it, so it
/// must be a pure function of the strand index and nothing else (no
/// address, no iteration order, no clock): a groom that tessellates
/// differently on its second run is a groom whose renders cannot be
/// compared. Degski's variant of the finalizer every modern PRNG ends with.
fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

impl CurveBasis {
    /// The basis matrix: weights are `[t³ t² t 1] · M`, rows ordered by
    /// descending power of `t`.
    fn matrix(self) -> [[f32; 4]; 4] {
        const SIXTH: f32 = 1.0 / 6.0;
        match self {
            Self::Bezier => [
                [-1.0, 3.0, -3.0, 1.0],
                [3.0, -6.0, 3.0, 0.0],
                [-3.0, 3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            Self::BSpline => [
                [-SIXTH, 3.0 * SIXTH, -3.0 * SIXTH, SIXTH],
                [3.0 * SIXTH, -6.0 * SIXTH, 3.0 * SIXTH, 0.0],
                [-3.0 * SIXTH, 0.0, 3.0 * SIXTH, 0.0],
                [SIXTH, 4.0 * SIXTH, SIXTH, 0.0],
            ],
            Self::CatmullRom => [
                [-0.5, 1.5, -1.5, 0.5],
                [1.0, -2.5, 2.0, -0.5],
                [-0.5, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
        }
    }

    /// How far the four-vertex window slides between spans — 3 for bezier,
    /// 1 for the approximating bases, as `UsdGeomBasisCurves` tabulates.
    fn vstep(self) -> usize {
        match self {
            Self::Bezier => 3,
            Self::BSpline | Self::CatmullRom => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One straight strand, four vertices, along +Y.
    fn straight() -> Vec<[f32; 3]> {
        (0..4).map(|index| [0.0, index as f32, 0.0]).collect()
    }

    fn linear(points: &[[f32; 3]], counts: &[u32], widths: Option<&Widths>) -> Strands {
        evaluate(
            points,
            counts,
            widths,
            CurveType::Linear,
            CurveBasis::Bezier,
            CurveWrap::Nonperiodic,
        )
        .expect("valid linear cells")
    }

    fn cubic(points: &[[f32; 3]], basis: CurveBasis, wrap: CurveWrap) -> Strands {
        evaluate(
            points,
            &[points.len() as u32],
            None,
            CurveType::Cubic,
            basis,
            wrap,
        )
        .expect("valid cubic cells")
    }

    /// `UsdGeomBasisCurves`' own worked examples, verbatim from the schema
    /// docs — the segment-count table is the contract, so the spec's rows
    /// are the oracle.
    #[test]
    fn segment_counts_match_the_usd_table() {
        let counts = |cells: &[usize], curve_type, basis, wrap| -> Vec<usize> {
            cells
                .iter()
                .map(|&count| segment_count(count, curve_type, basis, wrap).expect("valid"))
                .collect()
        };
        assert_eq!(
            counts(
                &[2, 3, 2, 5],
                CurveType::Linear,
                CurveBasis::Bezier,
                CurveWrap::Nonperiodic
            ),
            [1, 2, 1, 4]
        );
        assert_eq!(
            counts(
                &[4, 7, 10, 4, 7],
                CurveType::Cubic,
                CurveBasis::Bezier,
                CurveWrap::Nonperiodic
            ),
            [1, 2, 3, 1, 2]
        );
        assert_eq!(
            counts(
                &[5, 4, 6, 7],
                CurveType::Cubic,
                CurveBasis::BSpline,
                CurveWrap::Nonperiodic
            ),
            [2, 1, 3, 4]
        );
        // Pinned: one segment per authored gap, and two vertices is the
        // minimum the phantom points make renderable.
        assert_eq!(
            counts(
                &[2, 5],
                CurveType::Cubic,
                CurveBasis::CatmullRom,
                CurveWrap::Pinned
            ),
            [1, 4]
        );
        // A bezier count the vstep doesn't divide, and a cubic curve too
        // short to hold a span, are refused rather than rounded.
        for count in [5, 6, 3] {
            segment_count(
                count,
                CurveType::Cubic,
                CurveBasis::Bezier,
                CurveWrap::Nonperiodic,
            )
            .expect_err("the vstep must divide the count");
        }
    }

    #[test]
    fn linear_curves_pass_through_verbatim() {
        let strands = linear(&straight(), &[4], None);
        assert_eq!(strands.len(), 1);
        assert_eq!(strands.points(), 4);
        for index in 0..4 {
            assert_eq!(strands.point(index), Vec3::new(0.0, index as f32, 0.0));
            // Unauthored widths are USD's 1 — a radius of a half.
            assert!((strands.radius(index) - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn periodic_curves_are_refused() {
        let error = evaluate(
            &straight(),
            &[4],
            None,
            CurveType::Linear,
            CurveBasis::Bezier,
            CurveWrap::Periodic,
        )
        .expect_err("periodic is out of contract");
        assert!(format!("{error}").contains("periodic"), "{error}");
    }

    /// A bezier span interpolates its end control points, a b-spline
    /// approximates them, and pinning is what makes the approximating
    /// bases pass through the authored ends.
    #[test]
    fn the_bases_start_and_end_where_their_definitions_say() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let bezier = cubic(&points, CurveBasis::Bezier, CurveWrap::Nonperiodic);
        assert!(bezier.point(0).abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(
            bezier
                .point(bezier.points() - 1)
                .abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-6)
        );

        // The b-spline's one span starts at the average of its first three
        // control points weighted 1:4:1 — the basis row at t = 0.
        let bspline = cubic(&points, CurveBasis::BSpline, CurveWrap::Nonperiodic);
        let expected = (Vec3::from(points[0]) + 4.0 * Vec3::from(points[1]) + Vec3::from(points[2]))
            / 6.0;
        assert!(bspline.point(0).abs_diff_eq(expected, 1e-6));

        // Catmull-Rom interpolates its *inner* control points.
        let catmull = cubic(&points, CurveBasis::CatmullRom, CurveWrap::Nonperiodic);
        assert!(catmull.point(0).abs_diff_eq(Vec3::from(points[1]), 1e-6));
        assert!(
            catmull
                .point(catmull.points() - 1)
                .abs_diff_eq(Vec3::from(points[2]), 1e-6)
        );

        // Pinned: both approximating bases now reach the authored ends.
        for basis in [CurveBasis::BSpline, CurveBasis::CatmullRom] {
            let pinned = cubic(&points, basis, CurveWrap::Pinned);
            assert!(
                pinned.point(0).abs_diff_eq(Vec3::from(points[0]), 1e-5),
                "{basis} root"
            );
            assert!(
                pinned
                    .point(pinned.points() - 1)
                    .abs_diff_eq(Vec3::from(points[3]), 1e-5),
                "{basis} tip"
            );
        }
    }

    /// Flattening is measured against the strand's own radius: a span with
    /// no curvature never subdivides, and one whose deviation dwarfs its
    /// radius runs into the cap rather than to infinity.
    #[test]
    fn flattening_spends_segments_only_on_curvature() {
        let collinear = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let flat = cubic(&collinear, CurveBasis::Bezier, CurveWrap::Nonperiodic);
        assert_eq!(flat.points(), 2, "a straight span is one chord");

        let hairpin = [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let curled = cubic(&hairpin, CurveBasis::Bezier, CurveWrap::Nonperiodic);
        assert_eq!(
            curled.points(),
            MAX_SPAN_SEGMENTS as usize + 1,
            "the cap holds"
        );

        // The same hairpin at a radius wide enough to swallow its own
        // deviation collapses back to one chord — the rule is relative.
        let engulfed = evaluate(
            &hairpin,
            &[4],
            Some(&Widths {
                values: vec![200.0],
                interpolation: WidthInterpolation::Constant,
            }),
            CurveType::Cubic,
            CurveBasis::Bezier,
            CurveWrap::Nonperiodic,
        )
        .expect("valid");
        assert_eq!(engulfed.points(), 2);
    }

    #[test]
    fn every_width_interpolation_lands_where_it_says() {
        let points = straight();
        // Constant: one value for the whole batch.
        let constant = linear(
            &points,
            &[4],
            Some(&Widths {
                values: vec![0.4],
                interpolation: WidthInterpolation::Constant,
            }),
        );
        assert!((constant.radius(2) - 0.2).abs() < 1e-6);

        // Uniform: one per curve — two curves here, second one wider.
        let two: Vec<[f32; 3]> = points.iter().chain(points.iter()).copied().collect();
        let uniform = linear(
            &two,
            &[4, 4],
            Some(&Widths {
                values: vec![0.2, 0.6],
                interpolation: WidthInterpolation::Uniform,
            }),
        );
        assert!((uniform.radius(0) - 0.1).abs() < 1e-6);
        assert!((uniform.radius(4) - 0.3).abs() < 1e-6);

        // Varying and vertex agree on a linear curve: one per point.
        for interpolation in [WidthInterpolation::Varying, WidthInterpolation::Vertex] {
            let tapered = linear(
                &straight(),
                &[4],
                Some(&Widths {
                    values: vec![0.8, 0.6, 0.4, 0.0],
                    interpolation,
                }),
            );
            assert!((tapered.radius(0) - 0.4).abs() < 1e-6, "{interpolation:?}");
            assert!((tapered.radius(3) - 0.0).abs() < 1e-6, "{interpolation:?}");
        }

        // On a cubic curve, `vertex` widths ride the basis: a bezier's
        // root reads its first control width outright.
        let cells = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let cubic_widths = evaluate(
            &cells,
            &[4],
            Some(&Widths {
                values: vec![1.0, 0.6, 0.3, 0.0],
                interpolation: WidthInterpolation::Vertex,
            }),
            CurveType::Cubic,
            CurveBasis::Bezier,
            CurveWrap::Nonperiodic,
        )
        .expect("valid");
        assert!((cubic_widths.radius(0) - 0.5).abs() < 1e-6);
        assert!((cubic_widths.radius(cubic_widths.points() - 1) - 0.0).abs() < 1e-6);
    }

    /// A strand with nothing to sweep is dropped rather than carried: a
    /// single point has no segment, and its tube would be a ring floating
    /// in space. The cells path refuses such a curve outright, so this is
    /// the builder `.hair` fills — a groom that declares a zero-segment
    /// strand loses that strand and nothing else.
    #[test]
    fn strands_too_short_to_sweep_are_dropped() {
        let mut strands = Strands::new();
        strands.push_point(Vec3::ZERO, 0.1);
        strands.push_point(Vec3::Y, 0.1);
        strands.end_strand();
        strands.push_point(Vec3::splat(5.0), 0.1);
        strands.end_strand();
        assert_eq!(strands.len(), 1);
        assert_eq!(strands.points(), 2);
    }

    /// The tube's analytic normals are the whole reason three sides are
    /// enough: every ring vertex sits one radius off the axis, carries the
    /// radial direction it sits on, and its faces wind outward to agree.
    #[test]
    fn the_tube_is_a_prism_wearing_a_cylinder_normal() {
        let strands = linear(
            &straight(),
            &[4],
            Some(&Widths {
                values: vec![0.5],
                interpolation: WidthInterpolation::Constant,
            }),
        );
        let mesh = tessellate(&strands);
        assert_eq!(mesh.positions.len(), 4 * SIDES);
        assert_eq!(mesh.triangles.len(), 3 * SIDES * 2);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
            // The axis is +Y, so the radial offset is the XZ part.
            let radial = Vec3::new(position.x, 0.0, position.z);
            assert!((radial.length() - 0.25).abs() < 1e-5, "{position}");
            assert!(normal.is_normalized(), "{normal}");
            assert!(normal.dot(Vec3::Y).abs() < 1e-5, "{normal} leans along +Y");
            assert!(
                radial.normalize().dot(*normal) > 0.999,
                "the stored normal is the radial direction"
            );
        }

        for triangle in &mesh.triangles {
            let [a, b, c] = triangle.map(|index| mesh.positions[index as usize]);
            let face = (b - a).cross(c - a);
            let outward: Vec3 = triangle
                .iter()
                .map(|&index| mesh.normals[index as usize])
                .sum();
            assert!(
                face.dot(outward) > 0.0,
                "face {triangle:?} winds inward: {face} against {outward}"
            );
        }
    }

    /// `u` is arc length root to tip, `v` is one value the strand shares —
    /// the two handles a hair lookdev reaches for.
    #[test]
    fn the_strand_coordinates_run_root_to_tip() {
        let points = [[0.0; 3], [0.0, 1.0, 0.0], [0.0, 3.0, 0.0]];
        let mesh = tessellate(&linear(&points, &[3], None));
        let u: Vec<f32> = mesh.uvs.iter().step_by(SIDES).map(|uv| uv.x).collect();
        assert_eq!(u.len(), 3);
        assert!((u[0] - 0.0).abs() < 1e-6);
        // A third of the way along by *length*, not by point index.
        assert!((u[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((u[2] - 1.0).abs() < 1e-6);
        let v = mesh.uvs[0].y;
        assert!(mesh.uvs.iter().all(|uv| (uv.y - v).abs() < 1e-9));
        assert!((0.0..1.0).contains(&v));
    }

    /// Neighbouring strands must not share a ring phase (that is the
    /// moiré the hash exists to break), and the same groom must tessellate
    /// identically every time (that is what makes two renders comparable).
    #[test]
    fn the_per_strand_phase_is_varied_and_stable() {
        let one = straight();
        let two: Vec<[f32; 3]> = one.iter().copied().chain(one.iter().copied()).collect();
        let mesh = tessellate(&linear(&two, &[4, 4], None));
        let first = mesh.normals[0];
        let second = mesh.normals[4 * SIDES];
        assert!(first.dot(second) < 0.999, "both strands start at one angle");

        let again = tessellate(&linear(&two, &[4, 4], None));
        assert_eq!(mesh.positions, again.positions);
        assert_eq!(mesh.normals, again.normals);
        assert_eq!(mesh.uvs, again.uvs);
        assert_eq!(mesh.triangles, again.triangles);
    }

    /// A groom with a duplicated point (every exporter writes one
    /// eventually) must not produce a frame full of NaN.
    #[test]
    fn coincident_points_keep_the_frame_finite() {
        let points = [
            [0.0; 3],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let mesh = tessellate(&linear(&points, &[4], None));
        assert!(mesh.positions.iter().all(|p| p.is_finite()));
        assert!(mesh.normals.iter().all(|n| n.is_normalized()));
        assert!(mesh.uvs.iter().all(|uv| uv.is_finite()));
    }

    /// The frame is rotation-minimizing: carried along a curve that turns
    /// through a right angle, the ring picks up no twist about the tangent
    /// beyond what the turn itself forces.
    #[test]
    fn the_frame_adds_no_twist_of_its_own() {
        // An L: up +Y, then along +X. The reference direction starts
        // perpendicular to both and must survive the corner untouched.
        let points = [
            [0.0; 3],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let mesh = tessellate(&linear(&points, &[5], None));
        // The plane of the bend is XY, so +Z is the direction the frame
        // has no reason to rotate away from: every ring keeps a vertex
        // pointing the same way out of plane.
        let out_of_plane = |ring: usize| {
            (0..SIDES)
                .map(|side| mesh.normals[ring * SIDES + side].z)
                .fold(f32::MIN, f32::max)
        };
        let first = out_of_plane(0);
        for ring in 1..5 {
            assert!(
                (out_of_plane(ring) - first).abs() < 1e-4,
                "ring {ring} twisted: {} vs {first}",
                out_of_plane(ring)
            );
        }
    }
}
