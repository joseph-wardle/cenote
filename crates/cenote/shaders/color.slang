// Colour primitives shared across the wavefront, in one leaf so the film and the
// ReSTIR reservoir weigh brightness with the exact same scalar.

// Relative luminance of an ACEScg (AP1) colour — the CIE Y row of the AP1→XYZ
// matrix. The scalar the reservoir sorts candidates on, and the film's variance
// substrate; one definition, so they can never drift.
public float luminance(float3 acescg) {
    return dot(acescg, float3(0.2722287168, 0.6740817658, 0.0536895174));
}
