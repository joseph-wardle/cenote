// Colour primitives shared across the wavefront, in one leaf.

// Relative luminance of an ACEScg (AP1) colour — the CIE Y row of the AP1→XYZ
// matrix — the film's variance substrate.
public float luminance(float3 acescg) {
    return dot(acescg, float3(0.2722287168, 0.6740817658, 0.0536895174));
}
