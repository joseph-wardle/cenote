// Temporal reprojection: map this frame's shading point back into last frame's
// screen, so the temporal combine folds in the reservoir the *previous* camera
// resolved for that surface — not blindly the same pixel. Plus the persistent
// G-buffer that reprojection reads, and the disocclusion gate that rejects a
// reprojected sample whose surface no longer matches. All of it is **variance
// and quality, never bias** (D-094): a wrong reprojection or a gate that lets a
// mismatch through is only a poor candidate the pairwise MIS down-weights, so
// this module carries none of the temporal stage's correctness weight — that is
// the M-cap and the feed convention, both upstream in `restir_temporal.slang`.
//
// -- The G-buffer -------------------------------------------------------------
// A `StoredReservoir` holds only the light point, nothing about the receiving
// surface, so the gate — which compares this pixel's surface against the one the
// reprojected `prev` was resampled for — needs last frame's *hit* persisted.
// This stores the re-evaluable hit + ray (not a resolved position/normal), so
// `reconstructSurfaceFrom` rebuilds the full previous surface — closure included
// — the same way the path tracer would, and `p̂_i` weighs the neighbour on its
// real BSDF rather than a lossy normal. Written this frame, read (as `prev`)
// next frame; the pair ping-pongs with the reservoirs (D-094). No clear: a
// stale entry is only ever read where the reprojected `prev` reservoir carries
// confidence, and that is exactly a pixel this pass wrote its G-buffer for.
//
// -- Reprojection is the exact inverse of raygen ------------------------------
// raygen builds a pixel's ray as `dir ∝ forward + ndc.x·right + ndc.y·up` with
// the orthogonal camera basis (right/up scaled by tan(vfov/2)·aspect, forward
// unit). Because the basis is orthogonal, recovering the NDC of a world point is
// a dot-product decomposition — no matrix inverse. For a held camera this lands
// a point exactly back on its own pixel, so the combine reduces to same-pixel
// reuse; a moved camera relocates it. The pinhole basis is used even for a
// thin-lens camera (reprojecting the real hit, not a focal point) — an aperture
// approximation that costs a little reuse quality, never correctness.

import pathstate;

// Last frame's primary hit at one pixel, re-evaluable exactly as the path pool's
// live hit is (`reconstructSurfaceFrom`). 48 bytes; mirrors the host stride in
// `restir/view.rs`.
public struct GBufEntry {
    public Hit hit;          // instance + primitive + barycentrics
    public float4 origin;    // xyz = ray origin; w unused
    public float4 direction; // xyz = unit ray dir; w = camera-forward cosine (depth)
};

// The per-frame reprojection state the temporal stage reaches through one push
// pointer: last frame's camera basis (to reproject), the ping-ponged G-buffers,
// the frame dimensions, and the reuse-gate thresholds. Host-written each frame
// into a `CpuToGpu` buffer (`Reproject` in `wavefront.rs`); the basis is the
// *previous* frame's, captured after the swap. `valid` is 0 on the first frame
// after a reset or resize — no meaningful previous camera — so reprojection is
// skipped and the (empty) same-pixel history reads as nothing.
public struct Reproject {
    public float3 position;   public float normalThreshold; // prev camera eye
    public float3 right;      public float depthThreshold;  // scaled basis right
    public float3 up;         public float pad0;            // scaled basis up
    public float3 forward;    public float pad1;            // unit forward
    public GBufEntry* prevGbuffer; // last frame's hits — read
    public GBufEntry* currGbuffer; // this frame's hits — written (→ prev next frame)
    public uint width;
    public uint height;
    public uint valid;
    public uint pad2;
};

// Reproject a world point into last frame's pixel. Returns false — reuse the
// same pixel's (or no) history — when reprojection is disabled (`valid == 0`),
// the point is behind the previous camera, or it falls off last frame's screen.
// The recovered pixel is `floor(uv · dims)`, exactly raygen's `pixel → xy` map
// run backwards, so a held camera round-trips a point to its own pixel.
public bool reprojectPixel(Reproject r, float3 world, out uint pixel) {
    pixel = 0;
    if (r.valid == 0) {
        return false;
    }
    let v = world - r.position;
    let vf = dot(v, r.forward);
    if (vf <= 0.0) {
        return false; // behind the previous camera
    }
    // ndc from the orthogonal-basis decomposition: dot(v, right)/(vf·|right|²).
    let ndcX = dot(v, r.right) / (vf * dot(r.right, r.right));
    let ndcY = dot(v, r.up) / (vf * dot(r.up, r.up));
    let u = ndcX * 0.5 + 0.5;      // raygen: ndc.x = 2u − 1
    let t = 0.5 - ndcY * 0.5;      // raygen: ndc.y = 1 − 2t
    if (u < 0.0 || u >= 1.0 || t < 0.0 || t >= 1.0) {
        return false; // off last frame's screen
    }
    let px = uint(u * float(r.width));
    let py = uint(t * float(r.height));
    if (px >= r.width || py >= r.height) {
        return false; // guard the rare u·width == width at the boundary
    }
    pixel = py * r.width + px;
    return true;
}

// The disocclusion gate: do two surfaces describe the same shading context? The
// same orientation-and-camera-depth test the spatial stage runs on its
// neighbours (`restir_spatial.slang`), on the same thresholds, so temporal and
// spatial reuse gate identically. A reprojected `prev` that fails — a silhouette
// the camera moved across, a depth discontinuity — is dropped, leaving the
// canonical as the sole candidate (no bias, just less reuse).
public bool surfacesAgree(float3 currNormal, float currDepth, float3 prevNormal,
                          float prevDepth, float normalThreshold,
                          float depthThreshold) {
    if (dot(currNormal, prevNormal) <= normalThreshold) {
        return false;
    }
    if (abs(currDepth - prevDepth) > depthThreshold * currDepth) {
        return false;
    }
    return true;
}
