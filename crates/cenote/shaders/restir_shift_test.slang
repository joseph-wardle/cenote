// Test fixture: same-pixel replay bit-identity (the step-3b/3c gate,
// D-138/D-139). Shifting a stored indirect sample into its *own* pixel's
// domain must reproduce the stored integrand F bit for bit — and, for a
// reconnection sample, a Jacobian of 1 to the ulp (its squared-distance
// ratio's last bit belongs to per-kernel fma contraction; a replay sample's
// Jacobian is 1 by construction) — the strongest invariant the hybrid shift
// offers. A single mis-associated multiply, a draw keyed off the wrong
// stream, a roulette division leaking into the replayed chain, or a re-shade
// context drifting from generation's all break the equality outright, where a
// convergence gate would smear the same bug into plausible-looking noise.
//
// For every primary hit this kernel reconstructs the surface exactly as the
// reuse stages do, runs the *shipping* shift — `shiftReconnection` or
// `shiftReplay` by the stored kind — on the pixel's own candidate-stage
// survivor, and dumps the result beside a tag for the host (`wavefront.rs`)
// to compare against the stored sample. Not embedded — the host compiles it
// on demand (`shaders::compile_fixture`).

import pathstate;
import reservoir_path;
import restir_scene;
import restir_target;
import scene;

// The scene TLAS, as the reuse stages bind it: the k > 2 shift's replay
// traces the prefix through it.
[vk::binding(0, 0)]
RaytracingAccelerationStructure sceneTlas;

// An indirect sample was evaluated; the low byte carries its k (reconnection)
// or terminal bounce (replay).
static const uint SHIFT_EVALUATED = 0x100;
// A sample's own-pixel shift came back invalid — always a bug: generation and
// the shift share their validity predicates (`reconnectionEnd`; the replayed
// chain's own concrete traces), so a stored survivor must shift at its own
// pixel.
static const uint SHIFT_BROKEN = 0x200;
// The evaluated sample was replay-kind; bits 12..13 then carry its terminal
// kind (RC_TERMINAL_NEE / RC_TERMINAL_SCATTER) for the host's coverage.
static const uint SHIFT_REPLAY = 0x400;

struct Params {
    Paths paths;
    Queue<uint> hits;                // primary hits, dispatched over; not consumed
    SceneTable* scene;
    StoredPathReservoir* reservoirs; // the candidate stage's per-pixel output
    float4* shifted;                 // out: xyz = shifted F, w = Jacobian
    uint* kinds;                     // out: SHIFT_* tag | k (0 = not a shiftable sample)
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void restir_shift_test(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.hits.count()) {
        return;
    }
    uint slot = params.hits.entries[tid.x];
    uint pixel = params.paths.pixel[slot];

    var outF = float4(0.0);
    var outKind = 0u;
    let stored = params.reservoirs[pixel];
    if (stored.confidence > 0.0 && isReconnectionSample(stored.sample)
        && reconnectionShiftable(stored.sample)) {
        Surface s = reconstructSurface(params.paths, params.scene, slot);
        let sh = shiftReconnection(sceneTlas, params.scene, stored.sample, s);
        if (sh.valid) {
            outF = float4(sh.F, sh.jacobian);
            outKind = SHIFT_EVALUATED | reconnectionK(stored.sample);
        } else {
            outKind = SHIFT_BROKEN;
        }
    } else if (stored.confidence > 0.0 && isReplaySample(stored.sample)) {
        Surface s = reconstructSurface(params.paths, params.scene, slot);
        let sh = shiftReplay(sceneTlas, params.scene, stored.sample, s);
        var F = sh.F;
        var live = sh.valid;
        if (live && sh.tmax > 0.0) {
            // The NEE terminal's own shadow ray: generation folded the same
            // visibility into the stored F, so the comparison applies it in
            // the same last position.
            RayDesc ray;
            ray.Origin = sh.rayOrigin;
            ray.Direction = sh.direction;
            ray.TMin = 0.0;
            ray.TMax = sh.tmax;
            let vis = shadowVisibility(sceneTlas, params.scene, ray,
                                       sh.shadowLight, sh.shadowPrimitive);
            F *= vis;
            // Generation only streams a visible connection, so the same-pixel
            // redraw must see the same unoccluded ray.
            live = vis > 0.0;
        }
        if (live) {
            // The replay shift is a primary-sample-space identity: Jacobian 1
            // by construction, reported as such for the host's uniform check.
            outF = float4(F, 1.0);
            outKind = SHIFT_EVALUATED | SHIFT_REPLAY
                    | replayTerminalBounce(stored.sample)
                    | (replayTerminal(stored.sample) << 12);
        } else {
            outKind = SHIFT_BROKEN;
        }
    }
    params.shifted[pixel] = outF;
    params.kinds[pixel] = outKind;
}
