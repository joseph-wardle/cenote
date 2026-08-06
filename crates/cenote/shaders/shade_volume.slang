// The volume stage: paths whose segment has a medium event to sample
// arrive here from intersect instead of shade_surface or shade_miss. Each
// draws a distance against its medium's extinction; a path that scatters
// before the segment ends turns through the phase function and sends its
// own next-event connection, and one that does not carries the segment's
// transmittance on to whatever intersect found.
//
// A stage of its own rather than a branch in shade_surface: it is recorded
// only when the scene has media, so media-free scenes keep a byte-identical
// pass list and no scene pays register pressure for medium code it never
// runs.
//
// Radiance bookkeeping matches shade_surface's: every write is a
// pixel-owned add, and a path that ends here carries its pixel's one
// terminal alpha.

import lights;
import medium;
import nee;
import pathstate;
import rng;
import scene;
import surface;

// Bit 24 of `Params.packed`, as in shade_surface: the scene has closed
// transmissive surfaces, so the medium stack exists.
static const uint FLAG_INTERIORS = 1 << 24;

// The five queues this stage touches, behind one pointer: it consumes one
// and feeds four, which together are more push-constant bytes than Vulkan
// guarantees. Their addresses are fixed for the wavefront's lifetime, so the
// block is written once — unlike the per-bounce scalars below.
struct VolumeQueues {
    Queue<uint> volumes;      // this stage's input, pushed by intersect
    Queue<uint> hits;         // segments that reached a surface
    Queue<uint> misses;       // segments that reached nothing
    Queue<uint> rays;         // the next bounce's input: scatters push here
    Queue<ShadowRay> shadows; // next-event connections, for trace_shadow
};

struct Params {
    Paths paths;
    VolumeQueues* queues;
    SceneTable* scene;
    float4* radiance; // the film's per-pixel accumulator for this wave
    AovTable* aov;    // the wave's AOV targets; enabled = 0 skips them
    uint sampleIndex; // which sample of every pixel's sequence this wave is
    // bounce | maxBounces << 8 | lightSampling << 16 | FLAG_INTERIORS —
    // shade_surface's packing exactly.
    uint packed;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void shade_volume(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.queues.volumes.count()) {
        return;
    }
    let bounce = params.packed & 0xff;
    let maxBounces = (params.packed >> 8) & 0xff;
    let lightSampling = (params.packed >> 16) & 0xff;
    let interiors = (params.packed & FLAG_INTERIORS) != 0;
    uint path = params.queues.volumes.entries[tid.x];
    uint pixel = params.paths.pixel[path];
    Hit hit = params.paths.hit[path];
    let direction4 = params.paths.direction[path];
    let direction = direction4.xyz;
    let origin = params.paths.origin[path].xyz;
    var throughput = params.paths.throughput[path];

    // How far the segment runs. The surface point is recomputed rather than
    // carried from intersect: a distance in the hit record would cost every
    // scene 8 bytes of path state, and the pool is residency-bound.
    var tmax = 1e30;
    if (hit.instance != INSTANCE_NONE) {
        let mesh = params.scene.geometry[hit.instance];
        tmax = distance(origin, spawnPoint(mesh, hit, direction).position);
    }

    // What fills the segment: the interior on the path's stack, or — in
    // open space — the global medium. Intersect routes a path here only
    // when one of the two does.
    let entry = interiors && bounce != 0 ? params.paths.top(path) & STACK_INDEX_MASK
                                         : STACK_EMPTY;
    let record = params.scene.media[entry == STACK_EMPTY
                                        ? params.scene.globalMedium
                                        : params.scene.geometry[entry].medium];

    let uDistance = sample_2d(pixel, params.sampleIndex,
                              bounceDimension(bounce, VOLUME_DISTANCE));
    let event = sampleMedium(record, tmax, uDistance);
    throughput.xyz *= event.weight;

    if (!event.scattered) {
        // The segment reached its far end: whatever intersect found shades
        // as ever, carrying the medium's transmittance. Origin and the
        // previous scatter's pdf are left alone — the vertex those describe
        // is the one before this segment, which is still where the next
        // stage's MIS weight is measured from.
        params.paths.throughput[path] = throughput;
        if (hit.instance == INSTANCE_NONE) {
            params.queues.misses.push(path);
        } else {
            params.queues.hits.push(path);
        }
        return;
    }

    let position = origin + event.distance * direction;
    let aovOn = params.aov.enabled != 0;
    if (aovOn) {
        if (bounce == 0) {
            // First vertex: camera-plane z, via the cosine raygen stashed
            // in direction.w. A pixel that sees only fog has its depth here
            // or nowhere.
            params.aov.depth[pixel] += event.distance * direction4.w;
        }
        // The guides close on the medium: its single-scattering albedo is
        // what a denoiser should divide out here, and a scattering point
        // has no surface, so — as with the environment — no normal.
        let guide = bounce == 0 ? float3(1.0) : params.aov.guide[pixel].xyz;
        if (any(guide > 0.0)) {
            let albedo = record.sigmaS / max(record.sigmaT, 1e-9);
            params.aov.albedo[pixel] += float4(guide * albedo, 0.0);
        }
        params.aov.guide[pixel] = float4(0.0);
    }

    // The depth cap: the path ends at this vertex, so this is its pixel's
    // terminal add. No next-event estimation here — a connection from this
    // vertex would be one segment longer than any path phase sampling can
    // trace, and the two strategies must cover the same path lengths.
    if (bounce + 1 >= maxBounces) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Next-event estimation from inside the medium, drawn and weighted by
    // the shared `sampleNeeVolume` (nee.slang). The connection's own
    // transmittance is trace_shadow's to apply.
    let wo = -direction;
    if (lightSampling != LIGHT_SAMPLING_BSDF
        && (params.scene.lightCount > 0 || params.scene.env.selectProb > 0.0)) {
        let uLight = sample_1d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_LIGHT));
        let uPoint = sample_2d(pixel, params.sampleIndex,
                               bounceDimension(bounce, NEE_POINT));
        let nee = sampleNeeVolume(params.scene, position, record.g, wo,
                                  throughput.xyz, lightSampling, uLight, uPoint);
        if (any(nee.unshadowed > 0.0)) {
            ShadowRay shadow;
            shadow.origin = nee.origin;
            shadow.light = nee.light;
            shadow.direction = nee.direction;
            shadow.primitive = nee.primitive;
            shadow.contribution = nee.unshadowed;
            shadow.tmax = nee.tmax;
            shadow.pixel = pixel;
            shadow.pad0 = 0;
            shadow.pad1 = 0;
            shadow.pad2 = 0;
            params.queues.shadows.push(shadow);
        }
    }

    // Turn through the phase function. HG samples its own density exactly,
    // so the estimator's f/pdf is 1 and the throughput carries only what
    // the distance sample cost it.
    let uPhase = sample_2d(pixel, params.sampleIndex,
                           bounceDimension(bounce, BSDF_DIR));
    let scatter = samplePhase(record.g, wo, uPhase);
    var scattered = throughput.xyz;

    var survival = max(scattered.r, max(scattered.g, scattered.b));
    if (survival <= 0.0) {
        params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (bounce >= ROULETTE_START) {
        survival = min(1.0, survival);
        if (sample_1d(pixel, params.sampleIndex,
                      bounceDimension(bounce, RR)) >= survival) {
            params.radiance[pixel] += float4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        scattered /= survival;
    }

    // The path continues from inside the same medium — a scatter crosses no
    // boundary, so the stack is untouched. Direction.w is zeroed: only
    // camera rays have a camera to measure depth against.
    params.paths.origin[path] = float4(position, 0.0);
    params.paths.direction[path] = float4(scatter.wi, 0.0);
    params.paths.throughput[path] = float4(scattered, scatter.pdf);
    params.queues.rays.push(path);
}
