// The shared-evaluation records of the split spatial stage (M6 step 7b): what
// the gather pre-pass (`restir_spatial_gather.slang`) writes and the ray-free
// combine (`restir_spatial.slang`) reads. One module, imported by both, so the
// two kernels can never disagree about the schema.
//
// The split exists because selection is reciprocal (7a, pairing.slang): A
// gathers B on slot n exactly when B gathers A, so the pair's two shift
// evaluations — B's sample forward into A's domain, and A's sample forward
// into B's — are each *some* pixel's forward gather. The pre-pass runs every
// pixel's forward evaluations once (gates, shifts, the k+1 visibility rays at
// its own surface) and records the shift-dependent values; the combine then
// reads its **own** records for the forward terms and its **partner's** slot-n
// record for the backshift — the evaluation the pre-7b stage re-ran from
// scratch (`targetInDomain`, 59–65% of the spatial pass by D-146's
// measurement). Rays stay at the receiving pixel: nothing about the estimator
// changes, only where its numbers are computed — the checkpoint is
// bit-identity with 7a.
//
// Why a partner's record always exists when read: the combine reads slot n's
// partner records only when its own pre-pass accepted that partner, and every
// reuse gate is symmetric in the pair (bounds and range trivially; confidence
// because an accepted partner was reached and hit, so its own thread ran;
// the normal and min-depth tests by 7a's construction). Acceptance at A
// therefore implies B's thread ran and accepted A on the same slot — the
// involution guarantees B's slot-n partner *is* A — so B's records are fresh
// this dispatch, and no clear of the buffers is ever needed.
//
// Both records are indexed by *path slot*, not pixel: slots are live and
// unique within a range, a partner's slot is already in hand from the range
// guard, and capacity-sized buffers are allocatable at wavefront init. The
// layouts follow the reservoir convention (reservoir_path.slang): std430
// 16-byte lanes, each float3 opening a lane and a scalar closing it, so the
// host's byte sizes (wavefront.rs) hold without alignment surprises.

// One forward gather: pixel P's slot-n partner's sample, shifted into P's
// domain and priced there. Written at pairs[slot·k + n] for every accepted
// partner; never written (and never read — the accept mask gates it) for a
// rejected one. 32 B.
public struct PairRecord {
    // The shifted sample's integrand in P's domain — `DomainShift.sample.F`
    // for the indirect kinds (the F to patch onto the stored sample on a
    // win; the original F when the shift was undefined), the unshadowed RGB
    // contribution for an NEE sample. The chroma numerator either way — and,
    // read from the partner's side, an NEE canonical's backshift RGB (fBack).
    public float3 F;
    // The cachedJacobian to store on a win — `DomainShift.sample`'s, i.e. the
    // destination half for a valid reconnection shift, the stored sample's
    // own otherwise. Unused (never patched) for an NEE sample.
    public float partial;
    // p̂_c(shift(Y_i))·J — visibility-aware, the one ray this pair spends at P.
    public float visTarget;
    // U_c(shift(Y_i)) — the finalize divisor if this sample wins, and the
    // chroma denominator.
    public float unshadowed;
    // p̂_i(Y_i), the partner's own-domain target for the pairwise ratios —
    // shift-guarded for the indirect kinds (the D-136 partial-shift zero).
    // Zero for an NEE sample: its own-domain target is position-independent
    // of the shift and lives in the partner's SelfRecord instead.
    public float ownTarget;
    // Exactly what `targetInDomain(Y_i, P's surface)` returns — unshadowed
    // p̂·J through the same shift. The reciprocal partner reads this as its
    // backshift term p̂_i(Y_c): the sharing this split exists for.
    public float backTarget;
};

// One pixel's evaluation of its own sample at its own surface, written once
// per pre-pass thread at selfs[slot]. 48 B.
public struct SelfRecord {
    // The raw unshadowed RGB contribution (NEE) or the stored integrand F
    // (indirect) — the canonical chroma numerator, and, read from the
    // partner's side, an NEE partner's own-domain fOwn for the CV residual.
    public float3 rgb;
    // p̂_c(Y_c) — visibility-aware, under the pre-split stage's exact guard
    // (zero when the pixel's own W is zero; the ray is only spent when the
    // unshadowed target is positive). Only the pixel itself reads this.
    public float pcYc;
    // The primary closure's aovAlbedo — both sides of the CV's per-channel
    // albedo ratio α read it here, sparing the combine any surface
    // reconstruction.
    public float3 albedo;
    // U_c(Y_c) — *unguarded*: the partner reads it as an NEE sample's
    // own-domain target p̂_i(Y_i) (the pre-split stage evaluated that
    // unconditionally), and the pixel itself re-applies its W guard.
    public float ucYc;
    // Bit n set: the slot-n partner passed every reuse gate — the combine's
    // whole memory of the gather's accept/reject decisions (V, c_tot, and
    // which pair records are live).
    public uint accepted;
    public uint pad0;
    public uint pad1;
    public uint pad2;
};
