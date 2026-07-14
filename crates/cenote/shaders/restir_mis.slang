// Defensive pairwise-MIS ratios (Bitterli 2022 thesis §9.1, Eq 7.8), factored
// out of the reuse stages so the shipping combine and its analytic microtests
// evaluate the *same* normalizer. A retyped copy could drift a few percent and
// still render — the silent bias those tests exist to catch — so there is one
// definition here, with no scene dependency (pure confidences and targets), and
// the tests import exactly it.
//
// For one non-canonical candidate weighed against the canonical: `bi` is the
// neighbour's share of the pair at its own sample Y_i, `canonShare` the
// canonical's share at its sample Y_c. `ci`/`cCanon` are the two confidences;
// the four p̂ are the two techniques' targets at the two samples. Each
// denominator guards an all-zero pair — it contributes nothing and must not
// divide.
public struct PairwiseMis {
    public float bi;
    public float canonShare;
};

public PairwiseMis pairwiseMis(float ci, float piYi, float piYc,
                               float cCanon, float pcYi, float pcYc) {
    PairwiseMis m;
    let bDenom = ci * piYi + cCanon * pcYi;
    m.bi = bDenom > 0.0 ? (ci * piYi) / bDenom : 0.0;
    let shareDenom = ci * piYc + cCanon * pcYc;
    m.canonShare = shareDenom > 0.0 ? (cCanon * pcYc) / shareDenom : 0.0;
    return m;
}
