# The scene corpus campaign

*Interviewed 2026-07-30; rung 0 recon ran the same day. An off-roadmap
campaign: replace cenote's homemade test scenes with the corpus the
rendering literature actually measures on.*

## 1. The nine decisions

| # | Decision | Substance |
|---|---|---|
| 1 | Role | Benchmark + portfolio gallery. Tests and goldens stay put; promotion happens only through the swap rung (6) |
| 2 | Scope | All 19 scenes, one campaign. `volumetric-caustic` is a documented placeholder until M8 volumes |
| 3 | Storage | Committed: curated `.ron`s, `scenes/corpus/README.md`, `fetch.sh`, this plan. Untracked: `scenes/corpus/sources/` (~8.5 GB). Nothing redistributed — license variance sidestepped |
| 4 | RON lifecycle | The committed RON is **source of truth**. `import` bootstraps once; re-imports go to scratch for diffing, never clobber. Curation (header, camera, exposure, warning triage) lives in the RON |
| 5 | Gap docs | Story in each RON header (provenance, license, every knowing degradation with its unlock); greppable index table in the corpus README; one pointer in deferrals.md |
| 6 | Replacement | **Swap within the campaign**: veach-ajar takes brass-room's reuse-gate role and zero-day succeeds many-lights *if they measure well* (gate + convergence numbers beside brass-room's). Gated on M6 closed — the swap mutates a closed milestone's artifacts deliberately |
| 7 | Scene bar | Two-class: gap-free scenes must visually match a local pbrt-v4 reference side-by-side (divergence = importer bug → fix); gapped scenes get the same side-by-side with divergence documented. Contact sheets per batch. Baseline for all: scripted fetch reproduces, RON opens in the viewer, fits VRAM, warnings triaged, README row |
| 8 | Importer scope | Extend `cenote-pbrt` when it pays — importer-local, renderer already capable, ≲ a day, multi-scene wins first; mid-rung finds flagged and sized, bigger parked as importer-debt. Renderer gaps (volumes, curves, SSS, spectral) always document-only |
| 9 | Ladder | §4. Batched smalls, heavy singletons, per-rung Go |

## 2. Sources

Two, both discovered better than planned: Bitterli now publishes
**official pbrt-v4 exports** (no v3 upgrade step at all), and the repo
already had the pattern to extend — `tests/scenes/` vendors hand-converted
cornell-box/veach-mis/teapot-full for CI (those stay frozen; the corpus
imports the official zips instead, uniformly), and
`tests/scenes/fetch-showcase.sh` established sparse-pinned fetching from
pbrt-v4-scenes. `scenes/corpus/fetch.sh` generalizes it: Bitterli zips
whole, pbrt-v4-scenes as one sparse partial clone at the **same pin**
(`30cf4a0`), content-addressing as the checksum.

| Source | Scenes | License notes |
|---|---|---|
| [benedikt-bitterli.me/resources](https://benedikt-bitterli.me/resources/) | bathroom (Mareck), kitchen (Jay-Artist, CC BY 3.0), coffee (cekuhnen, CC BY 3.0), glass-of-water (aXel), spaceship (thecali), cornell-box, teapot-full, veach-ajar, veach-bidir, veach-mis, volumetric-caustic, water-caustic (Bitterli) | CC0 except the two CC BY; each zip ships its LICENSE.txt |
| [pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes) @ `30cf4a0` | bmw-m6 (CC0), bistro (CC-BY 4.0), crown, sanmiguel (permissions per repo README), kroken, watercolor (**CC-BY-ND 2.0**), zero-day (Beeple's release terms) | **ND flag**: a converted RON is arguably a derivative — whether kroken/watercolor RONs can be *committed* (vs script-generated locally) is decided at their rungs, with the license text in hand |

Multi-camera heavies (kroken ×7, watercolor ×18, sanmiguel ×9, bistro ×3,
zero-day ×9 frames): the curated RON picks **one canonical view** at its
rung; recon used camera-1 / courtyard / cafe / frame25.

## 3. Rung-0 recon (2026-07-30): every scene dry-imported

17 of 19 import; cornell-box renders end-to-end (the whole
import→relativize→prep→render pipeline proven). The two failures and
every warning class are below. Zero code changes were needed for the
mechanics: `ChangeSet::relativize_paths` already turns
`--out scenes/corpus/<name>.ron` into committed-friendly
`sources/...`-relative paths, and derived assets (`<name>-sky.exr`) land
beside the RON, gitignored.

**Import failures (both small importer-local fixes, proposed below):**

- **teapot-full** — its sky is a `.pfm`; the env pipeline reads EXR only.
- **sanmiguel** — `texture "Map #483" is defined twice`: pbrt tolerates
  redefinition (last wins), our importer rejects it.

**Render blocker found by smoke test:** the corpus leans on **TGA
textures** (veach-ajar, kitchen, bathroom, …). cenote's texture prep
decodes PNG/JPEG/HDR via format sniffing — TGA has no magic bytes, so it
needs the extension-hinted decode path plus the `image` crate's `tga`
feature. Import succeeds; *render* fails at prep. Core-side, small.

**Gap map** (per scene, from the harvested warning lists; scratch keeps
the verbatim lists):

| Scene | Warnings | The story |
|---|---|---|
| cornell-box | 0 | Clean |
| water-caustic | 1 | Clean (cosmetic integrator param) |
| veach-mis | 4 | Equal-axis aniso averaged — exact, effectively clean |
| veach-bidir | 1 | One aniso average |
| glass-of-water | 2 | Two aniso averages |
| coffee | 3 | Aniso averages |
| spaceship | 9 | Aniso averages |
| veach-ajar | 7 | Aniso ×4; **TGA-blocked** until the decode fix |
| teapot-full | fails | **PFM sky**; tea medium → M8 (known from the CI corpus) |
| volumetric-caustic | 5 | Media → M8 placeholder, as planned |
| bathroom | 25 | Aniso ×10, displacement ×2, one alpha-textured shape; TGA |
| kitchen | 62 | Aniso ×39, displacement ×2, diffuse transmission flattened; TGA |
| bmw-m6 | 7 | Near-clean; one interior "interface" material defaults |
| crown | 28 | Dispersion → IOR 1.5, gem media interfaces → M8, textured-roughness remap caveat |
| bistro | 251 | **243 alpha-textured shapes** — the foliage cutouts; solid until shape-alpha lands |
| kroken | 173 | UV transforms ×18, procedural-texture params dropped; ND license |
| watercolor | 184 | Same class as kroken; ND license |
| sanmiguel | fails | Duplicate texture definition (fix below); expect bistro-class alpha foliage after |
| zero-day | 188 | **ReverseOrientation on emissive plymesh ×97** (emission side needs the eyeball test), textured coat roughness imports smooth, texture scale 0.01 dropped |

**Proposed fix candidates** (each lands only with its rung's Go, per
decision 8):

| Fix | Class | Where | Unblocks |
|---|---|---|---|
| TGA decode (extension-hinted + `tga` feature) | core, small | **landed, rung 1** | veach-ajar renders; kitchen, bathroom, and others texture correctly |
| V-flip imported/PLY UVs to sampler storage order | importer + core, small | **landed, rung 1** (found by decision 7's side-by-side) | every textured import; see below |
| PFM read for infinite-light images | importer, small | **landed, rung 2** | teapot-full imports |
| Texture redefinition tolerated (last wins + warning) | importer, small | sanmiguel rung | sanmiguel imports |
| Shape `alpha` texture → material opacity | importer, medium | bistro rung (decide there) | bistro/sanmiguel foliage; bathroom's one shape |

**Rung-1 discovery — the UV convention gap.** veach-ajar's landscape
painting rendered upside-down, and the side-by-side bar (decision 7) ran
it to ground: pbrt inverts `t` at every image-texture lookup
(`textures.cpp`: "texture coordinates are (0,0) in the lower left
corner, but image coordinates are (0,0) in the upper left") — the
interchange convention PLY exporters share — while cenote samples
storage order, `v = 0` at the top row. Every textured import diverged.
Two one-line flips close it: the importer stores `1 - v` for authored,
sphere, and disk UVs (`cenote-pbrt/src/map/shape.rs`), and the PLY
reader flips `v` at the convention boundary (`cenote/src/ply.rs`). Fallout
worth knowing: the vendored teapot-full's checker floor had been
phase-inverted against pbrt since the CI trio landed — RMSE to the pbrt
reference improves from 0.250 to 0.133 (the rest is the tea medium, an
M8 gap) — so the importer golden and the README comparison figures
regenerate with this rung. Open question parked, not fixed: the Hydra
delegate passes USD `st` (also v-up interchange) verbatim, invisible
today because its only textured fixture is a symmetric checker — decide
at the next Hydra rung.

**Rung-2 discoveries.** Two the recon's import failure had masked (its
warning harvest never ran for teapot-full). First, teapot-full's floor
is a *procedural checkerboard* texture, which imports mid-gray; rather
than grow a procedural-texture baker mid-rung, the RON is curated to the
identical 20×20 checker the CI corpus already bakes to a committed PNG
(`tests/scenes/teapot-full/textures/checker.png`) — a checkerboard bake
stays a parked importer candidate for kroken/watercolor's rungs. Second,
derived skies were named after the *input* file stem, and every Bitterli
scene is a `scene-v4.pbrt` — spaceship's and teapot-full's skies
collided on `scene-v4-sky.exr` (the recon's one-sky-per-run imports
couldn't see it). The CLI now names derived assets after `--out`
(`cenote_pbrt::import_as`), which is also what §3's "`<name>-sky.exr`
lands beside the RON" always claimed; the corpus README's derived-sky
recipe covers regeneration.

**Rung-3 notes.** No code changes — the recon's gap map held exactly,
but the decision-7 bar earned its keep by *weighing* the gaps. Bathroom
measured 0.082 eighth-res against native pbrt — far above the gapped
expectation — and the hunt (window-region crops matching pbrt to four
decimals, a from-scratch two-quad microtest agreeing to five, blinds
and mirror bisections moving nothing) ended at the *displacement*
warnings everyone had priced as minor: the window light grazes the
bumpy foam and tile floor, the angle where bump maps darken hardest, so
the flat import runs the foam ~1.9× and the floor ~3× bright. Strip
displacement from the pbrt reference and bathroom drops to 0.045; the
rest is the ~10% Schlick-vs-exact-Fresnel surplus ajar documented
(the alpha-less rug is visually nil at this framing). Kitchen measured
0.053, and degrading pbrt by its two documented gaps (blinds
transmission flattened, displacement dropped) collapses it to 0.015 —
the gap-free baseline, nothing else diverging. Lesson recorded for the
remaining rungs: a "dropped texture" warning's cost is
lighting-geometry-dependent — the same class of gap is invisible in
one scene and the headline in the next. Side discovery, microtest-only:
pbrt orients a mesh's geometric normal to agree with authored shading
normals where cenote trusts winding — every corpus mesh so far winds
consistently, but a winding-vs-normals disagreement would silently flip
a one-sided emitter (watch at zero-day's rung, where emissive
orientation is already the crux).

**Documented-only renderer gaps** (unlock noted in each RON header when
its rung lands): anisotropic roughness, displacement, diffuse-transmission
lobe, textured coat roughness, dispersion (all unassigned material-depth
work); participating media (M8).

## 4. The ladder

Per-rung Go, one commit each, poetic subjects. Scene rungs follow the
decision-7 bar; the two-class pbrt side-by-side uses the local pbrt-v4
build (`~/Documents/pbrt-v4/build/pbrt`, the README-figure one).

| Rung | Content |
|---|---|
| 0 | **Done** (`0eb7560`): fetch.sh, sources recon, this plan, README skeleton, gitignore |
| 1 | **Done**: cornell-box, veach-mis, veach-ajar, veach-bidir; TGA decode + the UV v-flip landed with it |
| 2 | **Done**: glass-of-water, coffee, spaceship, teapot-full, water-caustic; the PFM reader, output-stem sky naming, and teapot's curated checker landed with it |
| 3 | **Done**: bathroom, kitchen — no code changes; the recon's gap map held exactly (rung-3 notes below) |
| 4 | zero-day (the ReSTIR showcase; emission-side eyeball is the rung's crux) |
| 5–10 | Heavies, one each: bmw-m6 (consider folding tests/scenes/showcase into the corpus here), crown, bistro (shape-alpha decision), kroken (ND decision), watercolor (ND), sanmiguel (redefinition fix) |
| 11 | Swap rung — **gated on M6 closed** (viewer checklist → D-152 addendum): measure veach-ajar and zero-day against brass-room/many-lights on the reuse gate + convergence harness; if they clear, migrate gates, goldens, README figures; either way the numbers land in the D-entry |
| 12 | Close: README final, full-corpus contact sheet, deferrals pointer, closing D-entry |

## 5. Mechanics

```sh
scenes/corpus/fetch.sh [name...]      # sources into scenes/corpus/sources/
cargo run --release -p cenote-cli -- import \
    scenes/corpus/sources/bitterli/<name>/scene-v4.pbrt \
    --out scenes/corpus/<name>.ron    # bootstrap only; RON is then hand-held
cargo run -p cenote-viewer -- scenes/corpus/<name>.ron
```

Re-import for diffing goes to scratch, never over the committed RON.
Derived assets (`<name>-sky.exr`) and `sources/` are gitignored; the
committed surface of `scenes/corpus/` is the RONs, README.md, and
fetch.sh.
