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
| bistro | 251 → 8 | **243 alpha-textured shapes** — the foliage cutouts; shape-alpha landed at rung 7, the re-import is near-clean (iso, maxcomponentvalue, CuZn ×2, coat thickness/albedo ×4) |
| kroken | 173 → 172 | UV transforms ×18 = planar mappings ×14 + affine ×4 (2 inert); mix/directionmix covers → mid-gray, curated to means; ND license → RON local-generated (rung 8) |
| watercolor | 184 | Same class as kroken; ND license |
| sanmiguel | fails → 1405 | **Duplicate-texture failure was a cross-kind name collision** (`Map #483`: float bump vs spectrum reflectance) — fixed by splitting the namespaces (rung 10). Post-fix warnings are near-all benign per-imagemap noise (`wrap`/`mapping "uv"`/`maxanisotropy` ×441 each — cenote defaults or a quality knob); the real gaps are displacement ×64, diffusetransmission ×7 (plant leaves), textured coat/roughness ×4. **No position-projected mappings** (unlike kroken/watercolor — every texture is `mapping "uv"`), **no mix/media** → no hand-curation |
| zero-day | 188 | **ReverseOrientation on emissive plymesh ×97** (emission side needs the eyeball test), textured coat roughness imports smooth, texture scale 0.01 dropped |

**Proposed fix candidates** (each lands only with its rung's Go, per
decision 8):

| Fix | Class | Where | Unblocks |
|---|---|---|---|
| TGA decode (extension-hinted + `tga` feature) | core, small | **landed, rung 1** | veach-ajar renders; kitchen, bathroom, and others texture correctly |
| V-flip imported/PLY UVs to sampler storage order | importer + core, small | **landed, rung 1** (found by decision 7's side-by-side) | every textured import; see below |
| PFM read for infinite-light images | importer, small | **landed, rung 2** | teapot-full imports |
| Texture namespaces split by kind (`float` vs `spectrum`) | importer, small | **landed, rung 10** — the recon reframed the proposed "last-wins" fix: sanmiguel's `Map #483` collision is a *cross-kind* reuse (a float bump and a spectrum reflectance sharing a name across two include files), which pbrt-v4 never conflates because it keeps two namespaces; the map now keys on `(kind, name)`, each slot resolves against its own kind (fallback to the other), and a genuine *same-kind* redefinition warns and takes the last (pbrt's deferred-creation semantics) | sanmiguel imports; guards every future scene against name collisions between independently authored includes |
| Shape `alpha` texture → material opacity | importer, medium | **landed, rung 7** (cutout material forks; mask channel by header probe) | bistro/sanmiguel foliage; bathroom's rug (curated in at rung 7) |
| UV affine transforms (uscale/vscale/udelta/vdelta) + imagemap `scale` | **core + schema**, medium — `TextureRef` gained an optional affine `uv` remap and value `scale`, applied at sample time (`ba83d27`) | **landed, rung 9** on watercolor's evidence (the easel drawing's remap + 28 value scales) — the transform rides the reference, not a bake, so tiling stays lossless and `scale > 1` exact; identity default keeps every golden bit-identical | kroken/watercolor books, magazines, fabrics tile correctly; scaled textures no longer warn-and-drop |
| Single-patch `bilinearmesh` shapes | importer, small | **landed, rung 9** (two triangles on the `dpdu × dpdv` side; indexed patches still skip) | watercolor's floor paint-splatter decals |
| Texture mapping modes (planar/cylindrical position-projection) | **core + schema**, medium — sampling has no position-derived UV modes | still parked: watercolor's 14 (9 cylindrical + 5 planar) all wear authored UVs instead, camera-visible on wall art and small props — a bigger feature (a second UV source), deferred past the corpus | kroken/watercolor wall pictures, cylindrical-wrapped props project correctly |

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

**Rung-4 notes — the eyeball earned a renderer fix.** Zero-day
(frame180, the README showcase frame) imported to 16559 ops, and
native-vs-native measured a dire 0.495 that is almost entirely *film*:
the scene's Film block carries a Canon EOS 100D sensor response, 5500 K
white balance, and ISO 175 — a new documented gap class (cenote renders
scene-referred radiance) — and stripping the trio from the pbrt
reference drops the score to 0.105. The residual hunt ran seven false
suspects to ground (blackbody normalization matches pbrt's 1-nit
photometric convention to three decimals; depth, convergence, clamps,
occlusion and tmax bisections all null) before a one-panel microtest
pinned the real bug: **a one-sided emitter under a negative-determinant
instance transform emitted from the wrong side.** The hit side was
always correct — the inverse-transpose normal lands on the mapped
object-front for any invertible transform — but the light records bake
world-space corners, whose cross-product normal carries the
determinant, so next-event refused the emitting side and MIS's power
heuristic (side-agnostic |cos| in the hit-side weight) kept deferring
to the strategy that never fired: ×500 dark, not ×2. Zero-day authors
nearly everything through mirrored chains, so most of its 283 lights
were wrong-sided — and the importer's 97 trap-4 warnings marked
exactly the *healthy* shapes (pbrt XORs handedness to cancel the
mirror out of its world-vertex bake; folding that into object-space
winding double-corrected). Fixed renderer-side ("An emitter keeps its
face through every mirror"): the light record carries sign(det), the
importer's flip is `ReverseOrientation` alone, and the microtest goes
0.0017 → 0.909 against pbrt's 0.910. Zero-day lands at **0.035**
eighth-res against the film-stripped reference (256 spp ≡ 1024 —
converged), with the roughness-scale/coat-texture flattening on the 30
RustMixed metals and the BASE ROOM displacement as its remaining
documented gaps. The earlier rungs are provably untouched: Bitterli
exports bake a mirror into every CTM, which the import fold cancels to
det +1 — goldens, README figures, and all eleven landed scenes verified
bit-identical, so the recorded rung-1…3 numbers stand. Rung-3's
winding-vs-normals watchpoint also closes for this scene: all 283
emissive PLYs wind with their authored normals (a future
disagreeing-mesh scene would follow winding — cenote's rule — where
pbrt follows authored normals; documented, adversarial-only so far).

**Rung-5 notes.** bmw-m6 landed, no code changes — the recon's
"near-clean" held. The one real gap is the interior LEATHER, pbrt's
`mix` material (unsupported → OpenPBR defaults at import): pbrt-v4's
semantics put `amount` on the *second* material (materials.h
`ChooseMaterial`: P(second) = amount), so amount 0.2 over
[black, white] is 80% black diffuse 0.01 + 20% white coated 0.8 — dark
seats. Curated to one blended material (base 0.168, the white's coat at
0.2 weight); worth 2× on the score — the defaults' bright cabin glows
through every window (0.059 → 0.029 clamped). Eighth-res RMSE vs native
pbrt: 0.077 raw / 0.029 display-clamped (cenote 4096 spp, pbrt 2048).
The raw-vs-clamped gap is uncorrelated firefly energy on *both* sides —
pbrt's `regularize true` is dropped at import and even its reference
speckles its glossy floor — so raw RMSE is spike-dominated and unstable
(cenote 1024→4096 spp moved raw 0.069→0.075 while clamped improved
0.035→0.030; pbrt's own 512-vs-2048 clamped self-distance is 0.009);
the clamped number is the signal. Clamped
residual = coat-Fresnel (ajar/bathroom family) + LEATHER approximation
+ spectral-vs-RGB aluminum rims. Both aniso warnings equal-axis (exact);
`iso 100` is pbrt's default (no-op) — so no film-stripping needed,
unlike zero-day. The ladder's fold happened: `tests/scenes/showcase/`
(the old tier-2 BMW fetch) is retired — fetch-showcase.sh deleted, its
gitignore entry dropped, tests/scenes/README.md tier-2 section now
points at the corpus, which uses the same commit pin.

**Rung-6 notes.** crown landed, no code changes. The film-response
class returns (iso 150, canon_eos_5d_mkiv sensor), so the honest number
rides a film-stripped reference: eighth-res RMSE 0.304 raw / 0.093
display-clamped (cenote 2048 spp, pbrt GPU 2048; pbrt's 512-vs-2048
self-distance is 0.003 — the reference is converged, unlike bmw the raw
excess is not both-sides speckle). Against native film: 0.133 clamped =
tone. The kitchen-style degradation experiment pinned the residual:
with crown's four displacement textures removed from pbrt (sapphire
bump, mitra bands), the score collapses to 0.072 raw / 0.048 clamped —
displacement is the dominant divergence (pbrt's sapphire dark and
faceted, cenote's smooth clear glass), now the third scene where it
leads (bathroom, kitchen). Three curations: (1) the ruby/sapphire
absorption media (MakeNamedMedium/MediumInterface, dropped at import)
carried onto their materials as Beer-Lambert transmission tint — exact,
not approximate, since sigma_s = 0 and cenote's transmission model *is*
Beer-Lambert; every binding of those materials carries the medium.
(2) Dispersive gem etas (3.5@200nm–3.3@900nm) import as the 1.5
fallback; curated to the visible-range mean 3.4. (3) Two texture-amount
`mix` materials (enamel-on-gold masks — a new mix flavor vs bmw's
constant amount) curated as *textured metalness*: the mask drives
metal-vs-diffuse per texel (P(second) = amount, white = gold), one
constant carries the mask-mean color blend (means 0.69 and 0.94 via
`magick identify`). Curations worth 1.5× against the degraded reference
(0.071 → 0.048 clamped). Remaining documented drops: textured-roughness
remap caveat ×2, dropped texture scale factors (0.62/1.5 reflectance,
0.1 roughness), one equal-axis aniso (exact). pbrt's `--gpu` build ran
all references (512 spp ≈ 4 min at 1000×1400, maxdepth 100).

**Rung-7 notes.** bistro landed, and the rung's importer-debt decision
resolved to *land it*: the renderer was already capable (stochastic
traversal opacity on `geometry_opacity`, per-sample-exact tests), so per
decision 8 the shape-`alpha` extension went in — a shape's alpha forks
its material into a shared cutout (`<base>-cutout-N`) carrying the mask
on `geometry_opacity`, composing under area lights (glow forks the
cutout). The mask's channel follows pbrt: an alpha-carrying image reads
A, anything else the red default (pbrt averages RGB — identical for gray
masks); the importer tells them apart by a bounded header sniff, PNG by
IHDR color type, TGA (no magic bytes) by pixel depth. 243 alpha
references became 53 cutout materials; warnings fell 251 → 8. Bathroom's
rug — the gap parked for this rung — is curated onto its RON's Rug
material. The rung also found a *tooling* bug the 23 MB RON exposed
(zero-day-class inline meshes, 3×): `from_ron`'s serde version probe
parsed the whole file through ron's skipped-value path, which re-scans
the remaining source for `..` at every number — quadratic, an hour of
parsing before the first sample. Fixed in format.rs: typed parse first,
version recovered by a bounded scan only on the error path; bistro now
parses in seconds. Vespa view (the repo's gallery shot) is the RON;
eighth-res RMSE vs native pbrt --gpu: 0.215 raw / 0.063 display-clamped
(cenote 2048 spp vs pbrt 2048; self-distance 0.003). A pinhole pair
scores the same — DOF exonerated. The dominant divergence is the
**coateddiffuse model**, worn by 123 of 132 materials: pbrt's stochastic
layered coat costs 32% of scene mean, OpenPBR's analytic coat 20%; both
sides degraded to bare diffuse and exposure-matched (film iso 110 =
pure ×1.1) agree at 0.116 raw / 0.051 clamped with means within 2%.
The coat deficit and iso surplus partially cancel, so the native
comparison is the honest headline (exposure-matching alone worsens it).
The derived sky was verified energy-exact through the octahedral →
equirect resample (total flux to 0.4%, the sun disk to 0.7%, tint
preserved). Curations: CuZn (brass) conductors ride the standard lookdev
brass F0 (the importer's copper fallback warned); `maxcomponentvalue 20`
documented as bmw's regularize class.

**Rung-8 notes.** kroken landed with the campaign's first *uncommitted*
RON, and the eyeball earned another renderer-bug find. The ND decision,
license text in hand: CC-BY-ND 2.0 §3 grants "modifications as are
technically necessary to exercise the rights in other media and formats,
but otherwise you have no rights to make Derivative Works" — and a
curated conversion (substituted materials, baked textures, one picked
camera) is a derivative, so distributing it in this repo would breach the
ND term. Resolution: `scenes/corpus/curate-kroken.py` is committed — a
content-free script that re-imports the untracked sources and re-applies
every curation — and `kroken.ron` joins the derived assets in gitignore.
The same pattern serves watercolor at rung 9. Recon's "procedural
textures" resolved to `mix`/`directionmix`/`scale`/planar classes, no
checkerboard bake needed; the 18 UV-transform warnings are 14 planar
projections + 4 affines (2 inert), all shelf props, so the core
UV-transform feature stayed parked on watercolor's evidence. Curations:
the polka-dot pillow (a `mix` material, dot mask as amount) baked to a
dots base-color texture plus its normal map; red-glass media as
approximate Beer–Lambert tint (sigma_s nonzero — pbrt in-scatters, M8);
gray'd mix/directionmix props (blanket, magazines, book covers, box,
pages) to linearized image means. Numbers, eighth-res vs native pbrt
(cenote 2048 spp GPU vs pbrt 512 CPU — pbrt's own GPU path OOMs on
kroken's displacement micro-tessellation; self-distance 0.009): **0.105
raw / 0.073 display-clamped**, cenote +8% mean. The decomposition ran
four degradation renders and three microtests: (1) coat class — pbrt's
simulated coat costs this white GI box 19% of scene mean vs OpenPBR's
analytic 4.4% (bistro's model gap, amplified by interior multi-bounce);
(2) displacement costs pbrt 5.8% (shag rug + concrete, bathroom class);
(3) **the alpha-0 invisible sun lands at exactly half strength in
cenote** — the fix candidate below; (4) an all-diffuse bisect agrees to
±3% where the real material set differs 14%, pinning the residual on
conductor/dielectric relay (the white metal shelving bounces the window
light; veach-ajar's Schlick-vs-exact class + textured-metal drops); (5)
the dropped portal is a real domain gap — pbrt's portal restricts env
light to window directions (kroken leaks ~15% env without it), cenote
lights a full dome. Blackbody normalization exonerated: an area-light
microtest matches pbrt to 3 decimals, and the derived 2×1 sky stores the
same hue ×7.5. Bare-floor and window-box env microtests match to 0.8% —
uniform-dome transport is exact.

**Rung-8 fix candidate — invisible-emitter MIS (renderer, medium).** An
emitter whose geometry BSDF rays pass through (geometry_opacity 0, the
rung-7 cutout fork) can never be hit, so the emitter-hit MIS strategy
never fires — yet NEE still discounts by the power heuristic against it.
pbrt's convention: an alpha-0 light illuminates at full strength
(microtest: pbrt alpha-0 ≡ pbrt visible to 5 digits; cenote lands at
half — scratchpad rung8/micro/*). The consistent weighting is
`powerHeuristic(neePdf, α·bsdfPdf)` with α the light's opacity — α=0
gives NEE weight 1, and partial α (bistro's lantern forks) stays
correctly balanced against the α-probability emitter-hit strategy.
Needs the light's opacity scalar on the light record (textured masks
would approximate by mean or stay document-only). Decided with a
later rung's Go, rung-4 style.

**Rung-9 notes.** watercolor — an attic art studio, the second CC-BY-ND
scene — landed as a derived asset like kroken (`curate-watercolor.py`
regenerates the RON and five bakes; the `.gitignore` block grew one
line). It is the rung that **spent the UV feature the campaign had
parked since rung 8** (`ba83d27`): recon found the scene's namesake
easel drawing wearing a pure affine remap, 28 more references carrying
value `scale`s the importer used to warn-and-drop, and three floor
paint-splatter *bilinearmesh* decals — enough camera-visible evidence
to land the core feature rather than curate around it. `TextureRef`
gained an optional affine `uv` remap and a value `scale`, both applied
at sample time (so tiling stays lossless and `scale > 1` exact against
the unorm BC formats) through a per-texture parameter table beside the
bindless images; the importer converts pbrt's uscale/vscale/udelta/
vdelta at the same v-flip boundary the UVs themselves cross
(`offset_v = 1 − vscale − vdelta`). Identity defaults keep every
existing golden bit-identical. What stayed parked: the 14 *mapping
modes* (9 cylindrical, 5 planar) that project position-derived UVs —
a second UV source, a bigger feature, deferred past the corpus; those
props wear authored UVs and small camera-visible art scales wrong.

Curations (all in the header): the near-white wall `mix` to its
constant; the walnut desk, jute rug, and concrete floor — the scene's
largest surfaces — to bakes (`watercolor-noce/carpet/concrete.exr`,
the AO-biased scales pbrt applies, folded in; the AO masks broadcast to
RGB so the multiply doesn't collapse to red); the six `mix` *materials*
(Tin 05, Case gold, the drippy tins, Spot catcher, Paper-script) to
blends of their arms, Tin 05's dirt mask becoming textured metalness
(crown's pattern, now exact since the reference carries the 1.5 scale);
the TiO2 paint-tube conductor (copper fallback) to a neutral F0; the
brush-water medium to Beer–Lambert on the cup glass; the floor splatter
decals to **pre-inverted** cutout masks (the source masks carry
`bool invert`, so the bake is `1 − mask`, spot10's 1.04 scale folded in
before inversion as pbrt applies it).

Numbers: **0.072 raw / 0.060 clamped** at eighth-res vs native pbrt
(volpath 512 spp CPU — pbrt renders this scene on the GPU too, but the
CPU reference matches the campaign's convention). Better than kroken's
0.105/0.073: watercolor's coateddiffuse family is less GI-amplified
(the room is not a white box). The signed diff is a near-uniform slight
over-brightness across the whole room — cenote's full dome delivers
more total light than pbrt's window-restricted portals — with localized
spots at the glass vases/cup (media approximation) and the abstract
wall art (the parked mapping modes). The room is lit *entirely* by two
portal'd copies of one infinite light (one per skylight; the source's
two blackbody area lights have their shapes commented out); cenote
keeps the first light and lights a full dome, so the portal domain is
the dominant gap, exactly kroken's class — the full dome admits blue
overhead sky the window portals never do, which is the cool cast in the
comparison. Degradation confirms it: a pbrt render with the portals
*removed* (full dome, `camera-1-noportal.pbrt` in the untracked
sources) diverges from stock pbrt by **0.034 clamped** — over half the
total — and cenote sits *closer* to that no-portal pbrt (0.050) than to
stock pbrt (0.060), i.e. cenote's full dome behaves like pbrt's full
dome, and the portal restriction is the single largest lever. The
residual 0.050 against the matched-domain reference is the coat model
(bistro/kroken's coateddiffuse class), the parked mapping modes, and
the media approximation.

**Rung-10 notes.** sanmiguel — the San Miguel de Allende courtyard, the
corpus's geometric heavy (70,603 ops, 242 alpha foliage shapes) — is the
first pbrt-v4 heavy to land from a **bare import with no hand-curation**:
no `mix`/`directionmix`, no participating media, and every texture uses
`mapping "uv"` (none of kroken/watercolor's position-projected modes), so
nothing needed working around. Its licence is a plain "thanks to
Guillermo M. Leal Llaguno" — the committable `permissions per repo
README` class (crown/zero-day), so the RON is committed directly, not
script-regenerated like the ND scenes.

The rung's fix reframed the recon's proposal. The import failed
`texture "Map #483" is defined twice`, and the proposed remedy was
"tolerate redefinition, last wins + warning". But the recon showed the
two `Map #483` definitions are a *different texture each*: a `float` bump
(`Fierro_A_Bump.png`) in `mesas_abajo-mat.pbrt` and a `spectrum`
reflectance (a carpet) in `mesas_arriba-mat.pbrt` — a name collision
between two independently authored include files. pbrt-v4 renders both
correctly because it keeps **float and spectrum textures in disjoint
namespaces**; naive last-wins would have mis-bound the chairs' bump to
the carpet. The fix keys the named-texture map on `(kind, name)`, each
material slot resolves against its own kind (`reflectance` → spectrum,
`alpha`/`roughness`/bump → float) with a fallback to the other namespace,
and a *genuine same-kind* redefinition still warns and takes the last —
pbrt's deferred-creation semantics, which the flattened pre-pass matches.
Identity for every existing scene (none carry a cross-kind collision), so
the corpus goldens stay bit-identical. The 242 alpha shapes → 97 cutout
materials rode rung-7's shape-alpha unchanged.

Numbers: **0.065 raw / 0.045 clamped** at eighth-res vs native pbrt
(`--gpu` 512 spp; cenote 256 spp, both downsampled 8× — sanmiguel's
`cropwindow` is dropped on import, so both render full-frame) — the
tightest corpus number so far, the coateddiffuse family far less
GI-amplified in an open-air courtyard than in kroken's white box. The
signed diff is near-uniform gray; residuals are the derived sky's slight
tint in the open sky gaps (the six-scene derived-sky class) and cyan
speckle in the tree's dappled floor light (high-frequency + the 256-vs-512
spp gap), with displacement ×64, diffusetransmission ×7 (plant leaves),
and textured coat/roughness ×4 dropped as documented classes.

**Rung-11 notes — the swap measured out, and the gate design held.** M6 closed
first: the user's viewer checklist (orbit warm-start, re-converge on hold, the
8192-frame park on a quiet desktop — §4g decision 6) passed on the brass-room
flagship, landing the D-152 addendum. That opened the one gate the swap was
waiting on. Then decision 6's hypothesis — veach-ajar takes `indirect_glossy`'s
GI-gate role, zero-day succeeds `many_lights` — was *measured* on the reuse-gate
protocol (`convergence.rs::assert_reuse_gate` replayed verbatim: 128², a deep
ReSTIR reference, brute + ReSTIR at 8/32 spp), and it did not survive contact
with the numbers.

**zero-day fails outright.** Despite 283 emitters, at 8 spp ReSTIR carries *more*
error than brute force (0.44–0.59× — a ratio below one, so reuse is a net loss at
the low budget); it recovers to 1.51× only at 32 spp. It is a mixed direct/
indirect architectural interior, not the starved-DI regime `many_lights` distills
(256 emitters over a cluster of occluders, where a single next-event draw is
mostly wasted), so the resampling has no rare good sample to rescue at 8 spp.

**veach-ajar is marginal, not the slam-dunk a noisy reference first faked.** A
256-spp reference made it look like a 22× win; a converged 4096-spp reference
(the 256 was itself unconverged on this brutal scene) collapsed that to the honest
~1.40× raw / **1.14× clamped at 8 spp — under the gate's 1.3× floor** — clearing
only at 32 spp (1.53× / 1.36×). veach-ajar's image is mostly directly-and-once-lit
with a small genuinely-hard-indirect fraction; the incumbent `indirect_glossy`
(emitter faced away, black environment, everything lit *only* through the panel's
glossy bounce) is pure hard-GI and wins a robust ~2×. A gate that passes by 1.14×
is a flaky gate — one driver's float reordering from red.

So the swap was **held**: the purpose-built synthetic gates are the stronger,
more robust reuse stressors, and the literature scenes stay the gallery/benchmark
scenes they already are (veach-ajar rung 1, zero-day rung 4). No harness, golden,
or README-figure change. This is the "either way the numbers land in the D-entry"
outcome decision 6 anticipated, and it *validates* M6's gate design rather than
mutating it — the measurement, not a preference, made the call. (The one-off
measurement harness lived in `cenote-cli`, where the RON loader and GPU meet, and
was removed after; a real migration would have keyed on `cenote::format::load`,
which crate `cenote`'s own tests can call without the circular `cenote-pbrt` dep.)

**Documented-only renderer gaps** (unlock noted in each RON header when
its rung lands): anisotropic roughness, displacement, diffuse-transmission
lobe, textured coat roughness, dispersion (all unassigned material-depth
work); film/sensor response (sensor, white balance, ISO — zero-day;
bistro's iso-only ×1.1); mix materials (bmw-m6 LEATHER, curated to a
hand-blend; crown's texture-amount masks, curated to textured metalness;
kroken's pillow, baked); firefly regularization (integrator `regularize`
— convergence class, not bias; bmw-m6; bistro's `maxcomponentvalue`);
the coateddiffuse layered-coat depth (pbrt simulates, OpenPBR
approximates — bistro's dominant divergence, bathroom/ajar's ~10% class;
GI-amplified to 19-vs-4.4% in kroken's white box); infinite-light
portals (kroken/watercolor — a sampling aid pbrt-side *and* an
emission-domain restriction; watercolor's dominant gap, the whole room
lit through two portal'd skylights); texture *mapping modes*
(position-projected planar/cylindrical UVs — a second UV source, still
parked after rung 9 landed the affine-remap half; kroken/watercolor
wall art and props); participating media (M8).

**Rung-12 notes — the campaign closes.** All 19 scenes are accounted for:
18 render as first-class RONs, and `volumetric-caustic` stays a documented
placeholder until M8 volumes. The close touched three tracked documents and
produced one untracked artifact. (1) The **corpus README** gained a
completion header (the two renderer/importer fixes the eyeball drove, versus
the rest that are provenance-and-curation) and a fetch→render snippet. (2)
The **top-level README** gained a "The research corpus" subsection under
*Next to pbrt-v4* — the corpus is the extended form of that section's
importer-fidelity story — and a `scenes/corpus/` repo-map row. (3) The **one
deferrals pointer** (decision 5) now sits in `deferrals.md`'s preamble: the
corpus is where those deferrals meet real scenes, and each RON header plus
§3's gap map is where a "first corpus scene that…" trigger is checked. (4) A
**send-only 18-scene contact sheet** was rendered fresh — every landed scene
at a thumbnail resolution through the tm tool's `aces` mode (cenote's own
look, 0 stops), montaged with labels — and sent to the user, not committed:
the send-only channel is exactly what keeps the two CC-BY-ND scenes' rendered
images out of the repo while still letting the user see the whole corpus at
once. The **closing D-entry** is drafted for the user to place (decisions.md
is append-only and its entries are the user's call), carrying the campaign's
arc: the two-class bar, the importer/renderer fixes each eyeball earned (TGA
decode, the UV v-flip, the PFM sky reader, output-stem sky naming, the
float/spectrum texture-namespace split, shape-alpha cutouts, the affine UV
remap + value scale, single-patch bilinearmesh, and the mirrored-emitter
render fix), the ND-as-script-regenerated pattern, the swap held (D-153), and
the documented gap classes that outlived the campaign.

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
| 4 | **Done**: zero-day — the emission-side eyeball found and fixed the mirrored-emitter renderer bug; film response documented as a new gap class (rung-4 notes below) |
| 5 | **Done**: bmw-m6 — no code changes; LEATHER mix curated to a hand-blend, tests/scenes/showcase folded into the corpus (rung-5 notes below) |
| 6 | **Done**: crown — no code changes; gem media curated as exact Beer–Lambert tint, dispersion at the mean IOR, mask mixes as textured metalness; displacement proven dominant by degradation (rung-6 notes below) |
| 7 | **Done**: bistro — shape-alpha landed (243 masks → 53 cutout forks), the RON version-probe parse fix landed with it, bathroom's rug curated in; coateddiffuse model proven the dominant divergence by two-sided degradation (rung-7 notes below) |
| 8 | **Done**: kroken — the ND decision resolved to **RON-as-derived-asset**: CC-BY-ND 2.0 grants format shifts but not derivative distribution, so `curate-kroken.py` (committed, content-free) regenerates the curated RON locally; pillow mix baked to a dots texture, red-glass media to Beer–Lambert tint; UV-transform core feature parked on watercolor's evidence; the divergence decomposition found the **invisible-emitter MIS bug** (alpha-0 sun at half strength — fix candidate, rung-8 notes below) |
| 9 | **Done**: watercolor — second ND scene (`curate-watercolor.py` regenerates the RON + five bakes locally, kroken's pattern); the rung **landed the parked UV feature** (`ba83d27`: affine `uv` remap + value `scale` on `TextureRef`, single-patch bilinearmesh) on this scene's evidence; mix textures curated to means/bakes, water medium to Beer–Lambert, splatter decals to pre-inverted cutout masks; portal domain proven dominant by degradation (rung-9 notes below) |
| 10 | **Done**: sanmiguel — the duplicate-texture failure was a *cross-kind* name collision, fixed by splitting the float/spectrum texture namespaces (pbrt-v4 semantics); the scene then imported and rendered from a **bare import, no hand-curation** (0.065 raw / 0.045 clamped, the tightest corpus number yet); 242 alpha foliage shapes → 97 cutout materials rode rung-7 shape-alpha (rung-10 notes below) |
| 11 | **Done**: swap rung — M6 closed on the viewer checklist (D-152 addendum), then veach-ajar and zero-day were measured against the reuse gate. **Neither cleared it, so the swap was held**: zero-day loses to brute force at 8 spp, veach-ajar clears only marginally (1.14× clamped, under the 1.3× floor). The in-code `many_lights` + `indirect_glossy` gates stay; the numbers landed in the D-entry (rung-11 notes below) |
| 12 | **Done**: close — corpus + top-level READMEs finalized, a send-only 18-scene contact sheet rendered through cenote's own look, the one deferrals pointer added, closing D-entry drafted for placement (rung-12 notes below) |

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
fetch.sh. The CC-BY-ND scenes are the exception: their RONs are derived
assets too (gitignored), regenerated by a committed `curate-<name>.py`
that re-imports and re-applies the curations — the script, carrying no
scene content, is what the repo tracks.
