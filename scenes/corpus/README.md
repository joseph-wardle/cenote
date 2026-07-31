# The research-scene corpus

The scenes the rendering literature measures on, as first-class cenote
`.ron` files — curated by hand after a one-time `cenote-cli import`
bootstrap, each RON's header telling its own story: provenance, license,
and every knowing degradation with the feature that unlocks it. The
campaign plan (and the full rung-0 gap map) is `docs/corpus-plan.md`.

Sources are **not** in git (~8.5 GB): `./fetch.sh` materializes them into
`sources/` — Bitterli zips whole, pbrt-v4-scenes as one sparse partial
clone pinned at `30cf4a0` (content addressing is the checksum). A fresh
clone renders any landed scene after `./fetch.sh <name>`.

Seven scenes light themselves through a **derived sky** beside their RON
(`spaceship-sky.exr`, `teapot-full-sky.exr`, `bmw-m6-sky.exr`,
`bistro-sky.exr`, `kroken-sky.exr`, `watercolor-sky.exr`,
`sanmiguel-sky.exr`) —
import-generated, gitignored like the sources. To (re)generate one,
re-run the bootstrap
import into scratch — never over the curated RON — and keep the sky it
writes beside its `--out`:

```sh
cargo run --release -p cenote-cli -- import \
    sources/bitterli/<name>/scene-v4.pbrt --out /tmp/<name>.ron
mv /tmp/<name>-sky.exr scenes/corpus/
```

| Scene | Source | License | Status | Gaps (unlock) |
|---|---|---|---|---|
| cornell-box | Bitterli | CC0 | landed (rung 1) | none |
| veach-mis | Bitterli | CC0 | landed (rung 1) | none — equal-axis aniso averages exactly |
| veach-ajar | Bitterli | CC0 | landed (rung 1) | none — TGA decode + the UV v-flip landed with this rung; equal-axis aniso exact |
| veach-bidir | Bitterli | CC0 | landed (rung 1) | none — equal-axis aniso exact; native oracle is bdpt |
| glass-of-water | Bitterli | CC0 (aXel) | landed (rung 2) | none — equal-axis aniso exact; native oracle is bdpt |
| coffee | Bitterli | CC BY 3.0 (cekuhnen) | landed (rung 2) | none — equal-axis aniso exact |
| spaceship | Bitterli | CC0 (thecali) | landed (rung 2) | none — equal-axis aniso exact |
| teapot-full | Bitterli | CC0 | landed (rung 2) | tea medium (M8 volumes); procedural checker curated to the CI-baked PNG; PFM sky landed with this rung |
| water-caustic | Bitterli | CC0 | landed (rung 2) | none — native oracle is SPPM; unidirectional PT resolves the caustic slowly (spp, not bias) |
| volumetric-caustic | Bitterli | CC0 | placeholder | participating media (M8 volumes) |
| bathroom | Bitterli | CC0 (Mareck) | landed (rung 3) | Floor/Foam displacement dropped — grazing window light makes this the dominant divergence (foam ~1.9×); rug cutout curated in at rung 7 (shape-alpha landed); equal-axis aniso exact |
| kitchen | Bitterli | CC BY 3.0 (Jay-Artist) | landed (rung 3) | window Blinds' diffuse transmission flattened to opaque; Cushion/BreadBin displacement — together the whole divergence (0.053→0.015 with pbrt degraded to match); equal-axis aniso exact |
| zero-day | pbrt-v4-scenes | Beeple's release | landed (rung 4) | Film sensor/whitebalance/ISO dropped — native 0.495 is tone, 0.035 against a film-stripped reference; ×0.01 roughness-scale + coat-roughness textures flattened on the 30 RustMixed metals; BASE ROOM displacement; the rung found and fixed the mirrored-emitter renderer bug |
| bmw-m6 | pbrt-v4-scenes | CC0 (tyrant monkey) | landed (rung 5) | interior LEATHER `mix` curated to a hand-blend (80/20 per pbrt-v4 mix semantics); `regularize` dropped — cenote keeps its dielectric-glass fireflies (convergence, not bias); equal-axis aniso exact |
| crown | pbrt-v4-scenes | per repo README (Martin Lubich) | landed (rung 6) | displacement dropped ×4 — the dominant divergence (sapphire bump, mitra bands); gem media curated as exact Beer–Lambert tint; dispersion carried at mean IOR 3.4; mask `mix`es as textured metalness; film iso/sensor dropped (zero-day class) |
| bistro | pbrt-v4-scenes | CC BY 4.0 (Amazon Lumberyard, ORCA) | landed (rung 7) | shape-alpha foliage LANDED with this rung (243 masks → 53 cutout materials); coateddiffuse model is the dominant divergence — pbrt's simulated coat darkens ~32% vs OpenPBR's 20% (both-degraded pair agrees to 0.051 clamped); film iso ×1.1 + maxcomponentvalue dropped; CuZn curated to brass F0 |
| kroken | pbrt-v4-scenes | CC-BY-ND 2.0 (Angelo Ferretti) | landed (rung 8) — **RON not committed**: the ND term withholds derivative distribution, so `curate-kroken.py` regenerates `kroken.ron` locally from the untracked sources | planar texture mappings ×14 + UV affine ×4 sample authored UVs (books/magazines; texture mapping modes); displacement + diffusetransmission dropped; red-glass media curated to approximate Beer–Lambert tint (M8 volumes); pillow `mix` baked to a dots texture; env portal dropped (full dome vs pbrt's window-restricted domain); **alpha-0 invisible sun lands at half strength — renderer MIS fix candidate (rung-8 notes)**; coateddiffuse model class (bistro) |
| watercolor | pbrt-v4-scenes | CC-BY-ND 2.0 (Angelo Ferretti) | landed (rung 9) — **RON not committed**: like kroken, `curate-watercolor.py` regenerates `watercolor.ron` (+ five bakes) locally | the rung that **landed the UV affine feature** (the easel drawing's remap + 28 value scales tile correctly now); mapping modes ×14 (cylindrical/planar position projection) still sample authored UVs — wall art scales wrong; six `mix` materials curated to arm-blends, walnut/rug/concrete to AO bakes, water medium to Beer–Lambert (M8), splatter decals to pre-inverted cutout masks; env portals dropped (full dome vs two window-restricted skylights — the dominant gap, kroken's class); diffusetransmission + displacement dropped; coateddiffuse model class |
| sanmiguel | pbrt-v4-scenes | per repo README (Guillermo M. Leal Llaguno) | landed (rung 10) | the rung that **fixed the duplicate-texture import** — pbrt keeps `float` and `spectrum` textures in separate namespaces, so `Map #483` (a float bump in one include, a spectrum reflectance in another) no longer collides; the first pbrt-v4 heavy to land from a **bare import, no hand-curation** (no `mix`/media to work around); 242 alpha foliage shapes → 97 cutout materials via rung-7 shape-alpha; displacement ×64 + diffusetransmission ×7 (plant leaves) + textured coat/roughness ×4 dropped (documented classes); env sky derived like the other six; film cropwindow + motion-blur shutter dropped, so the comparison renders full-frame on both sides |
