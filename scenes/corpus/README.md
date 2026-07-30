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

Two scenes light themselves through a **derived sky** beside their RON
(`spaceship-sky.exr`, `teapot-full-sky.exr`) — import-generated,
gitignored like the sources. To (re)generate one, re-run the bootstrap
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
| bathroom | Bitterli | CC0 (Mareck) | landed (rung 3) | Floor/Foam displacement dropped — grazing window light makes this the dominant divergence (foam ~1.9×); rug's alpha cutout renders solid (shape-alpha, bistro rung); equal-axis aniso exact |
| kitchen | Bitterli | CC BY 3.0 (Jay-Artist) | landed (rung 3) | window Blinds' diffuse transmission flattened to opaque; Cushion/BreadBin displacement — together the whole divergence (0.053→0.015 with pbrt degraded to match); equal-axis aniso exact |
| zero-day | pbrt-v4-scenes | Beeple's release | pending (rung 4) | emissive orientation eyeball; textured coat roughness |
| bmw-m6 | pbrt-v4-scenes | CC0 | pending (rung 5) | near-clean |
| crown | pbrt-v4-scenes | per repo README | pending (rung 6) | dispersion → 1.5; gem media (M8) |
| bistro | pbrt-v4-scenes | CC-BY 4.0 | pending (rung 7) | shape-alpha foliage (importer-debt decision) |
| kroken | pbrt-v4-scenes | CC-BY-**ND** 2.0 | pending (rung 8) | UV transforms; procedural textures; ND commit decision |
| watercolor | pbrt-v4-scenes | CC-BY-**ND** 2.0 | pending (rung 9) | UV transforms; procedural textures; ND commit decision |
| sanmiguel | pbrt-v4-scenes | per repo README | pending (rung 10) | duplicate-texture import fix (proposed); alpha foliage |
