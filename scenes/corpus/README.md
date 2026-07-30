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

| Scene | Source | License | Status | Gaps (unlock) |
|---|---|---|---|---|
| cornell-box | Bitterli | CC0 | pending (rung 1) | none |
| veach-mis | Bitterli | CC0 | pending (rung 1) | none (equal-axis aniso averages exactly) |
| veach-ajar | Bitterli | CC0 | pending (rung 1) | TGA decode (fix proposed); aniso averaged |
| veach-bidir | Bitterli | CC0 | pending (rung 1) | aniso averaged |
| glass-of-water | Bitterli | CC0 | pending (rung 2) | aniso averaged |
| coffee | Bitterli | CC BY 3.0 (cekuhnen) | pending (rung 2) | aniso averaged |
| spaceship | Bitterli | CC0 (thecali) | pending (rung 2) | aniso averaged |
| teapot-full | Bitterli | CC0 | pending (rung 2) | PFM sky (fix proposed); tea medium (M8 volumes) |
| water-caustic | Bitterli | CC0 | pending (rung 2) | none |
| volumetric-caustic | Bitterli | CC0 | placeholder | participating media (M8 volumes) |
| bathroom | Bitterli | CC0 (Mareck) | pending (rung 3) | TGA; aniso; displacement; one alpha shape |
| kitchen | Bitterli | CC BY 3.0 (Jay-Artist) | pending (rung 3) | TGA; aniso; displacement; diffuse transmission |
| zero-day | pbrt-v4-scenes | Beeple's release | pending (rung 4) | emissive orientation eyeball; textured coat roughness |
| bmw-m6 | pbrt-v4-scenes | CC0 | pending (rung 5) | near-clean |
| crown | pbrt-v4-scenes | per repo README | pending (rung 6) | dispersion → 1.5; gem media (M8) |
| bistro | pbrt-v4-scenes | CC-BY 4.0 | pending (rung 7) | shape-alpha foliage (importer-debt decision) |
| kroken | pbrt-v4-scenes | CC-BY-**ND** 2.0 | pending (rung 8) | UV transforms; procedural textures; ND commit decision |
| watercolor | pbrt-v4-scenes | CC-BY-**ND** 2.0 | pending (rung 9) | UV transforms; procedural textures; ND commit decision |
| sanmiguel | pbrt-v4-scenes | per repo README | pending (rung 10) | duplicate-texture import fix (proposed); alpha foliage |
