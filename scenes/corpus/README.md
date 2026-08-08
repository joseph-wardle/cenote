# The research-scene corpus

The scenes the rendering literature measures on, as first-class cenote
`.ron` files — curated by hand after a one-time `cenote-cli import`
bootstrap, each RON's header telling its own story: provenance, license,
and every knowing degradation with the feature that unlocks it. All 19 are
here; 18 render, and `volumetric-caustic` waits as a documented placeholder
until volumes exist.

Sources are **not** in git (~8.5 GB): `./fetch.sh` materializes them into
`sources/` — Bitterli zips whole, pbrt-v4-scenes as one sparse partial
clone pinned at `30cf4a0` (content addressing is the checksum). A fresh
clone renders any landed scene after `./fetch.sh <name>`:

```sh
./fetch.sh veach-ajar
cargo run --release -p cenote-viewer -- scenes/corpus/veach-ajar.ron   # or
cargo run --release -p cenote-cli -- render scenes/corpus/veach-ajar.ron --out ajar.exr
```

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

Every scene's own RON header repeats its provenance and gaps; this table is the
index. Anisotropic roughness is imported as the axis average, which is exact for
the equal-axis parameters every corpus scene uses, so it is not a gap anywhere.

| Scene | Source | License | Gaps (unlock) |
|---|---|---|---|
| cornell-box | Bitterli | CC0 | none |
| veach-mis | Bitterli | CC0 | none |
| veach-ajar | Bitterli | CC0 | none |
| veach-bidir | Bitterli | CC0 | none — native oracle is bdpt |
| glass-of-water | Bitterli | CC0 (aXel) | none — native oracle is bdpt |
| coffee | Bitterli | CC BY 3.0 (cekuhnen) | none |
| spaceship | Bitterli | CC0 (thecali) | none |
| teapot-full | Bitterli | CC0 | tea medium (M8 volumes); procedural checker curated to the CI-baked PNG |
| water-caustic | Bitterli | CC0 | none — native oracle is SPPM, so unidirectional PT resolves the caustic slowly (spp, not bias) |
| volumetric-caustic | Bitterli | CC0 | **placeholder, does not render** — participating media (M8 volumes) |
| bathroom | Bitterli | CC0 (Mareck) | Floor/Foam displacement dropped — grazing window light makes this the dominant divergence (foam ~1.9×) |
| kitchen | Bitterli | CC BY 3.0 (Jay-Artist) | window Blinds' diffuse transmission flattened to opaque; Cushion/BreadBin displacement — together the whole divergence (0.053→0.015 with pbrt degraded to match) |
| zero-day | pbrt-v4-scenes | Beeple's release | film sensor/whitebalance/ISO dropped — native 0.495 is tone, 0.035 against a film-stripped reference; ×0.01 roughness-scale + coat-roughness textures flattened on the 30 RustMixed metals; BASE ROOM displacement |
| bmw-m6 | pbrt-v4-scenes | CC0 (tyrant monkey) | interior LEATHER `mix` curated to a hand-blend (80/20 per pbrt-v4 mix semantics); `regularize` dropped, so dielectric-glass fireflies stay (convergence, not bias) |
| crown | pbrt-v4-scenes | per repo README (Martin Lubich) | displacement dropped ×4 — the dominant divergence (sapphire bump, mitra bands); gem media curated as exact Beer–Lambert tint; dispersion carried at mean IOR 3.4; mask `mix`es as textured metalness; film iso/sensor dropped (zero-day class) |
| bistro | pbrt-v4-scenes | CC BY 4.0 (Amazon Lumberyard, ORCA) | shape-alpha foliage (243 masks → 53 cutout materials); coateddiffuse is the dominant divergence — pbrt's simulated coat darkens ~32% vs OpenPBR's 20% (both-degraded pair agrees to 0.051 clamped); film iso ×1.1 + maxcomponentvalue dropped; CuZn curated to brass F0 |
| kroken | pbrt-v4-scenes | CC-BY-ND 2.0 (Angelo Ferretti) — **RON not committed**, `curate-kroken.py` regenerates it locally | planar texture mappings ×14 + UV affine ×4 sample authored UVs (books/magazines; texture mapping modes); displacement + diffusetransmission dropped; red-glass media carry the source σ exactly (σ_t (0.8, 10, 10), σ_s (0.4, 5, 5), isotropic); pillow `mix` baked to a dots texture; env portal dropped (full dome vs pbrt's window-restricted domain); **alpha-0 invisible sun lands at half strength — open renderer bug, see the TODO in `shaders/nee.slang`**; coateddiffuse class (bistro) |
| watercolor | pbrt-v4-scenes | CC-BY-ND 2.0 (Angelo Ferretti) — **RON not committed**, `curate-watercolor.py` regenerates it (+ five bakes) locally | mapping modes ×14 (cylindrical/planar position projection) sample authored UVs, so wall art scales wrong; six `mix` materials curated to arm-blends, walnut/rug/concrete to AO bakes, water medium carries the source σ exactly (σ_t (0.48, 0.88, 0.4), σ_s (0.24, 0.44, 0.2), isotropic), splatter decals to pre-inverted cutout masks; env portals dropped (full dome vs two window-restricted skylights — the dominant gap, kroken's class); diffusetransmission + displacement dropped; coateddiffuse class |
| sanmiguel | pbrt-v4-scenes | per repo README (Guillermo M. Leal Llaguno) | 242 alpha foliage shapes → 97 cutout materials via shape-alpha; displacement ×64 + diffusetransmission ×7 (plant leaves) + textured coat/roughness ×4 dropped (documented classes); env sky derived like the other six; film cropwindow + motion-blur shutter dropped, so the comparison renders full-frame on both sides |
