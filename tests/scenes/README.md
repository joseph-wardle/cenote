# The pbrt regression corpus

Three small pbrt-v4 scenes, vendored whole so CI can import → render →
FLIP-compare hermetically (the harness is
`crates/cenote-pbrt/tests/corpus.rs`; goldens live beside it). Every
scene is **CC0** — created and published by [Benedikt
Bitterli](https://benedikt-bitterli.me/resources/) — and each directory
ships the license text it was distributed with. Crediting the artist is
encouraged, so: thank you, Benedikt.

| Scene | Exercises |
|---|---|
| `cornell-box/` | Inline triangle meshes, named diffuse materials, an area light, a *reflective* camera transform (the mirrored-image trap) |
| `veach-mis/` | Conductors with RGB `eta`/`k` and `remaproughness false`, sphere emitters across four orders of luminance — the MIS stress scene |
| `teapot-full/` | Binary PLY meshes (normals + UVs), dielectrics, an imagemap texture, an equal-area octahedral sky under a rotated `CTM` |

## Provenance and modifications

The scenes are Bitterli's **pbrt-v3** exports, hand-converted to pbrt-v4
(pbrt-v4 renamed its materials and retired some spellings). The full
list of changes:

- `"matte"` → `"diffuse"`, with `Kd` → `reflectance`; `"metal"` →
  `"conductor"` (its `eta`/`k`/`*roughness` parameters carry over);
  `"glass"` → `"dielectric"`, with `index` → `eta`.
- `Film "image"` → `Film "rgb"`.
- teapot-full: the procedural `"checkerboard"` texture (20 `uscale` ×
  20 `vscale` cells of two colors) is pre-tiled into
  `textures/checker.png`, sRGB-encoded, since cenote imports image
  textures only; the lat-long `envmap.pfm` is resampled into the square
  equal-area octahedral layout pbrt-v4 requires (exactly what pbrt's
  `imgtool makeequiarea` does) as `textures/envmap-octahedral.exr`,
  512², half float. The Tungsten reference renders (`TungstenRender.*`
  in Bitterli's zips) are not vendored — see his site for them.

Expected import warnings, all by design: `veach-mis`'s equal-axis
`uroughness`/`vroughness` (imported as their average — exact here), and
`teapot-full`'s tea medium (`MakeNamedMedium`/`MediumInterface` are not
supported yet, so the tea renders as colorless glass). The corpus area
lights are one-sided in pbrt, which cenote emitters now match, so they
import clean; a `twosided` light would warn that its back faces stay dark.

## The comparison caveat

pbrt renders **spectrally**; cenote renders **RGB in ACEScg**, with its
own photometric conventions. Corpus goldens are therefore *cenote's own
renders* — they pin the importer + renderer against regression, not
against pbrt pixel-for-pixel. For manual side-by-side comparisons, the
reference implementation is pbrt-v4 at commit
[`5f7a606`](https://github.com/mmp/pbrt-v4/commit/5f7a606806a4ac7b939131ded9d7a30ebd02416e)
(the commit this importer's semantics were verified against, 2026-07-10);
expect perceptual agreement in layout, materials, and lighting, not
matching pixel values.

## Tier 2: showcase scenes

Heavier scenes stay out of the repo. `fetch-showcase.sh` pulls the
BMW M6 (CC0, ~72 MB with its sky) from
[pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes) at a pinned
commit — git's content addressing is the checksum — into
`tests/scenes/showcase/`, which is gitignored.
