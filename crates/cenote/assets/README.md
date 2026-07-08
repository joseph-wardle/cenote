# Bundled assets

## `kloofendal_puresky.exr`

The demo scene's environment: [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
by Greg Zaal (original) and Jarod Guest (sky edits), from Poly Haven,
licensed [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

The 4k (4096×2048) version, re-encoded from Poly Haven's 76 MB f32+PIZ to
43 MB with PXR24 (float mantissas truncated to 24 bits — relative error
below 0.002%, range fully preserved). Not f16: the unclipped sun peaks at
~75,400, past f16's 65,504 max, and clamping it there would cost about 2%
of the sun's energy — the sun is exactly what makes this a real HDRI
importance-sampling test and not just a backdrop. Not 8k: that lands past
GitHub's 100 MB file limit, and its 32-bit GPU image alone would be half
a gigabyte.

Too large to embed — [`Scene::demo`] reads it from this directory at
runtime, via the crate root.
