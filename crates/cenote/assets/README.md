# Bundled assets

## `kloofendal_puresky.exr`

The demo scene's environment: [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
by Greg Zaal (original) and Jarod Guest (sky edits), from Poly Haven,
licensed [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

The 2k (2048×1024) version, re-encoded from Poly Haven's 20 MB f32+PIZ to
11 MB with PXR24 (float mantissas truncated to 24 bits — relative error
below 0.002%, range fully preserved). Not f16: the unclipped sun peaks at
~73,600, past f16's 65,504 max, and clamping it there would cost a
measured 2% of the sun's energy — the sun is exactly what makes this a
real HDRI importance-sampling test and not just a backdrop.
