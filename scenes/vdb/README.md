# Volume assets

`./fetch.sh` materializes the sources into `sources/`, which stays
untracked — see the licensing notes below. `cenote-vdb-prep` caches
`.nvdb` conversions beside each `.vdb` (also untracked).

| asset | grids | licence | role |
|---|---|---|---|
| bunny_cloud | density | CC-BY 4.0 (openvdb.org) | sparse cloud; the unbiasedness oracle scene |
| smoke2 | density | CC-BY 4.0 (openvdb.org) | plume |
| explosion | density, temperature | CC-BY 4.0 (openvdb.org) | the temperature/blackbody emission scene |
| wdas_cloud | density (5 resolutions) | Disney research/**non-commercial** | production-scale stress |

wdas_cloud's licence withholds commercial use and redistribution: like
the kroken sources, nothing under `sources/` may be committed or passed
on from here.

`explosion.ron` is the blackbody-emission scene: its asset's
temperature field mapped to Kelvin (`temperature_scale`/`_offset`) and
read through the blackbody table. Its emission scale looks enormous
because it carries two small physical numbers — a 6500 K body is 1, and
emission rides on absorption, which thin smoke sets at a hundredth. Scene
load prints where the mapping lands, peak Kelvin and the radiance a thick
core settles at, so the number can be read off rather than dialled in.

`bunny-cloud-oracle.ron` is the tracked unbiasedness oracle: bunny_cloud
under a constant white sky, gray coefficients so the working-space
conversion cancels against pbrt-v4's volpath render of the same setup
(`bunny-pbrt.pbrt` beside it — mind pbrt's left-handed LookAt: compare
against the x-flopped frame). `bunny-cloud.ron` is the lookdev port of
pbrt-v4-scenes' bunny-cloud (HDR sky, coated ground).
