# The one list of gate scenes, sourced by every script that walks a set
# (corpus-identity.sh, perf-gates.sh, benchmark.sh) — a scene added to the
# corpus joins its gates here, in one place, instead of in each script's
# private copy. Entries are "<path-under-repo-root>:<label>"; scripts
# prefix $root themselves.

# The byte-identity sweep: every deterministic renderable scene the repo
# gates on. Rendered at capture settings and byte-compared against
# baselines/corpus — a one-byte difference is a defect to bisect.
# `kroken` and `watercolor` need their curate scripts run first; every
# corpus scene needs `scenes/corpus/fetch.sh`; the vdb scenes need
# `scenes/vdb/fetch.sh`.
identity_scenes=(
  "scenes/corpus/bathroom.ron:bathroom"
  "scenes/corpus/bistro.ron:bistro"
  "scenes/corpus/bmw-m6.ron:bmw-m6"
  "scenes/brass-room.ron:brass-room"
  "scenes/corpus/coffee.ron:coffee"
  "scenes/corpus/cornell-box.ron:cornell-box"
  "scenes/corpus/crown.ron:crown"
  "scenes/corpus/glass-of-water.ron:glass-of-water"
  "scenes/corpus/head.ron:head"
  "scenes/corpus/kitchen.ron:kitchen"
  "scenes/corpus/kroken.ron:kroken"
  "scenes/corpus/sanmiguel.ron:sanmiguel"
  "scenes/corpus/spaceship.ron:spaceship"
  "scenes/corpus/teapot-full.ron:teapot-full"
  "scenes/corpus/veach-ajar.ron:veach-ajar"
  "scenes/corpus/veach-bidir.ron:veach-bidir"
  "scenes/corpus/veach-mis.ron:veach-mis"
  "scenes/corpus/water-caustic.ron:water-caustic"
  "scenes/corpus/watercolor.ron:watercolor"
  "scenes/corpus/zero-day.ron:zero-day"
  # The feature scenes the sweep was blind to before: homogeneous and
  # priority-nested interiors, the subsurface walk, curves, and the two
  # heterogeneous-volume scenes. A byte-level regression in any of those
  # subsystems is invisible without them.
  "scenes/milk-glass.ron:milk-glass"
  "scenes/nested-glass.ron:nested-glass"
  "scenes/sss/teapot-wax-walk.ron:sss-wax-walk"
  "scenes/curves.ron:curves"
  "scenes/vdb/bunny-cloud-oracle.ron:bunny-cloud-oracle"
  "scenes/vdb/explosion.ron:explosion"
)

# The shading perf-gate set (2560x1440): drivers first (shading-bound),
# then the overhead controls, then the transmission/coat coverage the
# drivers lack — ending with the one scene that authors a nesting
# priority and has no media at all, the only place the volume stage's
# cost to a media-free scene can be seen.
shading_scenes=(
  "scenes/corpus/bistro.ron:bistro"
  "scenes/corpus/sanmiguel.ron:sanmiguel"
  "scenes/corpus/zero-day.ron:zero-day"
  "scenes/corpus/cornell-box.ron:cornell-box"
  "scenes/brass-room.ron:brass-room"
  "scenes/corpus/glass-of-water.ron:glass-of-water"
  "scenes/corpus/spaceship.ron:spaceship"
  "scenes/nested-glass.ron:nested-glass"
  "scenes/milk-glass.ron:milk-glass"
)

# The subsurface-walk gate set: the three optical-depth tiers share
# geometry and differ only in sigma, so a number that moves on one and
# not the others is the medium's doing rather than the scene's. Last is
# the head with its albedo map, the only driver whose interior is
# resolved per entry point instead of read from the table — watched for
# its own drift, not diffed against `head`.
walk_scenes=(
  "scenes/sss/teapot-wax-walk.ron:wax"
  "scenes/sss/teapot-marble-walk.ron:marble"
  "scenes/sss/teapot-skin-walk.ron:skin"
  "scenes/sss/head-walk.ron:head"
  "scenes/sss/head-mapped-walk.ron:head-mapped"
)

# The benchmark set: each scene probes one cost regime.
benchmark_scenes=(
  "scenes/corpus/cornell-box.ron:cornell-box" # cheapest frame — dispatch-bound
  "scenes/brass-room.ron:brass-room"          # heavy indirect GI
  "scenes/many-lights.ron:many-lights"        # NEE-bound
  "scenes/corpus/bistro.ron:bistro"           # production exterior at scale
  "scenes/corpus/zero-day.ron:zero-day"       # 283 lights in a dark interior
)
