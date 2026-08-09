"""The one copy of the material schema's field order, as the importer emits
it — shared by the curate-*.py scripts, whose text patching anchors each
replacement on the following field. Must track `Material` in
crates/cenote/src/scene/description.rs; a missing name silently drops any
curated value for it.
"""

FIELD_ORDER = [
    "base_color", "base_diffuse_roughness", "base_metalness", "specular_weight",
    "specular_roughness", "specular_ior", "transmission_weight",
    "transmission_color", "transmission_depth", "transmission_scatter",
    "transmission_scatter_anisotropy", "subsurface_weight", "subsurface_color",
    "subsurface_radius", "subsurface_radius_scale",
    "subsurface_scatter_anisotropy", "coat_weight", "coat_color",
    "coat_roughness", "coat_ior", "coat_darkening", "fuzz_weight", "fuzz_color",
    "fuzz_roughness", "emission_luminance", "emission_color",
    "geometry_opacity", "geometry_thin_walled", "geometry_normal",
]
