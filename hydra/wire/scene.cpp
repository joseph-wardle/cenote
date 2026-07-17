#include "scene.hpp"

#include <variant>

namespace cenote::wire {

void encode(Writer& writer, Kind value) {
    switch (value) {
    case Kind::Mesh:
        writer.str("Mesh");
        return;
    case Kind::Instance:
        writer.str("Instance");
        return;
    case Kind::Material:
        writer.str("Material");
        return;
    case Kind::Light:
        writer.str("Light");
        return;
    case Kind::Camera:
        writer.str("Camera");
        return;
    case Kind::Environment:
        writer.str("Environment");
        return;
    case Kind::Settings:
        writer.str("Settings");
        return;
    }
}

void encode(Writer& writer, ColorSpace value) {
    switch (value) {
    case ColorSpace::Srgb:
        writer.str("Srgb");
        return;
    case ColorSpace::Linear:
        writer.str("Linear");
        return;
    }
}

void encode(Writer& writer, Channel value) {
    switch (value) {
    case Channel::R:
        writer.str("R");
        return;
    case Channel::G:
        writer.str("G");
        return;
    case Channel::B:
        writer.str("B");
        return;
    case Channel::A:
        writer.str("A");
        return;
    }
}

void encode(Writer& writer, const TextureRef& value) {
    writer.array_header(3);
    encode(writer, value.path);
    encode(writer, value.color_space);
    encode(writer, value.channel);
}

void encode(Writer& writer, const Transform& value) {
    writer.map_header(1);
    std::visit(Overloaded{
                   [&](const Trs& trs) {
                       writer.str("Trs");
                       writer.array_header(3);
                       encode(writer, trs.translate);
                       encode(writer, trs.rotate_degrees);
                       encode(writer, trs.scale);
                   },
                   [&](const Matrix& matrix) {
                       writer.str("Matrix");
                       encode(writer, matrix.rows);
                   },
               },
               value);
}

void encode(Writer& writer, const MeshSource& value) {
    writer.map_header(1);
    std::visit(Overloaded{
                   [&](const Inline& mesh) {
                       writer.str("Inline");
                       writer.array_header(4);
                       encode(writer, mesh.positions);
                       encode(writer, mesh.normals);
                       encode(writer, mesh.uvs);
                       encode(writer, mesh.triangles);
                   },
                   [&](const Ply& ply) {
                       writer.str("Ply");
                       writer.array_header(1);
                       encode(writer, ply.path);
                   },
               },
               value);
}

void encode(Writer& writer, const Light& value) {
    writer.map_header(1);
    std::visit(Overloaded{
                   [&](const Distant& distant) {
                       writer.str("Distant");
                       writer.array_header(2);
                       encode(writer, distant.direction);
                       encode(writer, distant.irradiance);
                   },
                   [&](const Point& point) {
                       writer.str("Point");
                       writer.array_header(2);
                       encode(writer, point.position);
                       encode(writer, point.intensity);
                   },
               },
               value);
}

void encode(Writer& writer, const MeshPatch& value) {
    writer.array_header(2);
    encode(writer, value.name);
    encode(writer, value.source);
}

void encode(Writer& writer, const InstancePatch& value) {
    writer.array_header(5);
    encode(writer, value.name);
    encode(writer, value.mesh);
    encode(writer, value.material);
    encode(writer, value.transforms);
    encode(writer, value.camera_visible);
}

void encode(Writer& writer, const MaterialPatch& value) {
    writer.array_header(23);
    encode(writer, value.name);
    encode(writer, value.base_color);
    encode(writer, value.base_diffuse_roughness);
    encode(writer, value.base_metalness);
    encode(writer, value.specular_weight);
    encode(writer, value.specular_roughness);
    encode(writer, value.specular_ior);
    encode(writer, value.transmission_weight);
    encode(writer, value.transmission_color);
    encode(writer, value.transmission_depth);
    encode(writer, value.coat_weight);
    encode(writer, value.coat_color);
    encode(writer, value.coat_roughness);
    encode(writer, value.coat_ior);
    encode(writer, value.coat_darkening);
    encode(writer, value.fuzz_weight);
    encode(writer, value.fuzz_color);
    encode(writer, value.fuzz_roughness);
    encode(writer, value.emission_luminance);
    encode(writer, value.emission_color);
    encode(writer, value.geometry_opacity);
    encode(writer, value.geometry_thin_walled);
    encode(writer, value.geometry_normal);
}

void encode(Writer& writer, const LightPatch& value) {
    writer.array_header(2);
    encode(writer, value.name);
    encode(writer, value.light);
}

void encode(Writer& writer, const CameraPatch& value) {
    writer.array_header(7);
    encode(writer, value.name);
    encode(writer, value.position);
    encode(writer, value.look_at);
    encode(writer, value.up);
    encode(writer, value.vfov_degrees);
    encode(writer, value.focus_distance);
    encode(writer, value.aperture_radius);
}

void encode(Writer& writer, const EnvironmentPatch& value) {
    writer.array_header(4);
    encode(writer, value.name);
    encode(writer, value.path);
    encode(writer, value.tint);
    encode(writer, value.transform);
}

void encode(Writer& writer, const SettingsPatch& value) {
    writer.array_header(5);
    encode(writer, value.name);
    encode(writer, value.resolution);
    encode(writer, value.spp);
    encode(writer, value.max_bounces);
    encode(writer, value.seed);
}

void encode(Writer& writer, const Op& value) {
    writer.map_header(1);
    std::visit(Overloaded{
                   [&](const MeshPatch& patch) {
                       writer.str("Mesh");
                       encode(writer, patch);
                   },
                   [&](const InstancePatch& patch) {
                       writer.str("Instance");
                       encode(writer, patch);
                   },
                   [&](const MaterialPatch& patch) {
                       writer.str("Material");
                       encode(writer, patch);
                   },
                   [&](const LightPatch& patch) {
                       writer.str("Light");
                       encode(writer, patch);
                   },
                   [&](const CameraPatch& patch) {
                       writer.str("Camera");
                       encode(writer, patch);
                   },
                   [&](const EnvironmentPatch& patch) {
                       writer.str("Environment");
                       encode(writer, patch);
                   },
                   [&](const SettingsPatch& patch) {
                       writer.str("Settings");
                       encode(writer, patch);
                   },
                   [&](const Remove& remove) {
                       writer.str("Remove");
                       writer.array_header(2);
                       encode(writer, remove.kind);
                       encode(writer, remove.name);
                   },
               },
               value);
}

void encode(Writer& writer, const ChangeSet& value) {
    writer.array_header(1);
    encode(writer, value.ops);
}

} // namespace cenote::wire
