// cenote-vdb-prep: convert every scalar float grid in a .vdb into one
// uncompressed .nvdb the renderer reads directly. The renderer core never
// links OpenVDB — this shim is the whole dependency, invoked by the scene
// loader through its beside-the-source cache (see crates/cenote/src/vdb.rs)
// or by hand. Grid names and transforms survive the conversion; compression
// deliberately does not, because the GPU maps the file bytes as-is. Per-node
// statistics survive too but nothing reads them: the majorant lattice is
// built from the stored voxels, which is what interpolation actually reads.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/openvdb.h>

// Write synthesized grids as the renderer reads them, under the names the
// fixtures read them back by. Each grid carries its own transform: one
// fixture deliberately gives two fields different ones.
static int writeSynth(const std::vector<std::pair<std::string, openvdb::FloatGrid::Ptr>>& grids,
                      const std::string& out)
{
    std::vector<nanovdb::GridHandle<>> handles;
    for (const auto& named : grids) {
        named.second->setName(named.first);
        named.second->setGridClass(openvdb::GRID_FOG_VOLUME);
        handles.push_back(nanovdb::tools::createNanoGrid(*named.second));
    }
    nanovdb::io::writeGrids(out, handles, nanovdb::io::Codec::NONE, 0);
    return 0;
}

// A constant grid whose background equals the constant, so trilinear
// interpolation reads exactly `value` everywhere — inside the active bounds
// and beyond them.
static openvdb::FloatGrid::Ptr constantGrid(int resolution, float value, double voxel)
{
    auto grid = openvdb::FloatGrid::create(/*background=*/value);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel));
    auto accessor = grid->getAccessor();
    for (int i = 0; i < resolution; ++i)
        for (int j = 0; j < resolution; ++j)
            for (int k = 0; k < resolution; ++k)
                accessor.setValue(openvdb::Coord(i, j, k), value);
    return grid;
}

// The homogeneous-limit fixture: a dense constant grid whose background
// EQUALS the constant, so trilinear interpolation reads exactly `value`
// everywhere inside (and beyond) the active bounds — no falloff band at the
// edge voxels. That is what lets the gate compare the tracker against the
// closed form with no interpolation residual between the two transports.
static int synthConstant(int argc, char** argv)
{
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: cenote-vdb-prep --constant <resolution> <value> "
                     "<voxel-size> <out.nvdb>\n");
        return 2;
    }
    const int resolution = std::atoi(argv[2]);
    const float value = std::strtof(argv[3], nullptr);
    const double voxel = std::strtod(argv[4], nullptr);
    const std::string out = argv[5];
    if (resolution <= 0 || voxel <= 0.0) {
        std::fprintf(stderr, "cenote-vdb-prep: resolution and voxel size must be positive\n");
        return 2;
    }
    std::printf("  density: constant %g over %d^3 at %g m/voxel\n", value, resolution, voxel);
    return writeSynth({{"density", constantGrid(resolution, value, voxel)}}, out);
}

// The emission fixture: the constant grid above, beside a *second* grid of
// constant temperature. Two grids rather than one reused, so that a record
// pointing its emission at the density grid's pool offset — the plausible
// mistake — renders visibly wrong.
//
// The optional last argument gives the temperature grid a voxel size of its
// own, which is the pair lowering must refuse: one affine reaches both
// fields, so two lattices cannot be sampled through it.
static int synthFire(int argc, char** argv)
{
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr,
                     "usage: cenote-vdb-prep --fire <resolution> <density> <kelvin> "
                     "<voxel-size> <out.nvdb> [temperature-voxel-size]\n");
        return 2;
    }
    const int resolution = std::atoi(argv[2]);
    const float density = std::strtof(argv[3], nullptr);
    const float kelvin = std::strtof(argv[4], nullptr);
    const double voxel = std::strtod(argv[5], nullptr);
    const std::string out = argv[6];
    const double hot = argc == 8 ? std::strtod(argv[7], nullptr) : voxel;
    if (resolution <= 0 || voxel <= 0.0 || hot <= 0.0) {
        std::fprintf(stderr, "cenote-vdb-prep: resolution and voxel size must be positive\n");
        return 2;
    }
    std::printf("  density: constant %g at %g m/voxel, temperature: constant %g K at %g "
                "m/voxel, over %d^3\n",
                density,
                voxel,
                kelvin,
                hot,
                resolution);
    return writeSynth({{"density", constantGrid(resolution, density, voxel)},
                       {"temperature", constantGrid(resolution, kelvin, hot)}},
                      out);
}

// The varying-density fixture: `base + amplitude · sin(π (k + ½) / N)` along
// index z, constant across x and y, on a background of `base`. A ray down z
// therefore crosses the piecewise-linear interpolation of a known sample
// set, whose integral is a plain trapezoid sum — a closed form to hold delta
// tracking against where the majorant genuinely varies cell to cell, which
// no constant grid can test.
static int synthRamp(int argc, char** argv)
{
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: cenote-vdb-prep --ramp <resolution> <base> <amplitude> "
                     "<voxel-size> <out.nvdb>\n");
        return 2;
    }
    const int resolution = std::atoi(argv[2]);
    const float base = std::strtof(argv[3], nullptr);
    const float amplitude = std::strtof(argv[4], nullptr);
    const double voxel = std::strtod(argv[5], nullptr);
    const std::string out = argv[6];
    if (resolution <= 0 || voxel <= 0.0) {
        std::fprintf(stderr, "cenote-vdb-prep: resolution and voxel size must be positive\n");
        return 2;
    }
    auto grid = openvdb::FloatGrid::create(/*background=*/base);
    auto accessor = grid->getAccessor();
    for (int k = 0; k < resolution; ++k) {
        const float value = base
            + amplitude
                * static_cast<float>(std::sin(M_PI * (k + 0.5) / resolution));
        for (int i = 0; i < resolution; ++i)
            for (int j = 0; j < resolution; ++j)
                accessor.setValue(openvdb::Coord(i, j, k), value);
    }
    std::printf("  density: %g + %g·sin along z over %d^3 at %g m/voxel\n",
                base,
                amplitude,
                resolution,
                voxel);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel));
    return writeSynth({{"density", grid}}, out);
}

int main(int argc, char** argv)
{
    openvdb::initialize();
    if (argc >= 2 && argv[1][0] == '-') {
        try {
            if (std::strcmp(argv[1], "--constant") == 0)
                return synthConstant(argc, argv);
            if (std::strcmp(argv[1], "--ramp") == 0)
                return synthRamp(argc, argv);
            if (std::strcmp(argv[1], "--fire") == 0)
                return synthFire(argc, argv);
            std::fprintf(stderr, "cenote-vdb-prep: unknown option %s\n", argv[1]);
            return 2;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "cenote-vdb-prep: %s\n", e.what());
            return 1;
        }
    }
    if (argc != 3) {
        std::fprintf(stderr,
                     "usage: cenote-vdb-prep <in.vdb> <out.nvdb>\n"
                     "       cenote-vdb-prep --constant <resolution> <value> "
                     "<voxel-size> <out.nvdb>\n"
                     "       cenote-vdb-prep --ramp <resolution> <base> <amplitude> "
                     "<voxel-size> <out.nvdb>\n"
                     "       cenote-vdb-prep --fire <resolution> <density> <kelvin> "
                     "<voxel-size> <out.nvdb> [temperature-voxel-size]\n"
                     "Converts every scalar float grid; other grid types are "
                     "skipped with a note.\n");
        return 2;
    }
    const std::string in = argv[1];
    const std::string out = argv[2];
    try {
        openvdb::io::File file(in);
        file.open();
        std::vector<nanovdb::GridHandle<>> handles;
        for (auto name = file.beginName(); name != file.endName(); ++name) {
            openvdb::GridBase::Ptr base = file.readGrid(name.gridName());
            auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
            if (!grid) {
                std::fprintf(stderr,
                             "  skipping \"%s\" (%s): only scalar float grids "
                             "convert\n",
                             name.gridName().c_str(),
                             base->valueType().c_str());
                continue;
            }
            handles.push_back(nanovdb::tools::createNanoGrid(*grid));
            std::printf("  %s: %llu active voxels, %.1f MiB\n",
                        name.gridName().c_str(),
                        static_cast<unsigned long long>(grid->activeVoxelCount()),
                        static_cast<double>(handles.back().gridSize(0))
                            / (1024.0 * 1024.0));
        }
        file.close();
        if (handles.empty()) {
            std::fprintf(stderr,
                         "cenote-vdb-prep: no scalar float grids in %s\n",
                         in.c_str());
            return 1;
        }
        nanovdb::io::writeGrids(out, handles, nanovdb::io::Codec::NONE, 0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cenote-vdb-prep: %s\n", e.what());
        return 1;
    }
    return 0;
}
