# cenote-vdb-prep: the optional .vdb → .nvdb converter (B11's front door).
# Lives outside the Cargo workspace like the Hydra delegate: the renderer
# core reads only .nvdb, and this shim is the one place OpenVDB and its
# dependency stack appear. Build it once, put the binary on PATH (or point
# CENOTE_VDB_PREP at it), and `.vdb` scene references prep themselves.
#
#   cmake -S vdb-prep -B build/vdb-prep && cmake --build build/vdb-prep
cmake_minimum_required(VERSION 3.24)

project(cenote-vdb-prep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Werror)

# OpenVDB ships a find module, not a package config — Fedora installs it
# under lib64/cmake/OpenVDB. Whatever OpenVDB the system has does the .vdb
# reading; only the NanoVDB headers below need to match the renderer.
find_path(OPENVDB_CMAKE_DIR FindOpenVDB.cmake
    PATHS /usr/lib64/cmake/OpenVDB /usr/lib/cmake/OpenVDB
          /usr/local/lib/cmake/OpenVDB
    DOC "Directory containing FindOpenVDB.cmake")
if(NOT OPENVDB_CMAKE_DIR)
    message(FATAL_ERROR "FindOpenVDB.cmake not found — install openvdb-devel \
or set -DOPENVDB_CMAKE_DIR")
endif()
list(APPEND CMAKE_MODULE_PATH ${OPENVDB_CMAKE_DIR})
find_package(OpenVDB REQUIRED)

# The NanoVDB headers (header-only) at the exact tag the renderer's
# vendored PNanoVDB.h comes from, so writer and reader share one ABI
# (32.7). The hash is the pin, corpus-fetch style. SOURCE_SUBDIR points at
# a directory with no CMakeLists.txt so nothing of OpenVDB's own build
# configures — this is a header checkout, not a subproject.
include(FetchContent)
FetchContent_Declare(openvdb_headers
    URL https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v12.0.1.tar.gz
    URL_HASH SHA256=a3c8724ecadabaf558b6e1bd6f1d695e93b82a7cfdf144b8551e5253340ddce0
    SOURCE_SUBDIR cmake)
FetchContent_MakeAvailable(openvdb_headers)

add_executable(cenote-vdb-prep vdb_prep.cpp)
# SYSTEM: the template-heavy NanoVDB internals are not held to -Werror.
target_include_directories(cenote-vdb-prep SYSTEM PRIVATE
    ${openvdb_headers_SOURCE_DIR}/nanovdb)
target_link_libraries(cenote-vdb-prep PRIVATE OpenVDB::openvdb)
# Gates NanoVDB's openvdb→nanovdb adapters (CreateNanoGrid's whole point
# here) — without it the generic accessor instantiates against the OpenVDB
# grid and fails to compile.
target_compile_definitions(cenote-vdb-prep PRIVATE NANOVDB_USE_OPENVDB)

install(TARGETS cenote-vdb-prep)
