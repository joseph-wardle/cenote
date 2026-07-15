# The C++ half of M4: the wire mirror and the hdCenote delegate plugin.
# Lives outside the Cargo workspace; see docs/m4-plan.md.
cmake_minimum_required(VERSION 3.24)

project(cenote-hydra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# clang-tidy (the .clang-tidy ritual in README.md) reads the compile
# database from the build tree.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Werror)

enable_testing()

# USD enters exactly here: the full tree needs a pxr prefix on
# CMAKE_PREFIX_PATH (or pxr_DIR), while wire/ — built standalone by CI —
# must never see it. Imported pxr targets carry their headers as SYSTEM,
# so -Werror above stays ours alone.
find_package(pxr REQUIRED CONFIG)

# `cmake --install` lands in the gitignored dist/ tree that
# PXR_PLUGINPATH_NAME points at; -DCMAKE_INSTALL_PREFIX overrides as usual.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/dist" CACHE PATH "Install path prefix" FORCE)
endif()

add_subdirectory(wire)
add_subdirectory(transport)
add_subdirectory(hdCenote)
