# The wire mirror and the hdCenote delegate plugin. Lives outside the Cargo
# workspace; see hydra/README.md.
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

# USD enters exactly here, in one of two flavors. The delegate is
# one source; only which USD it binds to changes. `stock` is OpenUSD 26.05
# on CMAKE_PREFIX_PATH (or pxr_DIR) — the everyday build and the pixel
# oracle. `hdk` is Houdini's own USD, so the same .so loads inside husk.
# wire/ — built standalone by CI — never reaches here and never sees USD.
# Everything downstream links the flavor-neutral `cenote_usd` handle below,
# so hdCenote/ stays ignorant of which USD answered.
set(CENOTE_USD_FLAVOR "stock" CACHE STRING
    "Which USD the delegate builds against: stock (OpenUSD 26.05) or hdk (Houdini's own)")
set_property(CACHE CENOTE_USD_FLAVOR PROPERTY STRINGS stock hdk)
message(STATUS "cenote: USD flavor = ${CENOTE_USD_FLAVOR}")

add_library(cenote_usd INTERFACE)

if(CENOTE_USD_FLAVOR STREQUAL "stock")
    # Stock pxr exports component targets whose interface deps pull the whole
    # transitive closure (tf, sdf, vt, …) behind these three. Imported pxr
    # targets carry their headers as SYSTEM, so -Werror above stays ours alone.
    find_package(pxr REQUIRED CONFIG)
    target_link_libraries(cenote_usd INTERFACE hd hdsi usdLux)
elseif(CENOTE_USD_FLAVOR STREQUAL "hdk")
    # The HDK ships no pxrConfig.cmake — only HoudiniConfig.cmake, found via
    # Houdini_DIR=$HFS/toolkit/cmake. Its `Houdini` interface carries the
    # toolkit/include headers we need but also forces -std=c++17 and drags in
    # the entire core-Houdini/Qt stack; we take only its SYSTEM include dirs
    # so cenote stays C++23 and links nothing but USD.
    find_package(Houdini REQUIRED CONFIG)
    get_target_property(_hdk_includes Houdini INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(cenote_usd SYSTEM INTERFACE ${_hdk_includes})
    # Match the ABI the HDK libs were built with (GCC's default on this
    # platform already, stated so the binding is not left to chance).
    target_compile_definitions(cenote_usd INTERFACE _GLIBCXX_USE_CXX11_ABI=1)
    # USD arrives as Houdini::Dep::pxr_* imported targets that carry only a
    # location — no transitive deps — so the closure stock resolved through
    # hd/hdsi/usdLux is named in full here. usd/usdGeom/usdShade back usdLux;
    # the rest is the delegate's own base + imaging footprint.
    foreach(_pxr_lib
            arch tf js trace work plug vt ar gf sdf sdr ndr pcp kind
            hf cameraUtil pxOsd hd hdsi usd usdGeom usdShade usdLux)
        target_link_libraries(cenote_usd INTERFACE Houdini::Dep::pxr_${_pxr_lib})
    endforeach()
else()
    message(FATAL_ERROR
        "CENOTE_USD_FLAVOR must be 'stock' or 'hdk', got '${CENOTE_USD_FLAVOR}'")
endif()

# `cmake --install` lands in the gitignored dist/ tree that
# PXR_PLUGINPATH_NAME points at; -DCMAKE_INSTALL_PREFIX overrides as usual.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/dist" CACHE PATH "Install path prefix" FORCE)
endif()

add_subdirectory(wire)
add_subdirectory(transport)
add_subdirectory(hdCenote)
