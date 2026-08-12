# The hdCenote renderer plugin: one MODULE .so against the USD found at the
# top level, installed into the gitignored dist/ layout that
# PXR_PLUGINPATH_NAME points at (see ../README.md).
add_library(hdCenote MODULE
    domePrim.cpp
    instancerPrim.cpp
    lightPrim.cpp
    materialPrim.cpp
    meshPrim.cpp
    observer.cpp
    renderBuffer.cpp
    renderDelegate.cpp
    renderPass.cpp
    renderSettings.cpp
    rendererPlugin.cpp
    sceneIndexPlugins.cpp
    settingsPrim.cpp
)
# Plug resolves the library by the exact LibraryPath in plugInfo.json — no
# "lib" prefix.
set_target_properties(hdCenote PROPERTIES PREFIX "")
# cenote_usd is the flavor-neutral USD handle chosen at the top level
# (stock OpenUSD 26.05 or the HDK's own USD); usdLux rides inside it
# for UsdLuxBlackbodyTemperatureAsRgb, the one spot of USD outside imaging
# the delegate touches (the light translator's color temperature).
target_link_libraries(hdCenote PRIVATE cenote-transport cenote_usd)
# tf/token.h reaches <ext/hash_set> on every GCC/Linux build (USD's
# ARCH_HAS_GNU_STL_EXTENSIONS is unconditional there), and libstdc++'s
# deprecation #warning fires straight through -isystem, which -Werror then
# fatalizes. This is libstdc++'s own permit for exactly those backward-hash
# headers; every other deprecation diagnostic stays live.
target_compile_definitions(hdCenote PRIVATE _GLIBCXX_PERMIT_BACKWARD_HASH)

# plugInfo paths resolve relative to Root (".." from resources/ is the
# installed hdCenote/ directory), so LibraryPath is the bare .so name.
set(HD_CENOTE_LIBRARY_PATH "hdCenote.so")
set(HD_CENOTE_DISPLAY_NAME "Cenote")
configure_file(plugInfo.json.in plugInfo.json @ONLY)
# The same display name reaches C++ (sceneIndexPlugins.cpp registers its
# filters for this renderer), so the two spellings cannot drift apart.
target_compile_definitions(hdCenote PRIVATE
    HD_CENOTE_DISPLAY_NAME="${HD_CENOTE_DISPLAY_NAME}")

# Settings resolution is a pure function of a settings map, so it is the
# one piece of the delegate testable without a stage, a server, or a GPU.
# The source compiles a second time here rather than being split into its
# own archive: a MODULE cannot be linked into an executable, and one file
# is cheaper than a library boundary that exists only for a test.
add_executable(settings-test renderSettings_test.cpp renderSettings.cpp)
target_link_libraries(settings-test PRIVATE cenote-transport cenote_usd_exe)
target_compile_definitions(settings-test PRIVATE _GLIBCXX_PERMIT_BACKWARD_HASH)
add_test(NAME settings COMMAND settings-test)

install(TARGETS hdCenote DESTINATION hdCenote)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json" DESTINATION hdCenote/resources)

# husk discovers renderers from UsdRenderers.json files at the top level of
# each HOUDINI_PATH entry — a Houdini-path artifact, not a pxr plugin resource.
# Installed at the dist root so `HOUDINI_PATH=<dist>:&` makes husk offer
# `--renderer HdCenoteRendererPlugin`; the .so itself still loads through
# PXR_PLUGINPATH_NAME as usual. Only the hdk build is ever seen by husk, so the
# stock dist keeps none of this.
if(CENOTE_USD_FLAVOR STREQUAL "hdk")
    install(FILES UsdRenderers.json DESTINATION .)
endif()
