# The hdCenote renderer plugin: one MODULE .so against the USD found at the
# top level, installed into the gitignored dist/ layout that
# PXR_PLUGINPATH_NAME points at (see ../README.md).
add_library(hdCenote MODULE
    lightPrim.cpp
    materialPrim.cpp
    meshPrim.cpp
    observer.cpp
    renderBuffer.cpp
    renderDelegate.cpp
    renderPass.cpp
    rendererPlugin.cpp
    sceneIndexPlugins.cpp
)
# Plug resolves the library by the exact LibraryPath in plugInfo.json — no
# "lib" prefix.
set_target_properties(hdCenote PROPERTIES PREFIX "")
target_link_libraries(hdCenote PRIVATE cenote-transport hd hdsi)
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

install(TARGETS hdCenote DESTINATION hdCenote)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json" DESTINATION hdCenote/resources)
