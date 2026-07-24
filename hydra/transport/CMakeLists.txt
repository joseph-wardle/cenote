# The transport client: cenote-server spawn, loopback socket, framed
# request/response, and the shm framebuffer reader. A static archive
# folded into the hdCenote MODULE — subdirectories are source
# organization, never separate shared libraries.
add_library(cenote-transport STATIC client.cpp view.cpp)
# The archive's objects land inside the plugin .so.
set_target_properties(cenote-transport PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Consumers name headers from the hydra root — "transport/client.hpp",
# and through it "wire/protocol.hpp".
target_include_directories(cenote-transport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
# cenote_usd is the flavor-neutral USD handle (../CMakeLists.txt); the
# client reaches only pxr/base/tf (TfDiagnostic) but binds through the same
# handle hdCenote does, so neither the stock `tf` target nor the HDK's
# Houdini::Dep::pxr_tf is named here directly.
target_link_libraries(cenote-transport PUBLIC cenote-wire PRIVATE cenote_usd)
# Same libstdc++ backward-hash permit as hdCenote: tf headers reach the
# deprecated <ext/hash_set> on every GCC build (see ../hdCenote).
target_compile_definitions(cenote-transport PRIVATE _GLIBCXX_PERMIT_BACKWARD_HASH)
