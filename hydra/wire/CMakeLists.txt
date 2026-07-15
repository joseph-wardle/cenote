# Standalone-buildable: the byte-exact drift guard builds this directory on a
# bare CI runner, with no USD anywhere. Nothing under wire/ may depend on USD.
cmake_minimum_required(VERSION 3.24)

project(cenote-wire LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(-Wall -Wextra -Werror)
    enable_testing()
endif()

add_library(cenote-wire STATIC msgpack.cpp scene.cpp protocol.cpp)
# The archive reaches the hdCenote MODULE through the transport library,
# so its objects must be relocatable — in the standalone CI build too.
set_target_properties(cenote-wire PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(msgpack-test msgpack_test.cpp)
target_link_libraries(msgpack-test PRIVATE cenote-wire)
add_test(NAME msgpack COMMAND msgpack-test)

# The conformance runner reads the Rust crate's checked-in goldens — the
# one place wire/ reaches outside its directory, and it is data, not USD.
add_executable(corpus-test corpus_test.cpp)
target_link_libraries(corpus-test PRIVATE cenote-wire)
add_test(NAME corpus
         COMMAND corpus-test "${CMAKE_CURRENT_SOURCE_DIR}/../../crates/cenote-wire/tests/golden")
