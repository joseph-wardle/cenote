# The C++ half of M4: the wire mirror now, the hdCenote delegate plugin in
# the following commit. Lives outside the Cargo workspace; see docs/m4-plan.md.
cmake_minimum_required(VERSION 3.24)

project(cenote-hydra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Werror)

enable_testing()

add_subdirectory(wire)
