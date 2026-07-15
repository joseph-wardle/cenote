# Standalone-buildable: the byte-exact drift guard builds this directory on a
# bare CI runner, with no USD anywhere. Nothing under wire/ may depend on USD.
cmake_minimum_required(VERSION 3.24)

project(cenote-wire LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(-Wall -Wextra -Werror)
endif()
