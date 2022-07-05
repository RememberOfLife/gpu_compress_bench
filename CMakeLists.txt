# convenient cuda clang support was added in 3.19
cmake_minimum_required(VERSION 3.19)
set(CLANG_VERSION "12" CACHE STRING "specify clang version")
option(CLANG_LIBCPP "work around clang ICE on libstdc++ by using libc++ instead" ON)

if ("${CLANG_VERSION}" STREQUAL "")
set(CLANG_VERSION_POSTFIX "")
else()
set(CLANG_VERSION_POSTFIX "-${CLANG_VERSION}")
endif()
set(CMAKE_CUDA_COMPILER "clang++${CLANG_VERSION_POSTFIX}")
set(CMAKE_CXX_COMPILER "clang++${CLANG_VERSION_POSTFIX}")

# force c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# don't enable gnu extensions
set(CMAKE_CXX_EXTENSIONS OFF)

project(gpu_compress_bench LANGUAGES CXX CUDA)

set(INCLUDES
    src
    .
)

set(SOURCES
    src/main.cu
)

add_executable(gpu_compress_bench ${SOURCES})


set_property(TARGET gpu_compress_bench PROPERTY CUDA_ARCHITECTURES 75 61)

target_include_directories(gpu_compress_bench PRIVATE ${INCLUDES})

set_target_properties(gpu_compress_bench PROPERTIES EXPORT_COMPILE_COMMANDS true)
