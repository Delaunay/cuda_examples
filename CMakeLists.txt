cmake_minimum_required(VERSION 3.16)

project(cuda_examples CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDA)
include_directories(common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

add_subdirectory(0_Simple)
add_subdirectory(8_gemm)


ADD_CUSTOM_TARGET(zothers ALL SOURCES
    README.md
    .gitignore
)
