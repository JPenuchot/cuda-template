cmake_minimum_required(VERSION 2.8.9)
project(Proj)

# Adding compiler optimizations

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
  -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
  -O3 -mavx2 -mfma -DNDEBUG")


set(CMAKE_CXX_STANDARD 17)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")

set(CUDA_HOST_COMPILER "clang++-4.0")

# Adding includes

include_directories("include/")

# Adding sources

file(GLOB_RECURSE SOURCES       "src/*.cpp" )
file(GLOB_RECURSE CUDA_SOURCES  "src/*.cu"  )

cuda_add_executable(prog ${CUDA_SOURCES} ${SOURCES})
