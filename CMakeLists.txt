cmake_minimum_required(VERSION 2.8.9)
project(Proj)

# Adding compiler optimizations

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER "g++-7")

# set(COMMON_COMPILER_FLAGS "")
set(COMMON_COMPILER_FLAGS_DEBUG   "-O0")
set(COMMON_COMPILER_FLAGS_RELEASE "-O3 -DNDEBUG")

# C++ compiler flags

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -march=native -fPIC \
                             ${COMMON_COMPILER_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} \
                             ${COMMON_COMPILER_FLAGS_DEBUG}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
                             ${COMMON_COMPILER_FLAGS_RELEASE}")

# NVCC host compiler flags

set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS} --std=c++14 \
                             ${COMMON_COMPILER_FLAGS}")

set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG} \
                             ${COMMON_COMPILER_FLAGS_DEBUG}")

set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} \
                             ${COMMON_COMPILER_FLAGS_RELEASE}")

# Adding includes

include_directories("include/")

# Adding sources

file(GLOB_RECURSE SOURCES       "src/*.cpp" )
file(GLOB_RECURSE CUDA_SOURCES  "src/*.cu"  )

cuda_add_executable(prog ${CUDA_SOURCES} ${SOURCES})
