cmake_minimum_required(VERSION 3.18)
project(gemm_from_scratch LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # adjust based on your GPU

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include source and kernel directories
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(gemm_from_scratch
    src/main.cu
    src/kernels/1_gemm_naive.cu
)

# Optimize CUDA build
set_target_properties(gemm_from_scratch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
