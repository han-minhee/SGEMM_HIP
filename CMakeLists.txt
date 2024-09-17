cmake_minimum_required(VERSION 3.19)
project(HIP_SGEMM_PRACTICE LANGUAGES CXX HIP)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

set(CMAKE_C_COMPILER "${ROCM_ROOT}/bin/hipcc")
set(CMAKE_CXX_COMPILER "${ROCM_ROOT}/bin/hipcc")
set(CMAKE_HIP_COMPILER "${ROCM_ROOT}/bin/nvcc")

set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" ${CMAKE_MODULE_PATH})
find_package(hip REQUIRED)

# set(CMAKE_CXX_STANDARD 17)

set(HIP_COMPILE_OPTIONS "")
set(ROCM_TARGET_DEVICE "" CACHE STRING "AMD GPU target architecture")
set(CUDA_TARGET_ARCH "" CACHE STRING "NVIDIA GPU target architecture")

if(ROCM_TARGET_DEVICE)
    message(STATUS "Compiling for AMD GPU with target architecture: ${ROCM_TARGET_DEVICE}")
    list(APPEND HIP_COMPILE_OPTIONS --offload-arch=${ROCM_TARGET_DEVICE})
endif()

if(CUDA_TARGET_ARCH)
    message(STATUS "Compiling for NVIDIA GPU with target architecture: ${CUDA_TARGET_ARCH}")
    add_definitions(-DCMAKE_HIP_COMPILER=nvcc)
    add_definitions(-DCMAKE_HIP_ARCHITECTURES=${CUDA_TARGET_ARCH})
    add_definitions(-DCMAKE_GPU_RUNTIME=CUDA)
endif()

# Ensure at least one GPU target is specified
if(NOT ROCM_TARGET_DEVICE AND NOT CUDA_TARGET_ARCH)
    message(FATAL_ERROR "Please specify either ROCM_TARGET_DEVICE or CUDA_TARGET_ARCH.")
endif()

if(ROCM_TARGET_DEVICE AND CUDA_TARGET_ARCH)
    message(FATAL_ERROR "Please specify only one GPU target.")
endif()

add_compile_options(${HIP_COMPILE_OPTIONS})

# In debug mode, add debug symbols to device code
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:HIP>>:-g;-O0>")

if(NOT DEFINED TUNING)
    if(NOT DEFINED K9_BK)
        message("K9_BK not defined in the flags. Trying to load from best_kernel_9_config.cmake")
        if(EXISTS "${CMAKE_SOURCE_DIR}/benchmark_results/best_kernel_9_config.cmake")
            include("${CMAKE_SOURCE_DIR}/benchmark_results/best_kernel_9_config.cmake")
        else()
            message(WARNING "best_kernel_9_config.cmake not found. Using default configuration.")
        endif()
    endif()
else()
    message("TUNING flag set. Skipping loading from best_kernel_9_config.cmake.")
endif()

if(DEFINED K9_BK)
    message("Using K9_BK=${K9_BK}")
    add_definitions(-DK9_BK=${K9_BK})
    add_definitions(-DK9_TM=${K9_TM})
    add_definitions(-DK9_TN=${K9_TN})
    add_definitions(-DK9_BM=${K9_BM})
    add_definitions(-DK9_BN=${K9_BN})
    add_definitions(-DK9_NUM_THREADS=${K9_NUM_THREADS})
else()
    message("K9_BK is not defined.")
endif()

if(NOT DEFINED TUNING)
    if(NOT DEFINED K10_BK)
        message("K10_BK not defined in the flags. Trying to load from best_kernel_10_config.cmake")
        if(EXISTS "${CMAKE_SOURCE_DIR}/benchmark_results/best_kernel_10_config.cmake")
            include("${CMAKE_SOURCE_DIR}/benchmark_results/best_kernel_10_config.cmake")
        else()
            message(WARNING "best_kernel_10_config.cmake not found. Using default configuration.")
        endif()
    endif()
else()
    message("TUNING flag set. Skipping loading from best_kernel_10_config.cmake.")
endif()

if(DEFINED K10_BK)
    message("Using K10_BK=${K10_BK}")
    add_definitions(-DK10_NUM_THREADS=${K10_NUM_THREADS})
    add_definitions(-DK10_BN=${K10_BN})
    add_definitions(-DK10_BM=${K10_BM})
    add_definitions(-DK10_BK=${K10_BK})
    add_definitions(-DK10_WM=${K10_WM})
    add_definitions(-DK10_WN=${K10_WN})
    add_definitions(-DK10_WNITER=${K10_WNITER})
    add_definitions(-DK10_TN=${K10_TN})
    add_definitions(-DK10_TM=${K10_TM})
else()
    message("K10_BK is not defined.")
endif()

include_directories(${ROCM_ROOT}/include)
include_directories(${HIP_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

# temporary workaround for hipBLAS not working with NVCC
# You should manually install hipBLAS
set(HIPBLAS_LIBRARY_PATH "${ROCM_ROOT}/hipblas/lib")
link_directories(${HIPBLAS_LIBRARY_PATH})

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

add_executable(sgemm sgemm.cpp ${SRC})
target_compile_definitions(sgemm PRIVATE HIPBLAS_V2)
target_link_libraries(sgemm "${HIPBLAS_LIBRARY_PATH}/libhipblas.so")

add_executable(hipBLAS_sgemm hipBLAS_sgemm.cpp)
target_compile_definitions(hipBLAS_sgemm PRIVATE HIPBLAS_V2)
target_link_libraries(hipBLAS_sgemm "${HIPBLAS_LIBRARY_PATH}/libhipblas.so")
