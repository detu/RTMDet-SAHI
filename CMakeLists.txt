cmake_minimum_required(VERSION 3.18)
project(rtmdet_sahi CXX)

# Default to Release: a Debug build makes inference many times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU backend: CUDA or Vulkan -- one of them, not both (the CPU backend is
# always built). Default CUDA; for Vulkan: -DRTMDET_CUDA=OFF -DRTMDET_VULKAN=ON.
option(RTMDET_CUDA "Build MNN's CUDA backend (needs the CUDA toolkit / nvcc)" ON)
option(RTMDET_VULKAN "Build MNN's Vulkan backend (needs only the Vulkan loader at run time)" OFF)
if(RTMDET_CUDA AND RTMDET_VULKAN)
    message(FATAL_ERROR "RTMDET_CUDA and RTMDET_VULKAN cannot both be ON; choose one:\n"
                        "  CUDA:   -DRTMDET_CUDA=ON  -DRTMDET_VULKAN=OFF\n"
                        "  Vulkan: -DRTMDET_CUDA=OFF -DRTMDET_VULKAN=ON\n"
                        "(use a separate build directory for each)")
endif()
option(RTMDET_CUDA_NATIVE_ARCH "Compile MNN's CUDA kernels only for the GPU in this machine" ON)
# GPU architecture for the CUDA kernels, e.g. 8.9 (RTX 40xx / RTX Ada), 8.6 (RTX 30xx), 12.0 (RTX 50xx).
# Empty = detect the GPU in this machine while configuring.
set(RTMDET_CUDA_ARCH "" CACHE STRING "CUDA compute capability to build for (empty = detect)")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MNN/CMakeLists.txt")
    message(FATAL_ERROR "MNN submodule missing: run 'git submodule update --init'")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---- MNN (submodule), built as its own project: its CMake files expect to be
#      the top-level project. Also builds MNNConvert, used to turn .onnx into .mnn.
set(_mnn_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5        # CMake 4 refuses MNN's old minimum otherwise
    -DCMAKE_POLICY_DEFAULT_CMP0146=OLD        # MNN's CUDA backend uses FindCUDA
    -DMNN_BUILD_SHARED_LIBS=ON
    -DMNN_SEP_BUILD=OFF                       # backends inside libMNN.so
    -DMNN_BUILD_CONVERTER=ON
    -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF
    -DMNN_VULKAN=${RTMDET_VULKAN}
    -DMNN_CUDA=${RTMDET_CUDA})

if(RTMDET_CUDA)
    # nvcc is often not on PATH (e.g. WSL2 without /usr/local/cuda/bin in PATH):
    # look in the standard install locations too.
    file(GLOB _cuda_dirs LIST_DIRECTORIES true "/usr/local/cuda-*")
    list(SORT _cuda_dirs COMPARE NATURAL ORDER DESCENDING)  # newest versioned install first
    find_program(RTMDET_NVCC nvcc
                 HINTS "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin" /usr/local/cuda/bin
                 PATHS ${_cuda_dirs} PATH_SUFFIXES bin)
    if(NOT RTMDET_NVCC)
        message(FATAL_ERROR "nvcc not found. Install the CUDA toolkit (cuda-toolkit), add its bin/ to PATH "
                            "or pass -DRTMDET_NVCC=/path/to/nvcc; or build without CUDA: -DRTMDET_CUDA=OFF")
    endif()
    get_filename_component(_cuda_root "${RTMDET_NVCC}" DIRECTORY)
    get_filename_component(_cuda_root "${_cuda_root}" DIRECTORY)
    list(APPEND _mnn_args
         "-DCUDA_TOOLKIT_ROOT_DIR=${_cuda_root}"
         "-DCUDA_NVCC_EXECUTABLE=${RTMDET_NVCC}"
         -DMNN_CUDA_NATIVE_ARCH=${RTMDET_CUDA_NATIVE_ARCH})
    if(RTMDET_CUDA_ARCH)
        list(APPEND _mnn_args "-DCUDA_ARCHS=${RTMDET_CUDA_ARCH}")
    endif()
    message(STATUS "rtmdet_sahi: CUDA toolkit ${_cuda_root}")
endif()

include(ExternalProject)
set(MNN_BUILD_DIR "${CMAKE_BINARY_DIR}/mnn")
set(MNN_LIBRARY "${MNN_BUILD_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}MNN${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(MNN_CONVERTER "${MNN_BUILD_DIR}/MNNConvert")
ExternalProject_Add(mnn_build
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/MNN"
    BINARY_DIR "${MNN_BUILD_DIR}"
    CMAKE_ARGS ${_mnn_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target MNN MNNConvert
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${MNN_LIBRARY}" "${MNN_CONVERTER}"
    USES_TERMINAL_BUILD TRUE)

add_library(mnn_lib SHARED IMPORTED)
set_target_properties(mnn_lib PROPERTIES
    IMPORTED_LOCATION "${MNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/MNN/include")
add_dependencies(mnn_lib mnn_build)

# ---- The program
add_executable(rtmdet_sahi main.cpp mnn_detector.cpp onnx_meta.cpp)
add_dependencies(rtmdet_sahi mnn_build)
target_include_directories(rtmdet_sahi PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rtmdet_sahi PRIVATE mnn_lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(rtmdet_sahi PRIVATE
    RTMDET_WITH_CUDA=$<BOOL:${RTMDET_CUDA}>
    RTMDET_WITH_VULKAN=$<BOOL:${RTMDET_VULKAN}>
    RTMDET_MNNCONVERT="${MNN_CONVERTER}")
# Find libMNN.so at run time without LD_LIBRARY_PATH
set_target_properties(rtmdet_sahi PROPERTIES BUILD_RPATH "${MNN_BUILD_DIR}")

if(RTMDET_CUDA)
    set(_gpu "CUDA")
elseif(RTMDET_VULKAN)
    set(_gpu "Vulkan")
else()
    set(_gpu "no GPU")
endif()
message(STATUS "rtmdet_sahi: MNN backends: ${_gpu} + CPU | ${CMAKE_BUILD_TYPE}")
