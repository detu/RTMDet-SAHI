# ROCm build: ONNX Runtime with the MIGraphX execution provider (AMD GPUs).
#
# ONNX Runtime is found in this order:
#   1. -DONNXRUNTIME_ROOT=<dir> (or $ONNXRUNTIME_ROOT): an install / release folder
#      with include/ and lib/ (e.g. ONNX Runtime built with --use_migraphx)
#   2. the onnxruntime Python package (libonnxruntime.so.* in onnxruntime/capi),
#      as installed by AMD's instructions together with onnxruntime-ep-migraphx
# The C++ headers are not in the Python package: take include/ from the official
# release archive of the same version (onnxruntime-linux-x64-<version>.tgz) and
# pass -DONNXRUNTIME_INCLUDE_DIR=<dir>/include (or -DONNXRUNTIME_ROOT=<dir>).

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime install/release folder (include/, lib/)")
set(RTMDET_PYTHON "" CACHE FILEPATH "Python used to locate the onnxruntime / onnxruntime-ep-migraphx packages")

if(NOT RTMDET_PYTHON)
    find_program(_py NAMES python3 python)
    set(RTMDET_PYTHON "${_py}")
endif()

function(_rtmdet_py code out)
    set(${out} "" PARENT_SCOPE)
    if(RTMDET_PYTHON)
        execute_process(COMMAND "${RTMDET_PYTHON}" -c "${code}" OUTPUT_VARIABLE _o RESULT_VARIABLE _rc
                        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_rc EQUAL 0)
            set(${out} "${_o}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ---- Library
set(_ort_lib "")
if(ONNXRUNTIME_ROOT)
    find_library(_ort_lib_found NAMES onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
                 NO_DEFAULT_PATH)
    if(_ort_lib_found)
        set(_ort_lib "${_ort_lib_found}")
    endif()
endif()
if(NOT _ort_lib)
    _rtmdet_py("import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))" _ort_capi)
    if(_ort_capi)
        file(GLOB _cands "${_ort_capi}/libonnxruntime.so.*")
        list(FILTER _cands EXCLUDE REGEX "providers")
        list(SORT _cands COMPARE NATURAL ORDER DESCENDING)
        if(_cands)
            list(GET _cands 0 _ort_lib)
            message(STATUS "rtmdet_sahi: ONNX Runtime from the Python package: ${_ort_lib}")
        endif()
    endif()
endif()
if(NOT _ort_lib)
    find_library(_ort_lib_sys NAMES onnxruntime)
    if(_ort_lib_sys)
        set(_ort_lib "${_ort_lib_sys}")
    endif()
endif()
if(NOT _ort_lib)
    message(FATAL_ERROR "ONNX Runtime not found. Install AMD's ONNX Runtime + MIGraphX Python packages "
                        "(see README), or pass -DONNXRUNTIME_ROOT=<folder with include/ and lib/>.")
endif()

# Library version from the file name (libonnxruntime.so.1.29.0) or the real file it links to
get_filename_component(_ort_real "${_ort_lib}" REALPATH)
set(_ort_lib_minor "")
if(_ort_real MATCHES "libonnxruntime\\.so\\.1\\.([0-9]+)")
    set(_ort_lib_minor "${CMAKE_MATCH_1}")
endif()

# The Python package ships only libonnxruntime.so.1.x.y; the program needs it under
# its soname (libonnxruntime.so.1). Link through a symlink in the build folder.
get_filename_component(_ort_lib_dir "${_ort_real}" DIRECTORY)
set(_ort_link_dir "${CMAKE_BINARY_DIR}/onnxruntime")
file(MAKE_DIRECTORY "${_ort_link_dir}")
file(CREATE_LINK "${_ort_real}" "${_ort_link_dir}/libonnxruntime.so.1" SYMBOLIC)
file(CREATE_LINK "${_ort_real}" "${_ort_link_dir}/libonnxruntime.so" SYMBOLIC)
# ONNX Runtime loads its built-in providers (libonnxruntime_providers_migraphx.so, ...)
# from the folder of the libonnxruntime.so.1 it was loaded from: link them too.
file(GLOB _ort_providers "${_ort_lib_dir}/libonnxruntime_providers_*.so")
foreach(_p ${_ort_providers})
    get_filename_component(_pn "${_p}" NAME)
    file(CREATE_LINK "${_p}" "${_ort_link_dir}/${_pn}" SYMBOLIC)
endforeach()

# ---- Headers
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
                "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
          PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime /opt/onnxruntime/include)
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    set(_v "<version>")
    if(_ort_lib_minor)
        set(_v "1.${_ort_lib_minor}.x")
    endif()
    message(FATAL_ERROR "ONNX Runtime C++ headers (onnxruntime_cxx_api.h) not found. Download "
                        "onnxruntime-linux-x64-${_v}.tgz (same version as the library ${_ort_real}) from "
                        "https://github.com/microsoft/onnxruntime/releases, unpack it and pass "
                        "-DONNXRUNTIME_INCLUDE_DIR=<unpacked>/include")
endif()

# Headers newer than the library cannot work (the library would reject the API version)
file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _api_line REGEX "^#define ORT_API_VERSION [0-9]+")
if(_api_line MATCHES "ORT_API_VERSION ([0-9]+)" AND _ort_lib_minor)
    set(_api "${CMAKE_MATCH_1}")
    if(_api GREATER _ort_lib_minor)
        message(FATAL_ERROR "The ONNX Runtime headers (1.${_api}) are newer than the library (1.${_ort_lib_minor}, "
                            "${_ort_real}). Use the headers of version 1.${_ort_lib_minor}.")
    endif()
endif()

# ---- MIGraphX plugin EP (onnxruntime-ep-migraphx), used by default when installed
set(RTMDET_MIGRAPHX_EP "" CACHE FILEPATH "MIGraphX plugin EP library (empty = from the Python package, if any)")
set(_ep "${RTMDET_MIGRAPHX_EP}")
if(NOT _ep)
    _rtmdet_py("import onnxruntime_ep_migraphx as m; print(m.get_library_paths()[0])" _ep)
endif()
set(_rpath "${_ort_link_dir}" "${_ort_lib_dir}")
if(_ep)
    message(STATUS "rtmdet_sahi: MIGraphX plugin EP ${_ep}")
    get_filename_component(_ep_dir "${_ep}" DIRECTORY)
    list(APPEND _rpath "${_ep_dir}")
else()
    message(STATUS "rtmdet_sahi: no MIGraphX plugin EP package found; the MIGraphX EP built into "
                   "ONNX Runtime is used (if it has one), or pass --ep-lib at run time")
endif()
# MIGraphX/ROCm libraries the EP loads: pip's migraphx_libs, then the system ROCm
_rtmdet_py("import site, os; print(';'.join(d for d in (os.path.join(p, 'migraphx_libs') for p in site.getsitepackages()) if os.path.isdir(d)))" _mgx_libs)
list(APPEND _rpath ${_mgx_libs} /opt/rocm/lib)

# ---- The program
add_executable(rtmdet_sahi main.cpp region_detector.cpp ort_runner.cpp onnx_meta.cpp)
target_include_directories(rtmdet_sahi PRIVATE ${OpenCV_INCLUDE_DIRS} "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(rtmdet_sahi PRIVATE "${_ort_link_dir}/libonnxruntime.so" ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(rtmdet_sahi PRIVATE RTMDET_WITH_ROCM=1 RTMDET_DEFAULT_MIGRAPHX_EP="${_ep}")
# DT_RPATH (not RUNPATH) so the libraries the EP loads with dlopen are found too,
# without LD_LIBRARY_PATH.
list(JOIN _rpath ":" _rpath_str)
set_target_properties(rtmdet_sahi PROPERTIES BUILD_RPATH "${_rpath_str}")
target_link_options(rtmdet_sahi PRIVATE "-Wl,--disable-new-dtags")

message(STATUS "rtmdet_sahi: ONNX Runtime ${_ort_real} (headers ${ONNXRUNTIME_INCLUDE_DIR}) | "
               "backends: MIGraphX (ROCm) + CPU | ${CMAKE_BUILD_TYPE}")
