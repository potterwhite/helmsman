# Copyright (c) 2025 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# third_party/sherpa-onnx/CMakeLists.txt
project(sherpa-onnx-download NONE)          # NONE means current project does not need to be compiled

# ---------------------------------
# I. Import "FetchContent" Module
# ---------------------------------
include(FetchContent)

message(STATUS "[Configuration][ThirdParty] Preparing Sherpa-Onnx...")

# ---------------------------------
# II. Declare Download Detail
# ---------------------------------
# These variables come from CMakePresets.json
if(NOT DEFINED SHERPA_ONNX_URL OR NOT DEFINED SHERPA_ONNX_HASH)
    message(Warning "[Configuration][ThirdParty] SHERPA_ONNX_URL or HASH not defined. Check CMakePresets.json.")
endif()

set(CHOSEN_SHERPA_URL "${SHERPA_ONNX_URL}")

if(DEFINED ENV{ARC_DEP_CACHE_DIR})
    # Obtain the filename from the URL (e.g. sherpa-onnx-v1.12.20-rv1126bp.tar.xz)
    get_filename_component(DEP_FILENAME "${SHERPA_ONNX_URL}" NAME)

    set(LOCAL_FILE_PATH "$ENV{ARC_DEP_CACHE_DIR}/${DEP_FILENAME}")

    # If file exist then it will be added as first place of candidates
    # Caution here: We will not check Hashcode, FetchContent will do instead
    #               If Hashcode is not matched, FetchContent will take warning
    if(EXISTS "${LOCAL_FILE_PATH}")
        message(STATUS "[Configuration][ThirdParty] Found local cache candidate: ${LOCAL_FILE_PATH}")
        set(CHOSEN_SHERPA_URL "file://${LOCAL_FILE_PATH}")
    else()
        message(STATUS "[Configuration][ThirdParty] Local cache not found at: ${LOCAL_FILE_PATH}, using remote URL.")
    endif()
endif()

message(STATUS "[Configuration][ThirdParty] Final URL used: ${CHOSEN_SHERPA_URL}")

FetchContent_Declare(
    sherpa_onnx_pkg
    URL      ${CHOSEN_SHERPA_URL}
    URL_HASH ${SHERPA_ONNX_HASH}
    # Download and Decompress unto the directory of "build/_deps/sherpa_onnx_pkg-src"
    # Default is FALSE
    # DOWNLOAD_NO_EXTRACT FALSE
)

# ---------------------------------
# III. Exec Downloading and Decompressing (Configure Stage)
# ---------------------------------
FetchContent_MakeAvailable(sherpa_onnx_pkg)

set(SHERPA_ROOT "${sherpa_onnx_pkg_SOURCE_DIR}")
message(STATUS "[Configuration][ThirdParty] Sherpa-Onnx mounted at: ${SHERPA_ROOT}")

# ---------------------------------
# IV. Define "IMPORTED Targets" (Point to the Dir After Decompressing)
# ---------------------------------

# --- a. Dep Lib: OnnxRuntime ---
add_library(SherpaDeps::OnnxRuntime
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(SherpaDeps::OnnxRuntime
    PROPERTIES
        IMPORTED_LOCATION
            "${SHERPA_ROOT}/lib/libonnxruntime.so"
)

# --- b. Dep Lib: C API ---
add_library(SherpaDeps::C_API
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(SherpaDeps::C_API
    PROPERTIES
        IMPORTED_LOCATION
            "${SHERPA_ROOT}/lib/libsherpa-onnx-c-api.so"
)

# --- c. Dep Lib: CArgs ---
add_library(SherpaDeps::CArgs
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(SherpaDeps::CArgs
    PROPERTIES
        IMPORTED_LOCATION
            "${SHERPA_ROOT}/lib/libcargs.so"
)

# --- d. Core Lib: C++ API ---
add_library(sherpa_onnx_lib
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(sherpa_onnx_lib
    PROPERTIES
        IMPORTED_LOCATION
            "${SHERPA_ROOT}/lib/libsherpa-onnx-cxx-api.so"
        INTERFACE_INCLUDE_DIRECTORIES
            "${SHERPA_ROOT}/include"
        INTERFACE_LINK_LIBRARIES
            "${PROJECT_NAMESPACE}::ThirdParty::LibRKNNRT;SherpaDeps::C_API;SherpaDeps::OnnxRuntime;SherpaDeps::CArgs"
)

# --- e. Create Alias (External Interface) ---
add_library(${PROJECT_NAMESPACE}::ThirdParty::SherpaOnnx
    ALIAS
        sherpa_onnx_lib
)

# ---------------------------------
# V. Install Rule (lib)
#    Will Copy from "build/_deps/..." to "install/lib"
# ---------------------------------
file(
    GLOB
        ALL_SO_FILES
            "${SHERPA_ROOT}/lib/*.so*"
)
install(
    FILES
        ${ALL_SO_FILES}
    DESTINATION
        lib
    COMPONENT
        runtime
)

# ---------------------------------
# VI. Install Rule (include)
#    Will Copy from "build/_deps/..." to "install/include"
# ---------------------------------
if(EXISTS "${SHERPA_ROOT}/include")
    install(
        DIRECTORY "${SHERPA_ROOT}/include/"
        DESTINATION include
        COMPONENT headers
    )
endif()

# ---------------------------------
# VII. Install Rule (bin)
#    Will Copy from "build/_deps/..." to "install/bin"
# ---------------------------------
if(EXISTS "${SHERPA_ROOT}/bin")
    file(
        GLOB
            ALL_BIN_FILES
                "${SHERPA_ROOT}/bin/*"
    )

    # Default to OFF (base on .env file on the top directory of repository)
    string(TOUPPER "${ARC_INSTALL_SHERPA_TOOLS}" _upper_val)
    if(_upper_val STREQUAL "ON")
        message(STATUS "ARC_INSTALL_SHERPA_TOOLS=ON SherpaOnnx bin will be installed")
        install(
            PROGRAMS
                ${ALL_BIN_FILES}
            DESTINATION
                bin/tools
            COMPONENT
                tools
        )
    endif()
endif()

# ---------------------------------
#               End
# ---------------------------------
