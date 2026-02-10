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

# third_party/onnxruntime/CMakeLists.txt
project(onnxruntime-download NONE)          # NONE means current project does not need to be compiled

# ---------------------------------
# I. Import "FetchContent" Module
# ---------------------------------
include(FetchContent)

message(STATUS "[Configuration][ThirdParty] Preparing ONNXRuntime...")

# ---------------------------------
# II. Declare Download Detail
# ---------------------------------
# These variables come from CMakePresets.json
if(NOT DEFINED ORT_URL OR NOT DEFINED ORT_HASH)
    message(Warning "[Configuration][ThirdParty] ORT_URL or HASH not defined. Check CMakePresets.json.")
endif()

set(CHOSEN_PKG_URL "${ORT_URL}")

if(DEFINED ENV{ARC_DEP_CACHE_DIR})
    # Obtain the filename from the URL (e.g. onnxruntime-linux-x64-1.6.0.tgz)
    get_filename_component(DEP_FILENAME "${ORT_URL}" NAME)

    set(LOCAL_FILE_PATH "$ENV{ARC_DEP_CACHE_DIR}/${DEP_FILENAME}")

    # If file exist then it will be added as first place of candidates
    # Caution here: We will not check Hashcode, FetchContent will do instead
    #               If Hashcode is not matched, FetchContent will take warning
    if(EXISTS "${LOCAL_FILE_PATH}")
        message(STATUS "[Configuration][ThirdParty] Found local cache candidate: ${LOCAL_FILE_PATH}")
        set(CHOSEN_PKG_URL "file://${LOCAL_FILE_PATH}")
    else()
        message(STATUS "[Configuration][ThirdParty] Local cache not found at: ${LOCAL_FILE_PATH}, using remote URL.")
    endif()
endif()

message(STATUS "[Configuration][ThirdParty] Final URL used: ${CHOSEN_PKG_URL}")

FetchContent_Declare(
    onnxruntime_pkg
    URL      ${CHOSEN_PKG_URL}
    URL_HASH ${ORT_HASH}
    # Download and Decompress unto the directory of "build/_deps/onnxruntime_pkg-src"
    # Default is FALSE
    # DOWNLOAD_NO_EXTRACT FALSE
)

# ---------------------------------
# III. Exec Downloading and Decompressing (Configure Stage)
# ---------------------------------
FetchContent_MakeAvailable(onnxruntime_pkg)

set(ORT_ROOT "${onnxruntime_pkg_SOURCE_DIR}")
message(STATUS "[Configuration][ThirdParty] ONNXRuntime mounted at: ${ORT_ROOT}")

# ---------------------------------
# IV. Define "IMPORTED Targets" (Point to the Dir After Decompressing)
# ---------------------------------

# --- d. Core Lib: C++ API ---
add_library(prebuit_onnxruntime_lib
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(prebuit_onnxruntime_lib
    PROPERTIES
        IMPORTED_LOCATION
            "${ORT_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES
            "${ORT_ROOT}/include"
)

# --- e. Create Alias (External Interface) ---
add_library(${PROJECT_NAMESPACE}::ThirdParty::ONNXRuntime
    ALIAS
        prebuit_onnxruntime_lib
)

# ---------------------------------
# V. Install Rule (lib)
#    Will Copy from "build/_deps/..." to "install/lib"
# ---------------------------------
file(
    GLOB
        ALL_SO_FILES
            "${ORT_ROOT}/lib/*.so*"
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
if(EXISTS "${ORT_ROOT}/include")
    install(
        DIRECTORY
            "${ORT_ROOT}/include/"
        DESTINATION
            include
        COMPONENT
            headers
    )
endif()

# ---------------------------------
# VII. Install Rule (bin)
#    Will Copy from "build/_deps/..." to "install/bin"
# ---------------------------------
#if(EXISTS "${ORT_ROOT}/bin")
#    file(
#        GLOB
#            ALL_BIN_FILES
#                "${ORT_ROOT}/bin/*"
#    )
#endif()

# ---------------------------------
#               End
# ---------------------------------
