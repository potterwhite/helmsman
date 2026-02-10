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

# third_party/librknnrt/CMakeLists.txt
project(librknnrt-download NONE)          # NONE means current project does not need to be compiled

# ---------------------------------
# I. Import "FetchContent" Module
# ---------------------------------
include(FetchContent)

message(STATUS "[Configuration][ThirdParty] Preparing librknnrt...")

# ---------------------------------
# II. Declare Download Detail
# ---------------------------------
# These variables come from CMakePresets.json
if(NOT DEFINED LIBRKNNRT_URL OR NOT DEFINED LIBRKNNRT_HASH)
    message(Warning "[Configuration][ThirdParty] LIBRKNNRT_URL or HASH not defined. Check CMakePresets.json.")
endif()

set(CHOSEN_LIBRKNNRT_URL "${LIBRKNNRT_URL}")

if(DEFINED ENV{ARC_DEP_CACHE_DIR})
    # Obtain the filename from the URL (e.g. librknnrt.so)
    get_filename_component(DEP_FILENAME "${LIBRKNNRT_URL}" NAME)

    set(LOCAL_FILE_PATH "$ENV{ARC_DEP_CACHE_DIR}/${DEP_FILENAME}")

    # If file exist then it will be added as first place of candidates
    # Caution here: We will not check Hashcode, FetchContent will do instead
    #               If Hashcode is not matched, FetchContent will take warning
    if(EXISTS "${LOCAL_FILE_PATH}")
        message(STATUS "[Configuration][ThirdParty] Found local cache candidate: ${LOCAL_FILE_PATH}")
        set(CHOSEN_LIBRKNNRT_URL "file://${LOCAL_FILE_PATH}")
    else()
        message(STATUS "[Configuration][ThirdParty] Local cache not found at: ${LOCAL_FILE_PATH}, using remote URL.")
    endif()
endif()

message(STATUS "[Configuration][ThirdParty] Final URL used: ${CHOSEN_LIBRKNNRT_URL}")

FetchContent_Declare(
    librknnrt_pkg
    URL      ${CHOSEN_LIBRKNNRT_URL}
    URL_HASH ${LIBRKNNRT_HASH}
    # ----------
    # Download and Decompress unto the directory of "build/_deps/librknnrt_pkg-src"
    # Default is FALSE
    # DOWNLOAD_NO_EXTRACT FALSE
    # ----------
)

# ---------------------------------
# III. Exec Downloading and Decompressing (Configure Stage)
# ---------------------------------
FetchContent_MakeAvailable(librknnrt_pkg)

# ${librknnrt_pkg_SOURCE_DIR} = the Path after decompressed
set(LIBRKNNRT_ROOT "${librknnrt_pkg_SOURCE_DIR}")
message(STATUS "[Configuration][ThirdParty] LibRKNNRT mounted at: ${LIBRKNNRT_ROOT}")

# ---------------------------------
# IV. Define "IMPORTED Targets" (Point to the Dir After Decompressing)
# ---------------------------------

# --- a. Core Lib: C++ API ---
add_library(librknnrt_lib
    SHARED
    IMPORTED
    GLOBAL
)
set_target_properties(librknnrt_lib
    PROPERTIES
        IMPORTED_LOCATION
            "${LIBRKNNRT_ROOT}/lib/librknnrt.so"
)

# --- b. Create Alias (External Interface) ---
add_library(${PROJECT_NAMESPACE}::ThirdParty::LibRKNNRT
    ALIAS
        librknnrt_lib
)

# ---------------------------------
# V. Install Rule (lib)
#    Will Copy from "build/_deps/..." to "install/lib"
# ---------------------------------
file(
    GLOB
        ALL_SO_FILES
            "${LIBRKNNRT_ROOT}/lib/*.so*"
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
#               End
# ---------------------------------
