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

# ---------------------------------
# I. Protection for standalone Use
# ---------------------------------
if(NOT DEFINED GLOBAL_VERSION_STRING OR "${GLOBAL_VERSION_STRING}" STREQUAL "")
    set(GLOBAL_VERSION_STRING "99.99.99")
    message(WARNING "Expected Version is missing, Using Default Version: ${GLOBAL_VERSION_STRING}")
endif()

# ---------------------------------
# II. project name
# ---------------------------------
set(PROJECT_NAME "Matting_Server")
project(${PROJECT_NAME}
    VERSION
        ${GLOBAL_VERSION_STRING}
    LANGUAGES
        CXX
)

# ---------------------------------
# III. Initialize variables
#      to commonly define header folder
#      subdirectories will use this variable
# ---------------------------------
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------
# IV. Add the library target using collected sources ---
# ---------------------------------
add_executable(${PROJECT_NAME})

if(NOT DEFINED PROJECT_NAMESPACE)
    set(PROJECT_NAMESPACE "Helmsman")
endif()
add_executable(${PROJECT_NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ---------------------------------
# V. enter src directory
#      ** This cmd will invoke every called CMakeLists.txt in specific directories. **
#      ** And they will fill in the LIB_SOURCES **
# ---------------------------------
add_subdirectory(src)

# ---------------------------------
# VI. Expose All kinds of public Informations unto Src files
#     1. Generates 'system-info.h' from the global template.
#     2. Adds include directories (Source include + Generated include).
#     3. Convention: If "<TargetTopDir>/include" directory exists, use it.
#     4. Convention: If include/<TargetName>/pch.h exists, use it.
#     5. Sets target properties (VERSION, SOVERSION, OUTPUT_NAME).
#     6. Injects the PROJECT_NAME macro definition.
#     7. Links against the common configuration target "arc_base_settings".
# ---------------------------------
arc_setup_system_info(${PROJECT_NAME})

# ---------------------------------
# VII. Link Dependency
# ---------------------------------
if(NOT TARGET ${PROJECT_NAMESPACE}::Utils)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::Utils, start searching")
    find_package(${PROJECT_NAMESPACE}_Utils REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::Network)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::Network, start searching")
    find_package(${PROJECT_NAMESPACE}_Network REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::CVKit)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::CVKit, start searching")
    find_package(${PROJECT_NAMESPACE}_CVKit REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::Runtime)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::Runtime, start searching")
    find_package(${PROJECT_NAMESPACE}_Runtime REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::RGAKit)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::RGAKit, start searching")
    find_package(${PROJECT_NAMESPACE}_RGAKit REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::DmaKit)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::DmaKit, start searching")
    find_package(${PROJECT_NAMESPACE}_DmaKit REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::DRMKit)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::DRMKit, start searching")
    find_package(${PROJECT_NAMESPACE}_DRMKit REQUIRED)
endif()

if(NOT TARGET ${PROJECT_NAMESPACE}::MPPKit)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::MPPKit, start searching")
    find_package(${PROJECT_NAMESPACE}_MPPKit REQUIRED)
endif()

# FFmpeg: optional, needed for MPP hardware decode path
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG_AVFORMAT QUIET IMPORTED_TARGET libavformat)
    pkg_check_modules(FFMPEG_AVCODEC QUIET IMPORTED_TARGET libavcodec)
    pkg_check_modules(FFMPEG_AVUTIL QUIET IMPORTED_TARGET libavutil)

    if(FFMPEG_AVFORMAT_FOUND)
        message(NOTICE "FFmpeg(libavformat) found via pkg-config")
    else()
        message(FATAL_ERROR "FFmpeg(libavformat) not found (MPP hardware decode path disabled)")
    endif()

    if(FFMPEG_AVCODEC_FOUND)
        message(NOTICE "FFmpeg(libavcodec) found via pkg-config")
    else()
        message(FATAL_ERROR "FFmpeg(libavcodec) not found (MPP hardware decode path disabled)")
    endif()

    if(FFMPEG_AVUTIL_FOUND)
        message(NOTICE "FFmpeg(libavutil) found via pkg-config")
    else()
        message(FATAL_ERROR "FFmpeg(libavutil) not found (MPP hardware decode path disabled)")
    endif()

else()
    message(FATAL_ERROR "pkg-config not found (FFmpeg detection skipped)")
endif()

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_NAMESPACE}::Utils
        ${PROJECT_NAMESPACE}::Network
        ${PROJECT_NAMESPACE}::CVKit
        ${PROJECT_NAMESPACE}::Runtime
        ${PROJECT_NAMESPACE}::RGAKit
        ${PROJECT_NAMESPACE}::DmaKit
        ${PROJECT_NAMESPACE}::DRMKit
        FFMPEG_AVCODEC
        FFMPEG_AVFORMAT
        FFMPEG_AVUTIL
)

# ---------------------------------
# VIII. installation rules
# ---------------------------------
arc_install_executable(${PROJECT_NAME})

# ---------------------------------
#               End
# ---------------------------------
