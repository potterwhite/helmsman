# Copyright (c) 2026 PotterWhite
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
set(PROJECT_NAME "DRMKit")
project(${PROJECT_NAME}
    VERSION
        ${GLOBAL_VERSION_STRING}
    LANGUAGES
        CXX
)

# ---------------------------------
# III. Initialize variables
# ---------------------------------
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------
# IV. Add the library target ---
# ---------------------------------
add_library(${PROJECT_NAME})

if(NOT DEFINED PROJECT_NAMESPACE)
    set(PROJECT_NAMESPACE "ArcForge")
endif()
add_library(${PROJECT_NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ---------------------------------
# V. enter src directory
# ---------------------------------
add_subdirectory(src)

# ---------------------------------
# VI. Public information / pch / system-info
# ---------------------------------
arc_setup_system_info(${PROJECT_NAME})

# ---------------------------------
# VII. Link Dependency — libdrm
#
# libdrm headers usually live under <sysroot>/usr/include/libdrm/.
# Find them via pkg-config when available, else fall back to a path probe.
# Only the .cpp implementation needs them; consumers see only DrmDisplay.
# ---------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBDRM QUIET libdrm)
endif()

if(LIBDRM_FOUND)
    target_include_directories(${PROJECT_NAME}
        SYSTEM
        PRIVATE
            ${LIBDRM_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${LIBDRM_LIBRARIES}
    )
    message(STATUS "DRMKit: libdrm found via pkg-config (${LIBDRM_VERSION})")
else()
    # Fallback: probe common sysroot location and link -ldrm directly
    find_path(LIBDRM_INCLUDE_HINT drm.h
        PATH_SUFFIXES libdrm
    )
    if(LIBDRM_INCLUDE_HINT)
        target_include_directories(${PROJECT_NAME}
            SYSTEM
            PRIVATE
                ${LIBDRM_INCLUDE_HINT}
        )
        message(STATUS "DRMKit: libdrm headers at ${LIBDRM_INCLUDE_HINT}")
    else()
        message(FATAL_ERROR
            "DRMKit: libdrm headers not found.\n"
            "Install libdrm-dev in the sysroot or set CMAKE_FIND_ROOT_PATH."
        )
    endif()

    find_library(LIBDRM_LIBRARY NAMES drm)
    if(LIBDRM_LIBRARY)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBDRM_LIBRARY})
        message(STATUS "DRMKit: libdrm.so at ${LIBDRM_LIBRARY}")
    else()
        message(FATAL_ERROR "DRMKit: libdrm.so not found.")
    endif()
endif()

# ---------------------------------
# VIII. installation rules
# ---------------------------------
arc_install_library(${PROJECT_NAME} ${INCLUDE_DIR})

# ---------------------------------
#               End
# ---------------------------------
