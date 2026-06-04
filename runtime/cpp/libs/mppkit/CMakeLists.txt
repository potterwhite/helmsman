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
# I. Platform Guard — Rockchip only
# ---------------------------------
if(NOT "rockchip" IN_LIST CMAKE_PLATFORM)
    message(WARNING "MPPKit: Skipping build — not a Rockchip platform")
    return()
endif()

# ---------------------------------
# II. Protection for standalone Use
# ---------------------------------
if(NOT DEFINED GLOBAL_VERSION_STRING OR "${GLOBAL_VERSION_STRING}" STREQUAL "")
    set(GLOBAL_VERSION_STRING "99.99.99")
    message(WARNING "Expected Version is missing, Using Default Version: ${GLOBAL_VERSION_STRING}")
endif()

# ---------------------------------
# II. project name
# ---------------------------------
set(PROJECT_NAME "MPPKit")
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
add_library(${PROJECT_NAME})                  # Let ${BUILD_SHARED_LIBS} determine static or shared

if(NOT DEFINED PROJECT_NAMESPACE)
    set(PROJECT_NAMESPACE "Helmsman")
endif()
add_library(${PROJECT_NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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
# MPP SDK headers are in the RK3588 SDK buildroot output.
# They are only needed by the .cpp implementation files (PRIVATE),
# not by consumers of this library (PIMPL hides them).
set(MPP_SDK_INCLUDE_DIR "" CACHE PATH "Path to MPP SDK include directory (containing rk_mpi.h)")

if("${MPP_SDK_INCLUDE_DIR}" STREQUAL "")
    set(_MPP_AUTO_PATH
        "/development/src/sdk/rk3588s-linux/buildroot/output/rockchip_rk3588/build/rockchip-mpp-develop/inc"
    )
    if(EXISTS "${_MPP_AUTO_PATH}/rk_mpi.h")
        set(MPP_SDK_INCLUDE_DIR "${_MPP_AUTO_PATH}")
        message(STATUS "MPPKit: auto-detected MPP headers at ${MPP_SDK_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "MPPKit: MPP SDK include directory not found.\n"
            "Please set MPP_SDK_INCLUDE_DIR to the directory containing rk_mpi.h.\n"
            "Example: -DMPP_SDK_INCLUDE_DIR=/path/to/rockchip-mpp-develop/inc"
        )
    endif()
endif()

# SYSTEM tells the compiler to treat these headers as "system headers",
# which suppresses warnings from them (e.g. -Wpedantic for C99 features
# in the MPP SDK). This is the standard CMake way to handle external SDKs.
target_include_directories(${PROJECT_NAME}
    SYSTEM
    PRIVATE
        ${MPP_SDK_INCLUDE_DIR}
)

# Link against librockchip_mpp (found in sysroot)
find_library(MPP_LIBRARY NAMES rockchip_mpp)
if(MPP_LIBRARY)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${MPP_LIBRARY}
    )
    message(STATUS "MPPKit: found librockchip_mpp at ${MPP_LIBRARY}")
else()
    message(FATAL_ERROR
        "MPPKit: librockchip_mpp.so not found.\n"
        "Ensure the RK3588 SDK sysroot is in CMAKE_FIND_ROOT_PATH."
    )
endif()

# ---------------------------------
# VIII. installation rules
# ---------------------------------
arc_install_library(${PROJECT_NAME} ${INCLUDE_DIR})

# ---------------------------------
#               End
# ---------------------------------
