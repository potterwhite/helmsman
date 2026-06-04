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
    message(WARNING "RKNNKit: Skipping build — not a Rockchip platform")
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
# III. project name
# ---------------------------------
set(PROJECT_NAME "RKNNKit")
project(${PROJECT_NAME}
    VERSION
        ${GLOBAL_VERSION_STRING}
    LANGUAGES
        CXX
)

# ---------------------------------
# IV. Initialize variables
#      to commonly define header folder
#      subdirectories will use this variable
# ---------------------------------
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------
# V. Add the library target using collected sources ---
# ---------------------------------
add_library(${PROJECT_NAME})                  # Let ${BUILD_SHARED_LIBS} determine static or shared

if(NOT DEFINED PROJECT_NAMESPACE)
    set(PROJECT_NAMESPACE "Helmsman")
endif()
add_library(${PROJECT_NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ---------------------------------
# VI. enter src directory
#      ** This cmd will invoke every called CMakeLists.txt in specific directories. **
#      ** And they will fill in the LIB_SOURCES **
# ---------------------------------
add_subdirectory(src)

# ---------------------------------
# VII. Expose All kinds of public Informations unto Src files
# ---------------------------------
arc_setup_system_info(${PROJECT_NAME})

# ---------------------------------
# VIII. Link Dependency
# ---------------------------------
if(NOT TARGET ${PROJECT_NAMESPACE}::Utils)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::Utils, start searching")
    find_package(${PROJECT_NAMESPACE}_Utils REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_NAMESPACE}::Utils
        ${PROJECT_NAMESPACE}::ThirdParty::LibRKNNRT
)

# ---------------------------------
# IX. installation rules
# ---------------------------------
arc_install_library(${PROJECT_NAME} ${INCLUDE_DIR})

# ---------------------------------
#               End
# ---------------------------------
