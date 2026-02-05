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
set(PROJECT_NAME "ASR_Server")
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
    set(PROJECT_NAMESPACE "ArcForge")
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

if(NOT TARGET ${PROJECT_NAMESPACE}::ASREngine)
    message(STATUS "Can`t find ${PROJECT_NAMESPACE}::ASREngine, start searching")
    find_package(${PROJECT_NAMESPACE}_ASREngine REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_NAMESPACE}::Utils
        ${PROJECT_NAMESPACE}::Network
        ${PROJECT_NAMESPACE}::ASREngine
        ${PROJECT_NAMESPACE}::ThirdParty::SherpaOnnx
)

# ---------------------------------
# VIII. installation rules
# ---------------------------------
arc_install_executable(${PROJECT_NAME})

# ---------------------------------
#               End
# ---------------------------------
