# CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

# I. Config PROJECT_NAME And C++ Standards
set(PROJECT_NAME "executableTemplate")
project(${PROJECT_NAME} CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 通常需要

# II. 追溯顶层目录
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SRC_DIR "${TOP_DIR}/src")
set(INCLUDE_DIR "${TOP_DIR}/include")

# III. 设置build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    # set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "No build type specified, defaulting to 'Debug'")
endif()
add_definitions(-DBUILD_TYPE="${CMAKE_BUILD_TYPE}")


#############################################################
# IV. Compile options Sets global compilation flags
set(BASIC_COMPILE_OPTIONS -Wall -Wextra -Werror)

# advanced & most serious harsh options let no mistakes go
set(STRINGENT_COMPILE_OPTIONS
    -Wconversion
    -Wsign-conversion
    -Wfloat-conversion
    -Wpedantic
    -Wcast-qual
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wformat=2
    -Wformat-security
    -Wformat-nonliteral
    -Wuninitialized
    -Winit-self
    -Wswitch-enum
    -Wswitch-default
    -Wmissing-include-dirs
    -Wredundant-decls
    -Wshadow
    -Wundef
    -Wdouble-promotion)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug: No optimization, with debug info
    add_compile_options(-Og -g -DDEBUG ${STRINGENT_COMPILE_OPTIONS})
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # RelWithDebInfo: Full optimization with debug info
    add_compile_options(-O2 -g -DNDEBUG ${STRINGENT_COMPILE_OPTIONS})
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    # MinSizeRel: Size optimization
    add_compile_options(-Os -DNDEBUG ${STRINGENT_COMPILE_OPTIONS})
    # add_link_options(-s)
else() # Release
    # Release: Full optimization, no debug info
    add_compile_options(-O3 -DNDEBUG ${STRINGENT_COMPILE_OPTIONS})
    # add_link_options(-s)
endif()

########################### End of Section V ##################################

# V. 设置 RPATH (关键！让可执行文件在ARM板上运行时能找到 .so 文件) $ORIGIN 表示可执行文件所在的目录。 这意味着你需要把所依赖的所有 lib*.so这些文件 和编译出来的
# bin(可执行文件)放在ARM板上的【同一个目录】下。
set(CMAKE_INSTALL_RPATH "$ORIGIN")
# 在构建时也应用 RPATH，这样你在PC上构建目录中测试（如果架构匹配）或检查RPATH时也能看到效果
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)


# ##################################################################################################
# VI. 依赖管理 (Dependency Management)
# 在这里定义你的项目所需要的所有库。
# 这样src/CMakeLists.txt就不需要知道具体的依赖了。
# 对于一个干净的模板，我们先留空。未来Helmsman项目会在这里添加OpenCV等。
set(COMMON_LIBRARIES "") # 例如: "OpenCV::opencv" "Threads::Threads"
# ##################################################################################################


# VII. 添加子目录
add_subdirectory(${SRC_DIR})

option(BUILD_TEST "Build the tests for the project" OFF)
if(BUILD_TEST STREQUAL "TRUE")
    # enable_testing()
    add_subdirectory(test)
endif()
