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

# test/unit/CMakeLists.txt
# *********************************************************************************************

# 定义模块目录名

set(MODULE_DIR "app-bin")

file(GLOB TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test-bin.cpp"
)

# sources
# 我们需要编译业务逻辑的实现文件 processor_logic.cpp
# 注意：这里应该只包含 .cpp 文件，.h 文件由编译器自动处理
file(GLOB MODULE_SOURCES
    # "${CMAKE_CURRENT_SOURCE_DIR}/processor_logic.cpp"
)

set(INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${INCLUDE_DIR}"
)

set(LINK_LIBS

)

set(LINK_DIRS

)

set(PCH_FILES

)

add_unit_test(
    NAME "${BIN_PREFIX}_${MODULE_DIR}"
    TEST_SOURCES "${TEST_SOURCES}"
    MODULE_SOURCES "${MODULE_SOURCES}"
    INCLUDE_DIRS "${INCLUDE_DIRS}"
    LINK_LIBS "${LINK_LIBS}"
    LINK_DIRS "${LINK_DIRS}"
    PCH_FILES ${PCH_FILES}
    MODULE_DIR "${MODULE_DIR}"
)
