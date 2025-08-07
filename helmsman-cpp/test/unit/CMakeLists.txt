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
