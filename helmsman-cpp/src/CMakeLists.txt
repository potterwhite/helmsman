
# I. Config Executable
set(BIN_NAME "${PROJECT_NAME}.bin")
add_executable(${BIN_NAME} main.cpp)

# II. Make ${PROJECT_NAME} share with src files
target_compile_definitions(${BIN_NAME} PRIVATE
    PROJECT_NAME="${PROJECT_NAME}"
)

# III. 指定头文件搜索路径
target_include_directories(${BIN_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIR}
)

# IV. 链接公共库
# 直接链接在顶层CMakeLists.txt中定义的COMMON_LIBRARIES列表
target_link_libraries(${BIN_NAME} PRIVATE
    ${COMMON_LIBRARIES}
)

# V. install rules
install(
    TARGETS ${BIN_NAME}
    DESTINATION bin
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# VI. customized cmds
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    # 获取 strip 工具的路径 CMAKE_STRIP 是 CMake 通常能找到的 strip 工具，但交叉编译时可能需要指定 find_program(CROSS_STRIP NAMES
    # ${CMAKE_C_COMPILER_TARGET}-strip aarch64-linux-gnu-strip aarch64-none-linux-gnu-strip)
    #set(CROSS_STRIP ${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-strip) # 直接使用你的工具链中的 strip
    find_program(STRIP_COMMAND NAMES ${CMAKE_CROSSCOMPILING_EMULATOR}-strip aarch64-linux-gnu-strip strip)

    if(STRIP_COMMAND)
        add_custom_command(
            TARGET ${BIN_NAME}
            POST_BUILD
            COMMAND ${STRIP_COMMAND} ${BIN_NAME}
            COMMENT "Stripping ${BIN_NAME}"
            COMMAND ls -lha ${BIN_NAME}
            COMMAND file ${BIN_NAME})
    else()
        message(WARNING "Strip command not found. Executable will not be stripped automatically.")
    endif()
endif()

