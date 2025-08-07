# test/CMakeLists.txt

# GTest 配置
set(GTEST_ROOT ${CMAKE_SYSROOT}/usr
        CACHE PATH "GTest root directory"
)
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/include"
        # ${SRC_DIR}
        CACHE PATH "GTest include directory"
)

# libgtest_main must be put before libgtest
set(GTEST_LIBRARY ${GTEST_ROOT}/lib/libgtest.a
        CACHE PATH "GTest library")
set(GTEST_MAIN_LIBRARY ${GTEST_ROOT}/lib/libgtest_main.a
        CACHE PATH "GTest main library")

# --- GMock (新增部分) ---
set(GMOCK_LIBRARY ${GTEST_ROOT}/lib/libgmock.a
        CACHE PATH "GMock library")
set(GMOCK_MAIN_LIBRARY ${GTEST_ROOT}/lib/libgmock_main.a
        CACHE PATH "GMock main library") # gmock_main通常包含了gtest_main，但显式链接更安全

# ========================================================
# 定义一个更模块化的函数来创建测试目标
function(add_unit_test)
        # 定义支持的命名参数
        set(options "")
        set(oneValueArgs NAME MODULE_DIR)
        set(multiValueArgs TEST_SOURCES MODULE_SOURCES INCLUDE_DIRS LINK_LIBS LINK_DIRS PCH_FILES)

        # ---------------------------------------------------------
        # a. 解析命名参数
        cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        # ---------------------------------------------------------
        # b. 验证必要参数
        if(NOT DEFINED TEST_NAME)
                message(FATAL_ERROR "必须提供 NAME 参数")
        endif()
        # ---------------------------------------------------------

        # *********************************************************
        # i. 设置目标名称
        set(TARGET_NAME ${TEST_NAME})

        # *********************************************************
        # ii. 设置默认值
        if(NOT DEFINED TEST_MODULE_DIR)
                set(TEST_MODULE_DIR "tests")
        endif()

        # *********************************************************
        # iii. 创建测试可执行文件
        add_executable(${TARGET_NAME}
                ${TEST_TEST_SOURCES}
                ${TEST_MODULE_SOURCES}
        )

        # *********************************************************
        # iv. 根据模块特定需求添加预编译头文件
        if(TEST_PCH_FILES)
                target_precompile_headers(${TARGET_NAME} PUBLIC ${TEST_PCH_FILES})
        elseif(EXISTS "${TOP_DIR}/src/pch.h")
                # 只添加基本PCH文件，避免不必要的依赖
                target_precompile_headers(${TARGET_NAME} PUBLIC "${TOP_DIR}/src/pch.h")
        endif()

        # *********************************************************
        # v. 链接依赖库（始终需要的库）
        # target_link_libraries(${TARGET_NAME} PUBLIC
        #         ${GTEST_LIBRARY}
        #         ${GTEST_MAIN_LIBRARY}
        #         ${GMOCK_LIBRARY}
        #         ${GMOCK_MAIN_LIBRARY}
        #         ${MODULE_NAME_COMMON}
        # )

        target_link_libraries(${TARGET_NAME} PUBLIC
                # gmock_main 调用 gmock 和 gtest 的功能
                ${GMOCK_MAIN_LIBRARY}
                # gmock 调用 gtest 的功能
                ${GMOCK_LIBRARY}
                # gtest_main 调用 gtest 的功能
                ${GTEST_MAIN_LIBRARY}
                # gtest 是核心实现，被大家依赖，放在最后
                ${GTEST_LIBRARY}
                # ${MODULE_NAME_COMMON} # 这个变量看起来没有定义，先注释掉
        )

        # 添加额外的链接库
        if(TEST_LINK_LIBS)
                target_link_libraries(${TARGET_NAME} PUBLIC ${TEST_LINK_LIBS})
        endif()

        if(TEST_LINK_DIRS)
                target_link_directories(${TARGET_NAME} PUBLIC
                        ${TEST_LINK_DIRS}
                )
        endif()

        # *********************************************************
        # vi. 包含目录
        target_include_directories(${TARGET_NAME} PRIVATE
                ${GTEST_INCLUDE_DIR}
        )

        # 添加额外的包含目录
        if(TEST_INCLUDE_DIRS)
                target_include_directories(${TARGET_NAME} PRIVATE
                        ${TEST_INCLUDE_DIRS}
                )
        endif()

        # *********************************************************
        # vii. 按模块分类安装 install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION
        # ${CMAKE_BINARY_DIR}/bin/${TEST_MODULE_DIR})
        install(TARGETS ${TARGET_NAME}
                RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/${TEST_MODULE_DIR})

        # *********************************************************
        # viii. 设置运行时输出目录 set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        # "${CMAKE_BINARY_DIR}/bin/${TEST_MODULE_DIR}")
        set_target_properties(
                ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                "${CMAKE_INSTALL_PREFIX}/bin/${TEST_MODULE_DIR}")

        # *********************************************************
        # ix. 添加到 CTest
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

# *********************************************************************************************

add_subdirectory(unit)
# add_subdirectory(integration)

# enabling test is only one time execution
enable_testing()
