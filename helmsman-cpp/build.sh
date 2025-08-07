#!/bin/bash

1_0_load_env() {
    set -e
    MAKE_VERBOSE_OPTION=""
    if [ "$V" != "" ]; then
        set -x
        MAKE_VERBOSE_OPTION="VERBOSE=1"
    fi

    REVISION_CODE="v1.0.0"
    BASH_SCRIPT_PATH="$(realpath ${BASH_SOURCE[0]})"
    BASH_SCRIPT_DIR="$(dirname ${BASH_SCRIPT_PATH})"
    BUILD_TOP_DIR="${BASH_SCRIPT_DIR}/build"
    INSTALL_TOP_DIR="${BASH_SCRIPT_DIR}/install"
    INSTALL_INC_DIR="${INSTALL_TOP_DIR}/include"
    INSTALL_BIN_DIR="${INSTALL_TOP_DIR}/bin"

    # 5. sysroot places
    CMAKE_SYSROOT="/development/docker_volumes/src/sdk/rk3588s-linux/buildroot/output/rockchip_rk3588/host/aarch64-buildroot-linux-gnu/sysroot"

    SRC_TOP_DIR="${BASH_SCRIPT_DIR}/src"

    # MODIFICATION START
    TOOLCHAINS_DIR="${BASH_SCRIPT_DIR}/toolchains"
    # ARGC and ARGC_2ND_OPTION are deprecated in favor of a more flexible approach
    COMMAND=""
    PLATFORM=""
    # MODIFICATION END

    ARGC="0"
    ARGC_2ND_OPTION=""
    MACRO_OPTION_1_CLEAN="clean"
    MACRO_OPTION_2_BUILD="build"
    MACRO_OPTION_3_BUILD_AFTER_CLEAN="cb"
    MACRO_OPTION_4_INSTALL="install"
    MACRO_OPTION_5_TEST="test"
}

1_1_arguments_validation() {
    if [ "$#" -eq 0 ]; then
        1_2_helper_print
        exit 1
    fi

    # The first argument is always the command
    COMMAND=$1
    shift # Consume the first argument

    # The next optional argument could be the platform for 'build' and 'cb' commands
    if [ "$COMMAND" = "${MACRO_OPTION_2_BUILD}" ] || [ "$COMMAND" = "${MACRO_OPTION_3_BUILD_AFTER_CLEAN}" ]; then
        if [ "$#" -gt 0 ]; then
            PLATFORM=$1
            shift
        else
            PLATFORM="native" # Default to native build if no platform is specified
        fi
    fi

    # Check if the command is valid
    case "$COMMAND" in
        "${MACRO_OPTION_1_CLEAN}"|"${MACRO_OPTION_2_BUILD}"|"${MACRO_OPTION_3_BUILD_AFTER_CLEAN}"|"${MACRO_OPTION_4_INSTALL}"|"${MACRO_OPTION_5_TEST}")
            # Valid command, do nothing
            ;;
        *)
            echo "Error: Unknown command '$COMMAND'"
            1_2_helper_print
            exit 1
            ;;
    esac

}


1_2_helper_print() {
    echo
    echo "Usage: ./build.sh <command> [platform]"
    echo
    echo "Commands:"
    echo "  clean                - Remove build and install directories"
    echo "  build [platform]     - Configure and build the project for a specific platform"
    echo "  cb [platform]        - Clean, then build and install for a specific platform"
    echo "  install              - Install the already built project"
    echo "  test                 - Build and run tests (for native platform only)"
    echo
    echo "Available platforms:"
    echo "  native(default)               - Build for the host machine (x86_64, default)"

    # Dynamically find and list available toolchains
    if [ -d "${TOOLCHAINS_DIR}" ]; then
        for toolchain_file in $(find "${TOOLCHAINS_DIR}" -type f -name "*.cmake"); do
            platform_name=$(basename "${toolchain_file}" .cmake)
            echo "  ${platform_name}         - Cross-compile using ${platform_name}.cmake"
        done
    fi
    echo
}

2_1_preparation() {
    mkdir -p ${BUILD_TOP_DIR}
    echo "mkdir -p ${BUILD_TOP_DIR}"
    mkdir -p ${INSTALL_INC_DIR}
    echo "mkdir -p ${INSTALL_INC_DIR}"
    mkdir -p ${INSTALL_BIN_DIR}
    echo "mkdir -p ${INSTALL_BIN_DIR}"
}

2_2_clean_all() {
    rm -rf ${BUILD_TOP_DIR}
    echo "Removed ${BUILD_TOP_DIR}"
    rm -rf ${INSTALL_TOP_DIR}
    echo "Removed ${INSTALL_TOP_DIR}"
}

2_3_cmake_generate_makefile() {
    local CMAKE_ARGS=(
        -S "${BASH_SCRIPT_DIR}"
        -B "${BUILD_TOP_DIR}"
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_TOP_DIR}"
    )

    if [ -n "$PLATFORM" ] && [ "$PLATFORM" != "native" ]; then
        local TOOLCHAIN_FILE="${TOOLCHAINS_DIR}/${PLATFORM}.cmake"
        if [ -f "${TOOLCHAIN_FILE}" ]; then
            echo "Using toolchain file for platform '${PLATFORM}': ${TOOLCHAIN_FILE}"
            CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
        else
            echo "Error: Toolchain file for platform '${PLATFORM}' not found at '${TOOLCHAIN_FILE}'"
            1_2_helper_print
            exit 1
        fi
    else
        echo "Configuring for native build (x86_64)."
        # we will add the vcpkg toolchain file here for native builds
        # For a generic template, this is fine.
    fi

    cmake "${CMAKE_ARGS[@]}"

    ################################ original obsoleted #################################
    # cmake \
    #     -S ${BASH_SCRIPT_DIR} \
    #     -B ${BUILD_TOP_DIR} \
    #     "-DCMAKE_INSTALL_PREFIX=${INSTALL_TOP_DIR}"
}

2_4_make_execution() {
    make -C ${BUILD_TOP_DIR} ${MAKE_VERBOSE_OPTION}
}

2_6_install_target() {
    cmake --install ${BUILD_TOP_DIR}

    # 2_5_copy_dependencies

    # rsync -av --progress ${SRC_TOP_DIR}/* ${INSTALL_INC_DIR}
    # echo "Installed headers to ${INSTALL_INC_DIR}"

    # rsync -av --progress ${BUILD_TOP_DIR}/${FINAL_BIN_NAME} ${INSTALL_BIN_DIR}
    # echo "Installed library to ${INSTALL_BIN_DIR}"

    # rm -rf ${INSTALL_PLACE_2ND_DIR}

    # rsync -av --progress ${INSTALL_TOP_DIR}/* ${INSTALL_PLACE_2ND_DIR}
    # echo "Installed whole install dir to ${INSTALL_PLACE_2ND_DIR}"

    echo
    echo "Installation completed successfully."
}

2_7_make_test() {
    # 确保构建目录存在
    2_1_preparation

    # 生成 CMake 构建文件，启用 BUILD_TEST
    cmake \
        -S ${BASH_SCRIPT_DIR} \
        -B ${BUILD_TOP_DIR} \
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_TOP_DIR}" \
        "-DBUILD_TEST=TRUE"

    # 仅构建测试目标（假设 test/CMakeLists.txt 定义了目标，例如 'test_main'）
    make -C ${BUILD_TOP_DIR}/test ${MAKE_VERBOSE_OPTION}

    echo "测试构建完成。"
}

3_0_exec_as_requirement() {
    case "$COMMAND" in
        "${MACRO_OPTION_1_CLEAN}")
            2_2_clean_all
            2_1_preparation
            ;;
        "${MACRO_OPTION_2_BUILD}")
            2_3_cmake_generate_makefile
            2_4_make_execution
            ;;
        "${MACRO_OPTION_3_BUILD_AFTER_CLEAN}")
            2_2_clean_all
            2_1_preparation
            2_3_cmake_generate_makefile
            2_4_make_execution
            2_6_install_target
            ;;
        "${MACRO_OPTION_4_INSTALL}")
            2_6_install_target
            ;;
        "${MACRO_OPTION_5_TEST}")
            2_7_make_test
            2_6_install_target
            ;;
    esac
}

main() {
    1_0_load_env
    1_1_arguments_validation "$@"
    3_0_exec_as_requirement
}

main "$@"
