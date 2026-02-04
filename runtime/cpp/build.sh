#!/bin/bash
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


# ==============================================================================
# Pure CMake Presets Wrapper (Fixed)
# ==============================================================================

1_1_load_env() {
    set -e

    BASH_SCRIPT_PATH="$(realpath ${BASH_SOURCE[0]})"
    BASH_SCRIPT_DIR="$(dirname ${BASH_SCRIPT_PATH})"
    REPO_DIR="${BASH_SCRIPT_DIR}"

    # --- [NEW] Load .env file if it exists ---
    # This exports all variables defined in .env to the current shell
    if [ -f "${REPO_DIR}/.env" ]; then
        set -a  # Automatically export all variables
        source "${REPO_DIR}/.env"
        set +a
    else
        echo ">> [Fatal] .env file not found in repository root."
        echo -e "\tPlease create one based on .env.example. You can refer to the README for guidance."
        exit
    fi

    # --------------------------------------------------------------------------
    # [Shell 语法详解] : "${VAR:=VALUE}"
    # --------------------------------------------------------------------------
    # 这行代码的作用是：如果变量 ARC_INSTALL_SHERPA_TOOLS 未定义或为空，
    # 则将其赋值为默认值 "OFF"。
    #
    # 语法拆解：
    # 1. ${VAR:=VALUE} (参数扩展):
    #    这是 Bash 的赋值扩展语法。
    #    - 如果 VAR 有值，表达式返回 VAR 的值。
    #    - 如果 VAR 为空或未定义，Bash 会将 VALUE (此处为 "OFF") **赋值** 给 VAR，
    #      并返回这个新值。
    #    注意与 ${VAR:-VALUE} 的区别：":-" 只返回默认值但不赋值，":=" 会修改变量本身。
    #
    # 2. "" (双引号):
    #    防止展开后的值包含空格导致语法错误。虽然 "OFF" 没空格，但这是一种防御性编程习惯。
    #
    # 3. : (空命令/冒号):
    #    这是一个内建命令，相当于 "不做任何事，只返回 true"。
    #    - 为什么要加它？
    #      如果不加冒号，Shell 会完成变量赋值后，尝试把结果 ("OFF") 当作一个命令去执行。
    #      这会导致报错：'bash: OFF: command not found'。
    #    - 加上冒号后，Shell 完成变量赋值扩展，然后把结果传给空命令，完美结束。
    # --------------------------------------------------------------------------
    : "${ARC_INSTALL_SHERPA_TOOLS:="OFF"}"

    # Defaults
    COMMAND=""
    PLATFORM="native"
    BUILD_TYPE="release"
    LIB_TYPE="shared"

    # Globals derived later
    PRESET_NAME=""
    BUILD_DIR=""
}

1_2_interactive_wizard() {
    local timeout=10

    echo
    echo ">> [Interactive Mode] Wizard initiated."
    echo ">> Defaults will be auto-selected in ${timeout} seconds (or press Enter)..."
    echo

    # ---------------------------------------
    # Step 0: Command Selection
    # Only ask if COMMAND is not already provided via CLI
    # ---------------------------------------
    if [ -z "$COMMAND" ]; then
        echo "---------------------------------------"
        echo "Select Action:"
        echo "  1) cb       (Clean, Build & Install) [Default]"
        echo "  2) build    (Incremental Build)"
        echo "  3) install  (Install only)"
        echo "  4) clean    (Clean current preset)"
        echo "  5) cleanall (Remove ALL build/install dirs)"
        echo "  6) test     (Run tests)"
        echo "  7) list     (List presets)"
        echo "---------------------------------------"
        read -t ${timeout} -p "Enter choice [1-7]: " CMD_INPUT || true
        echo

        case "$CMD_INPUT" in
            "2") COMMAND="build" ;;
            "3") COMMAND="install" ;;
            "4") COMMAND="clean" ;;
            "5") COMMAND="cleanall" ;;
            "6") COMMAND="test" ;;
            "7") COMMAND="list" ;;
            *)   COMMAND="cb" ;; # Default
        esac
        echo ">> Selected Action:   ${COMMAND}"
    else
        echo ">> Action provided:   ${COMMAND}"
    fi


    # ---------------------------------------
    # Step 1: Platform Selection
    # ---------------------------------------
    echo "---------------------------------------"
    echo "Select Platform:"
    echo "  1) native   (Default)"
    echo "  2) rv1126bp"
    echo "  3) aarch64"
    echo "---------------------------------------"
    read -t ${timeout} -p "Enter choice [1-3]: " PLAT_INPUT || true
    echo

    case "$PLAT_INPUT" in
        "2") PLATFORM="rv1126bp" ;;
        "3") PLATFORM="aarch64" ;;
        *)   ;; # Keep default (native)
    esac
    echo ">> Selected Platform: ${PLATFORM}"


    # ---------------------------------------
    # Step 2: Build Type Selection
    # ---------------------------------------
    echo "---------------------------------------"
    echo "Select Build Type:"
    echo "  1) Release  (Default)"
    echo "  2) Debug"
    echo "---------------------------------------"
    read -t ${timeout} -p "Enter choice [1-2]: " TYPE_INPUT || true
    echo

    case "$TYPE_INPUT" in
        "2") BUILD_TYPE="debug" ;;
        *)   ;; # Keep default (release)
    esac
    echo ">> Selected Build Type: ${BUILD_TYPE}"


    # ---------------------------------------
    # Step 3: Library Type Selection
    # ---------------------------------------
    echo "---------------------------------------"
    echo "Select Library Type:"
    echo "  1) Shared   (Default)"
    echo "  2) Static"
    echo "---------------------------------------"
    read -t ${timeout} -p "Enter choice [1-2]: " LIB_INPUT || true
    echo

    case "$LIB_INPUT" in
        "2") LIB_TYPE="static" ;;
        *)   ;; # Keep default (shared)
    esac
    echo ">> Selected Lib Type:   ${LIB_TYPE}"
    echo
}

1_3_helper_print() {
    echo -e "\n\033[1;33mUsage: $0 <command> [options...]\033[0m"
    echo ""
    echo -e "\033[1mCommands:\033[0m"
    echo "  build                - Configure and Compile (Incremental build)"
    echo "  cb                   - Clean, Build AND Install (Fresh build)"
    echo "  clean                - Remove build directory for current preset"
    echo "  cleanall             - Remove ALL build and install directories"
    echo "  install              - Install the built targets"
    echo "  test                 - Run tests"
    echo "  list                 - List all available CMake Presets"
    echo ""
    echo -e "\033[1mOptions (Order does not matter):\033[0m"
    echo "  <platform>           - native | rv1126bp  (Default: native)"
    echo "  debug / release      - Build Type         (Default: release)"
    echo "  static / shared      - Library Type       (Default: shared)"
    echo ""
    echo -e "\033[1mExamples:\033[0m"
    echo "  # 1. Default Build (Host PC, Release, Shared Libs)"
    echo "     $0 build"
    echo ""
    echo "  # 2. Cross-compile for RV1126 (Release, Shared)"
    echo "     $0 build rv1126bp"
    echo ""
    echo "  # 3. Build Debug version for Host PC"
    echo "     $0 build debug"
    echo ""
    echo "  # 4. Build Static Libraries (Host PC)"
    echo "     $0 build static"
    echo ""
    echo "  # 5. Full Rebuild & Install for RV1126 (Debug mode, Static Libs)"
    echo "     $0 cb rv1126bp debug static"
    echo ""
    echo "  # 6. Just Clean the RV1126 build files"
    echo "     $0 clean rv1126bp"
    echo ""
}

2_1_arguments_parsing() {
    # if [ "$#" -eq 0 ]; then
    #     1_3_helper_print
    #     exit 1
    # fi

    # COMMAND=$1
    # shift

    # Check if there are remaining arguments.
    # If YES ($# > 0): Parse them as before (Command Line Mode).
    # If NO  ($# == 0): Trigger Interactive Wizard.
    if [ "$#" -gt 0 ]; then
        COMMAND=$1
        shift

        # --- Existing Logic: Loop through arguments ---
        while [ "$#" -gt 0 ]; do
            case "$1" in
                "debug"|"Debug"|"DEBUG")
                    BUILD_TYPE="debug"
                    ;;
                "release"|"Release"|"RELEASE")
                    BUILD_TYPE="release"
                    ;;
                "static"|"Static"|"STATIC")
                    LIB_TYPE="static"
                    ;;
                "shared"|"Shared"|"SHARED")
                    LIB_TYPE="shared"
                    ;;
                *)
                    PLATFORM=$1
                    ;;
            esac
            shift
        done
    else
        # --- New Logic: No args provided, ask the user ---
        # Exclude 'list' and 'cleanall' from wizard, as they usually don't need params
        # if [ "$COMMAND" != "list" ] && [ "$COMMAND" != "cleanall" ]; then
            1_2_interactive_wizard
        # fi
    fi

    # Construct Preset Name
    PRESET_NAME="${PLATFORM}-${BUILD_TYPE}"
    if [ "$LIB_TYPE" == "static" ]; then
        PRESET_NAME="${PRESET_NAME}-static"
    fi

    # Derive Build Directory (Matches CMakePresets.json logic)
    BUILD_DIR="${REPO_DIR}/build/${PRESET_NAME}"

    echo "--------------------------------------------"
    echo "Action:      $COMMAND"
    echo "Target:      [${PRESET_NAME}]"
    echo "Build Dir:   ${BUILD_DIR}"
    echo "--------------------------------------------"
}

2_2_check_preset_existence() {
    if ! cmake --list-presets | grep -q "${PRESET_NAME}"; then
        echo "Error: Preset '${PRESET_NAME}' is not defined in CMakePresets.json."
        exit 1
    fi
}

2_3_validate_platform_env() {
    # If platform is native, we rely on host compiler, no SDK root needed usually.
    if [ "$PLATFORM" == "native" ]; then
        return 0
    fi

    # 1. Construct the variable name dynamically based on PLATFORM
    #    Example: rv1126bp -> ARC_RV1126BP_SDK_ROOT
    #    We convert platform to uppercase.
    local PLATFORM_UPPER=$(echo "$PLATFORM" | tr '[:lower:]' '[:upper:]')
    local VAR_NAME="ARC_${PLATFORM_UPPER}_SDK_ROOT"

    # 2. Get the value of this variable (Indirect reference)
    local SDK_PATH="${!VAR_NAME}"

    # 3. Validation Logic
    if [ -z "$SDK_PATH" ]; then
        echo -e "\033[1;31mError: Environment variable '${VAR_NAME}' is not set!\033[0m"
        echo "Please define it in your '.env' file."
        echo "Example: ${VAR_NAME}=\"/development/toolchain/host\""
        exit 1
    fi

    if [ ! -d "$SDK_PATH" ]; then
        echo -e "\033[1;31mError: SDK path '${SDK_PATH}' does not exist!\033[0m"
        echo "Variable: ${VAR_NAME}"
        echo "Please check your '.env' configuration."
        exit 1
    fi

    echo ">> [Env Check] Platform SDK: ${VAR_NAME}=${SDK_PATH}"
}


# ==============================================================================
# 3. Atomic Core Operations (Helper Functions)
# ==============================================================================

3_1_core_clean_target() {
    if [ -d "${BUILD_DIR}" ]; then
        echo ">> [Clean] Removing ${BUILD_DIR}..."
        rm -rf "${BUILD_DIR}"
    else
        echo "[Clean] Directory ${BUILD_DIR} does not exist."
    fi
}

3_2_core_configure() {
    # Accepts optional arguments (e.g., -DBUILD_TEST=ON)
    local EXTRA_ARGS="$@"

    2_2_check_preset_existence

    echo ">> [Configure] Preset: ${PRESET_NAME}"
    echo ">> [Configure] Options: ARC_INSTALL_SHERPA_TOOLS=${ARC_INSTALL_SHERPA_TOOLS} ${EXTRA_ARGS}"

    cmake --preset "${PRESET_NAME}" \
          -DARC_INSTALL_SHERPA_TOOLS="${ARC_INSTALL_SHERPA_TOOLS}" \
          ${EXTRA_ARGS}
}

3_3_core_build() {
    echo ">> [Build] Directory: ${BUILD_DIR}"
    cmake --build "${BUILD_DIR}" -j$(nproc)
}

3_4_core_install() {
    echo ">> [Install] Installing to prefix..."
    cmake --install "${BUILD_DIR}"
}

3_5_core_test() {
    echo ">> [Test] Running CTest..."
    ctest --test-dir "${BUILD_DIR}" --output-on-failure --verbose
}

# ==============================================================================
# Main Execution Flow
# ==============================================================================

4_1_execute() {
    # 1. Pre-execution Validation
    if [ "$COMMAND" != "clean" ] && [ "$COMMAND" != "cleanall" ] && [ "$COMMAND" != "list" ]; then
        2_3_validate_platform_env
    fi

    # 2. Command Dispatch
    case "$COMMAND" in
        "list")
            cmake --list-presets
            ;;

        "clean")
            3_1_core_clean_target
            ;;

        "cleanall")
            echo ">> [CleanAll] Removing entire build/ and install/ directory..."
            rm -rf "${REPO_DIR}/build"
            rm -rf "${REPO_DIR}/install"
            ;;

        "build")
            # Flow: Configure -> Build
            3_2_core_configure
            3_3_core_build
            ;;

        "cb")
            # Flow: Clean -> Configure -> Build -> Install
            3_1_core_clean_target
            3_2_core_configure
            3_3_core_build
            3_4_core_install
            ;;

        "install")
            # Flow: Check -> Install
            2_2_check_preset_existence
            if [ ! -d "${BUILD_DIR}" ]; then
                echo "Error: Build directory not found. Please build first."
                exit 1
            fi
            3_4_core_install
            ;;

        "test")
            # Flow: Configure(Test=ON) -> Build -> Test
            # Force Re-configure to ensure ENABLE_TESTING is picked up.
            # (If we don't, a previous 'build' might have cached ENABLE_TESTING=OFF)

            echo ">> [Test] Re-configuring with tests enabled..."
            3_2_core_configure "-DBUILD_TEST=TRUE"

            echo ">> [Test] Building..."
            3_3_core_build

            echo ">> [Test] Executing..."
            3_5_core_test
            ;;

        *)
            echo "Error: Unknown command '$COMMAND'"
            1_3_helper_print
            exit 1
            ;;
    esac
}

main() {
    1_1_load_env
    2_1_arguments_parsing "$@"
    4_1_execute
}

main "$@"