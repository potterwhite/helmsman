
# ==============================================================================
# CPP Build Module (Fully Inlined)
# ==============================================================================

# ==============================================================================
# Level 6: Cpp last(3rd) layer execution unit
# ==============================================================================
func_6_2_cpp_setup_env(){

    # Paths
    CPP_TOP_DIR="${REPO_TOP_DIR}/runtime/cpp"
    # ------------------
    # CPP_BUILD_DIR only used in cleaning execution
    CPP_BUILD_DIR="${CPP_TOP_DIR}/build"
    # ------------------
    # CPP_BUILD_WITH_PLATFORM_DIR will be assigned after params parsing
    # and all build progress occurs in this place
    # it will be assigned like: runtime/cpp/build/native-release
    # ------------------
    CPP_BUILD_WITH_PLATFORM_DIR=""
    # ------------------
    CPP_INSTALL_DIR="${CPP_TOP_DIR}/install"
    CPP_INSTALL_INC_DIR="${CPP_INSTALL_DIR}/include"
    CPP_INSTALL_BIN_DIR="${CPP_INSTALL_DIR}/bin"
    CPP_INSTALL_LIB_DIR="${CPP_INSTALL_DIR}/lib"
    CPP_SRC_DIR="${CPP_TOP_DIR}/src"
    CPP_TOOLCHAINS_DIR="${CPP_TOP_DIR}/cmake/toolchains"
    CPP_DOT_ENV_PATH="${CPP_TOP_DIR}/.env"

    # Commands
    CPP_CMD_CLEAN="clean"
    CPP_CMD_BUILD="build"
    CPP_CMD_CB="cb"
    CPP_CMD_INSTALL="install"
    CPP_CMD_TEST="test"

    # ORT Version Define - Conan will use it
    ONNXRUNTIME_VERSION=""
    ONNXRUNTIME_ROOT=""

    # --- [NEW] Load .env file if it exists ---
    # This exports all variables defined in .env to the current shell
    if [ -f "${CPP_DOT_ENV_PATH}" ]; then
        set -a  # Automatically export all variables
        source "${CPP_DOT_ENV_PATH}"
        set +a
    else
        echo ">> [Fatal] .env file not found in $(basename ${CPP_DOT_ENV_PATH})."
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

    # CPP related options


    # Defaults
    CPP_ACTION=""
    CPP_PLATFORM="native"
    CPP_BUILD_TYPE="release"
    CPP_LIB_TYPE="shared"
    CPP_PRESET_NAME=""

}

func_6_1_cpp_check_preset_existence() {
    cmake --version
    pwd -P
    cmake --list-presets

    if ! cmake --list-presets | grep -q "${CPP_PRESET_NAME}"; then
        func_1_2_err "Error: Preset '${CPP_PRESET_NAME}' is not defined in CMakePresets.json."
        exit 1
    fi
}

func_6_3_cpp_helper_print() {
    echo -e "\n\033[1;33mUsage: $0 build cpp <command> [options...]\033[0m"
    echo ""
    echo -e "\033[1mCommands:\033[0m"
    echo "  build                - Configure and Compile (Incremental build)"
    echo "  cb                   - Clean, Build AND Install (Fresh build)"
    echo "  clean                - Remove build directory for current preset"
    # echo "  cleanall             - Remove ALL build and install directories"
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
    echo "     $0 build cpp build"
    echo ""
    echo "  # 2. Cross-compile for RV1126 (Release, Shared)"
    echo "     $0 build cpp build rv1126bp"
    echo ""
    echo "  # 3. Build Debug version for Host PC"
    echo "     $0 build cpp build debug"
    echo ""
    echo "  # 4. Build Static Libraries (Host PC)"
    echo "     $0 build cpp build static"
    echo ""
    echo "  # 5. Full Rebuild & Install for RV1126 (Debug mode, Static Libs)"
    echo "     $0 build cpp cb rv1126bp debug static"
    echo ""
    echo "  # 6. Just Clean the RV1126 build files"
    echo "     $0 build cpp clean rv1126bp"
    echo ""
}

func_6_4_cpp_print_every_param(){
    local index
    for (( index = 0; index <= $#; index++ )); do
        func_1_1_log "param[${index}]=\"${!index}\""
    done
}

# ==============================================================================
# Level 7: Cpp sencond layer execution unit
# ==============================================================================
func_7_1_cpp_core_clean_target() {
    if [ -d "${CPP_BUILD_WITH_PLATFORM_DIR}" ]; then
        echo ">> [Clean] Removing directory: ${CPP_BUILD_WITH_PLATFORM_DIR}"
        rm -rf "${CPP_BUILD_WITH_PLATFORM_DIR}"
    else
        echo "[Clean] Directory ${CPP_BUILD_WITH_PLATFORM_DIR} does not exist."
    fi

    # if [ -d "${CPP_BUILD_DIR}" ]; then
    #     echo ">> [Clean] Removing ${CPP_BUILD_DIR}..."
    #     rm -rf "${CPP_BUILD_DIR}"
    # else
    #     echo "[Clean] Directory ${CPP_BUILD_DIR} does not exist."
    # fi

    if [ -d "${CPP_INSTALL_DIR}" ]; then
        echo ">> [Clean] Removing ${CPP_INSTALL_DIR}..."
        rm -rf "${CPP_INSTALL_DIR}"
    else
        echo "[Clean] Directory ${CPP_INSTALL_DIR} does not exist."
    fi
}

func_7_2_cpp_core_configure() {
    # Accepts optional arguments
    local extra_args="$@"

    func_6_1_cpp_check_preset_existence

    func_1_1_log ">> [Configure] Using Preset: ${CPP_PRESET_NAME}" "blue"

    # We rely purely on the Preset now.
    # The Toolchain, CMAKE_BUILD_TYPE, and BUILD_SHARED_LIBS are all in the JSON.

    cmake --preset "${CPP_PRESET_NAME}" \
          -DARC_INSTALL_SHERPA_TOOLS="${ARC_INSTALL_SHERPA_TOOLS}" \
          ${extra_args}
}

func_7_3_cpp_core_build() {
    echo ">> [Build] Directory: ${CPP_BUILD_WITH_PLATFORM_DIR}"
    cmake \
        --build "${CPP_BUILD_WITH_PLATFORM_DIR}" \
        -j$(nproc)
}

func_7_4_cpp_core_install() {
    echo ">> [Install] Installing to prefix..."
    cmake --install "${CPP_BUILD_WITH_PLATFORM_DIR}"
}

func_7_5_cpp_core_test() {
    echo ">> [Test] Running CTest..."
    ctest --test-dir "${CPP_BUILD_WITH_PLATFORM_DIR}" --output-on-failure --verbose
}


# ==============================================================================
# Level 8: Cpp action execution (Top layer)
# ==============================================================================
# ==============================================================================
# [NEW] Helper: Conan configuration workflow for Native platform
# ==============================================================================
func_8_5_cpp_run_native_conan_configure() {
    local extra_args="$@"
    echo "[1]extra_args=\"${extra_args}\""

    # 1. Check Conan existence
    if ! command -v conan &> /dev/null; then
        func_1_2_err "Conan is not installed. Please run './helmsman prepare' or install it manually."
    fi

    # 2. Check for conanfile
    if [ ! -f "${CPP_TOP_DIR}/conanfile.py" ] && [ ! -f "${CPP_TOP_DIR}/conanfile.txt" ]; then
        func_1_2_err "No 'conanfile.py' or 'conanfile.txt' found in ${CPP_TOP_DIR}. Cannot run Conan build."
    fi

    func_1_1_log ">> [Conan] Detecting profile (idempotent)..." "blue"
    conan profile detect --force 2>/dev/null || true

    func_1_1_log ">> [Conan] Installing dependencies..." "blue"
    func_1_1_log "   Output Dir: ${CPP_BUILD_WITH_PLATFORM_DIR}"
    func_1_1_log "   Build Type: ${CPP_BUILD_TYPE}"

    # Conan Install
    # Note: We use --output-folder to direct generated files (toolchains) to the build dir.
    # We map 'debug/release' to Conan's settings.
    local conan_build_type="Release"
    if [[ "${CPP_BUILD_TYPE}" == "debug" ]]; then
        conan_build_type="Debug"
    fi

    conan install "${CPP_TOP_DIR}" \
        --output-folder="${CPP_BUILD_WITH_PLATFORM_DIR}" \
        --build=missing \
        -s build_type="${conan_build_type}" \
        -s compiler.cppstd=17

    if [ $? -ne 0 ]; then
        func_1_2_err "Conan install failed."
    fi

    func_1_1_log ">> [CMake] Configuring with Conan Toolchain..." "blue"

    # 3. CMake Configure using the Conan Toolchain
    # Instead of --preset, we manually specify the toolchain generated by Conan.
    # Conan 2.x usually generates 'conan_toolchain.cmake' in the output folder (or build/ generators).

    local toolchain_file="${CPP_BUILD_WITH_PLATFORM_DIR}/conan_toolchain.cmake"

    # Fallback check for generators folder (common in some Conan configs)
    if [ ! -f "$toolchain_file" ]; then
        toolchain_file="${CPP_BUILD_WITH_PLATFORM_DIR}/build/Release/generators/conan_toolchain.cmake"
    fi

    # If still not found, try a generic search inside build dir (robustness)
    if [ ! -f "$toolchain_file" ]; then
        toolchain_file=$(find "${CPP_BUILD_WITH_PLATFORM_DIR}" -name "conan_toolchain.cmake" | head -n 1)
    fi

    if [ -z "$toolchain_file" ] || [ ! -f "$toolchain_file" ]; then
         func_1_1_log "⚠️  Could not locate conan_toolchain.cmake. Attempting standard configure..." "yellow"
         cmake -S "${CPP_TOP_DIR}" -B "${CPP_BUILD_WITH_PLATFORM_DIR}" \
            -DCMAKE_BUILD_TYPE="${conan_build_type}" \
            -DARC_INSTALL_SHERPA_TOOLS="${ARC_INSTALL_SHERPA_TOOLS}" \
            ${extra_args}
    else
         func_1_1_log "   Toolchain: $toolchain_file" "green"
         echo -e "\n\n"
         echo "[2]extra_args=\"${extra_args}\""
         cmake -S "${CPP_TOP_DIR}" -B "${CPP_BUILD_WITH_PLATFORM_DIR}" \
            -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
            -DCMAKE_BUILD_TYPE="${conan_build_type}" \
            -DARC_INSTALL_SHERPA_TOOLS="${ARC_INSTALL_SHERPA_TOOLS}" \
            ${extra_args}
    fi
}

func_8_4_cpp_dispatch() {
    # 1. Pre-execution Validation (Common)
    if [ "$CPP_ACTION" != "clean" ] && [ "$CPP_ACTION" != "cleanall" ] && [ "$CPP_ACTION" != "list" ]; then
        func_8_3_cpp_validate_platform_env
    fi

    # 2. Branch Logic: Native (Keep Conan flow) vs Others (Pure Preset flow)
    if [ "$CPP_PLATFORM" == "native" ] && [ "$CPP_ACTION" != "list" ] && [ "$CPP_ACTION" != "clean" ]; then

        # --- NATIVE / CONAN PATH (Kept logic for safety as requested) ---
        func_1_1_log "🚀 Dispatching Native Build (Conan Powered)..." "green"
        func_3_4_detect_onnxruntime_version # Ensure we have the version

        case "$CPP_ACTION" in
            "build")
                # Native flow still uses the manual Conan injector
                func_8_5_cpp_run_native_conan_configure "$(func_3_6_echo_ort_root_for_cmake)"
                func_7_3_cpp_core_build
                ;;

            "cb")
                func_7_1_cpp_core_clean_target
                func_8_5_cpp_run_native_conan_configure "$(func_3_6_echo_ort_root_for_cmake)"
                func_7_3_cpp_core_build
                func_7_4_cpp_core_install
                ;;

            "install")
                # Install check
                if [ ! -d "${CPP_BUILD_WITH_PLATFORM_DIR}" ]; then
                    func_1_2_err "Build directory not found. Please build first."
                fi
                func_7_4_cpp_core_install
                ;;

            "test")
                func_8_5_cpp_run_native_conan_configure "-DBUILD_TEST=TRUE" "$(func_3_6_echo_ort_root_for_cmake)"
                func_7_3_cpp_core_build
                func_7_5_cpp_core_test
                ;;

            *)
                 func_1_2_err "Unknown native command '$CPP_ACTION'"
                 ;;
        esac

    else
        # --- CROSS-COMPILE / PURE PRESET PATH ---
        # This is where the optimization happens.
        # We no longer manually handle toolchains here, the JSON does it.

        if [ "$CPP_PLATFORM" != "native" ]; then
            func_1_1_log "⚙️  Dispatching Preset Build (Target: ${CPP_PRESET_NAME})..." "blue"
        fi

        case "$CPP_ACTION" in
            "list")
                cmake --list-presets
                ;;

            "clean")
                # Uses the directory calculated in func_8_2 based on Preset Name
                func_7_1_cpp_core_clean_target
                ;;

            "build")
                # Incremental build
                func_7_2_cpp_core_configure
                func_7_3_cpp_core_build
                ;;

            "cb")
                # Clean -> Configure -> Build -> Install
                func_7_1_cpp_core_clean_target
                func_7_2_cpp_core_configure
                func_7_3_cpp_core_build
                func_7_4_cpp_core_install
                ;;

            "install")
                # Presets usually define installDir, but cmake --install needs build dir
                if [ ! -d "${CPP_BUILD_WITH_PLATFORM_DIR}" ]; then
                    func_1_2_err "Build directory ${CPP_BUILD_WITH_PLATFORM_DIR} not found."
                fi
                func_7_4_cpp_core_install
                ;;

            "test")
                # Re-configure with Test ON (passed as extra arg)
                func_7_2_cpp_core_configure "-DBUILD_TEST=TRUE"
                func_7_3_cpp_core_build
                func_7_5_cpp_core_test
                ;;

            *)
                func_1_2_err "Unknown CPP command '$CPP_ACTION'"
                ;;
        esac
    fi
}

func_8_3_cpp_validate_platform_env() {
    # If platform is native, we rely on host compiler, no SDK root needed usually.
    if [ "$CPP_PLATFORM" == "native" ]; then
        return 0
    fi

    # 1. Construct the variable name dynamically based on CPP_PLATFORM
    #    Example: rv1126bp -> ARC_RV1126BP_SDK_ROOT
    #    We convert platform to uppercase.
    local platform_upper=$(echo "$CPP_PLATFORM" | tr '[:lower:]' '[:upper:]')
    local var_name="ARC_${platform_upper}_SDK_ROOT"

    # 2. Get the value of this variable (Indirect reference)
    local sdk_path="${!var_name}"

    # 3. Validation Logic
    if [ -z "$sdk_path" ]; then
        echo -e "\033[1;31mError: Environment variable '${var_name}' is not set!\033[0m"
        echo "Please define it in your '.env' file."
        echo "Example: ${var_name}=\"/development/toolchain/host\""
        exit 1
    fi

    if [ ! -d "$sdk_path" ]; then
        echo -e "\033[1;31mError: SDK path '${sdk_path}' does not exist!\033[0m"
        echo "Variable: ${var_name}"
        echo "Please check your '.env' configuration."
        exit 1
    fi

    echo ">> [Env Check] Platform SDK: ${var_name}=${sdk_path}"
}

func_8_2_cpp_arguments_parsing() {
    # If verbose, print inputs
    if [ x"${DEBUG_MODE}" == x"1" ];then func_6_4_cpp_print_every_param "$@"; fi

    CPP_ACTION=$1
    shift || true

    # Defaults matching your CMakePresets "base" logic or common sense
    # You might want to default to empty if you want to force user input,
    # but here are reasonable defaults based on your script history.
    CPP_BUILD_TYPE="release"
    CPP_LIB_TYPE="shared"
    CPP_PLATFORM="native"

    # --- Loop through arguments ---
    while [ "$#" -gt 0 ]; do
        case "$1" in
            "debug"|"Debug"|"DEBUG")
                CPP_BUILD_TYPE="debug"
                ;;
            "release"|"Release"|"RELEASE")
                CPP_BUILD_TYPE="release"
                ;;
            "static"|"Static"|"STATIC")
                CPP_LIB_TYPE="static"
                ;;
            "shared"|"Shared"|"SHARED")
                CPP_LIB_TYPE="shared"
                ;;
            *)
                # Assuming anything else is the platform (e.g., rv1126bp, rk3588s)
                CPP_PLATFORM=$1
                ;;
        esac
        shift
    done

    # -----------------------------------------------------------
    # [CORE CHANGE] Construct the Preset Name
    # Mapping logic: <platform>-<build_type>[-static]
    # Examples based on your JSON:
    #   native-release
    #   rv1126bp-debug-static
    #   rk3588s-release
    # -----------------------------------------------------------

    CPP_PRESET_NAME="${CPP_PLATFORM}-${CPP_BUILD_TYPE}"

    if [ "$CPP_LIB_TYPE" == "static" ]; then
        CPP_PRESET_NAME="${CPP_PRESET_NAME}-static"
    fi

    # -----------------------------------------------------------
    # [SYNC] Align Bash Build Dir with JSON "binaryDir"
    # JSON Line 68: "binaryDir": "${sourceDir}/build/${presetName}"
    # -----------------------------------------------------------
    CPP_BUILD_WITH_PLATFORM_DIR="${CPP_TOP_DIR}/build/${CPP_PRESET_NAME}"

    if [ x"${DEBUG_MODE}" == x"1" ];then
        func_1_1_log ">> [Preset Logic] Calculated Preset: ${CPP_PRESET_NAME}" "blue"
        func_1_1_log ">> [Preset Logic] Binary Dir: ${CPP_BUILD_WITH_PLATFORM_DIR}" "blue"
    fi
}

func_8_1_cpp_build_dispatch() {
    # ------------------------------
    # This is CPP Functionalities Entry
    # ------------------------------

    if [ x"${DEBUG_MODE}" == x"1" ];then
        func_6_4_cpp_print_every_param "$@"
    fi

    local target="$1"
    # shift || true   # shed the bin name $0
    shift || true   # shed the "cpp" indicator

    # func_6_4_cpp_print_every_param "$@"
    func_8_2_cpp_arguments_parsing "$@"

    if [ -d "${CPP_TOP_DIR}" ];then
        cd "${CPP_TOP_DIR}"
    else
        func_1_2_err "${CPP_TOP_DIR} is not exist! Check and Re-run again."
        exit 1
    fi

    case "$target" in
        cpp)
            if [ -z "$CPP_ACTION" ]; then
                func_1_2_err \
                    "Usage: ./helmsman build cpp <clean|build|cb|install|test|list> [platform]"
            fi

            func_1_1_log \
                "🛠️  CPP Build: cmd=${CPP_ACTION}, platform=${CPP_PLATFORM}" \
                "blue"

            # func_8_4_cpp_dispatch "$CPP_ACTION" "$CPP_PLATFORM"
            func_8_4_cpp_dispatch
            ;;
        *)
            func_1_2_err "Unknown build target '$target'"
            ;;
    esac
}
