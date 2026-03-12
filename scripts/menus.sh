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


# ==============================================================================
# Level 9: Cpp menu interactive
# ==============================================================================

func_9_4_cpp_list_platforms() {
    echo "native"
    # func_1_1_log "CPP_TOOLCHAINS_DIR=${CPP_TOOLCHAINS_DIR}"
    if [ -d "${CPP_TOOLCHAINS_DIR}" ]; then
        for f in "${CPP_TOOLCHAINS_DIR}"/*.cmake; do
            [ -f "$f" ] || continue
            basename "$f" .cmake
        done
    fi
}

func_9_3_cpp_menu_select_platform() {
    local platforms=()
    mapfile -t platforms < <(func_9_4_cpp_list_platforms)

    echo ""
    func_1_1_log "--- Select C++ Build Platform ---" "blue"

    local i=1
    for p in "${platforms[@]}"; do
        echo "${i}. ${p}"
        ((i++))
    done
    echo "${i}. Back"

    read -p "   Select [1-${i}]: " choice

    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#platforms[@]} )); then
        CPP_PLATFORM="${platforms[choice-1]}"
        # echo "${platforms[choice-1]}"
        echo "${CPP_PLATFORM}"
        return 0
    fi

    return 1
}

func_9_2_cpp_menu_select_action() {
    local timeout=10

    echo
    echo ">> [Interactive Mode] Wizard initiated."
    echo ">> Defaults will be auto-selected in ${timeout} seconds (or press Enter)..."
    echo

    # ---------------------------------------
    # Step 0: Command Selection
    # Only ask if COMMAND is not already provided via CLI
    # ---------------------------------------
    if [ -z "$CPP_ACTION" ]; then
        echo "---------------------------------------"
        echo "Select Action:"
        echo "  1) cb       (Clean, Build & Install) [Default]"
        echo "  2) build    (Incremental Build)"
        echo "  3) install  (Install only)"
        echo "  4) clean    (Clean current preset)"
        echo "  5) test     (Run tests)"
        echo "  6) list     (List presets)"
        echo "  7) back"
        echo "---------------------------------------"
        read -t ${timeout} -p "Enter choice [1-7]: " CMD_INPUT || true
        echo

        case "$CMD_INPUT" in
            "2") CPP_ACTION="build" && return;;
            "3") CPP_ACTION="install" && return;;
            "4") CPP_ACTION="clean" && return;;
            "5") CPP_ACTION="test" ;;
            "6") CPP_ACTION="list" && return;;
            "7") CPP_ACTION="back" && return;;
            *)   CPP_ACTION="cb" ;; # Default
        esac
        echo ">> Selected Action:   ${CPP_ACTION}"
    else
        echo ">> Action provided:   ${CPP_ACTION}"
    fi


    # ---------------------------------------
    # Step 1: Platform Selection
    # ---------------------------------------
    func_9_3_cpp_menu_select_platform
    # echo "---------------------------------------"
    # echo "Select Platform:"
    # echo "  1) native   (Default)"
    # echo "  2) rv1126bp"
    # echo "  3) aarch64"
    # echo "---------------------------------------"
    # read -t ${timeout} -p "Enter choice [1-3]: " PLAT_INPUT || true
    # echo

    # case "$PLAT_INPUT" in
    #     "2") PLATFORM="rv1126bp" ;;
    #     "3") PLATFORM="aarch64" ;;
    #     *)   ;; # Keep default (native)
    # esac
    echo ">> Selected Platform: ${CPP_PLATFORM}"


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
        "2") CPP_BUILD_TYPE="debug" ;;
        *)   ;; # Keep default (release)
    esac
    echo ">> Selected Build Type: ${CPP_BUILD_TYPE}"


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
        "2") CPP_LIB_TYPE="static" ;;
        *)   ;; # Keep default (shared)
    esac
    echo ">> Selected Lib Type:   ${CPP_LIB_TYPE}"
    echo

    # ---------------------------------------------------
    # echo ""
    # func_1_1_log "--- Select C++ Build Action ---" "blue"
    # echo "1. build"
    # echo "2. clean + build + install (cb)"
    # echo "3. clean"
    # echo "4. install"
    # echo "5. back"

    # read -p "   Select [1-5]: " choice

    # case "$choice" in
    #     1) CPP_ACTION="build" ;;
    #     2) CPP_ACTION="cb" ;;
    #     3) CPP_ACTION="clean" ;;
    #     4) CPP_ACTION="install" ;;
    #     5) CPP_ACTION="back" ;;
    #     *) func_1_1_log "Invalid choice." "red"; return 1 ;;
    # esac
}

func_9_1_cpp_menu_entry() {
    echo ""
    # while true; do
        func_9_2_cpp_menu_select_action

        if [[ "$CPP_ACTION" == "back" ]]; then
            return
        fi

        func_8_1_cpp_build_dispatch cpp "$CPP_ACTION" "$CPP_PLATFORM" || return
    # done
}

# --- 追加到 scripts/menus.sh 末尾 ---

# 这是为了配合分文件，把原脚本 main 函数里的主菜单逻辑提取出来的封装
func_9_1_main_menu_entry() {
    # while true; do  <-- 原脚本这里注释掉了 while，保持原样
        echo ""
        func_1_1_log "--- Helmsman Project Main Menu ---" "blue"
        echo "1. Prepare Environment"
        echo "2. Convert .ckpt model to .onnx"
        echo "3. Run Python inference"
        echo "4. Generate Golden Files (Interactive)"
        echo "5. Build C++ Helmsman Engine"
        echo "6. Clean build"
        echo "7. Clean all(build/venv/models/MODNet SDK etc)"
        echo "8. Exit"
        read -p "   Select [1-8]: " choice

        case $choice in
            1) func_3_2_setup_dep_before_build ;;
            2) func_2_1_check_env_ready && func_4_1_ckpt_2_onnx ;;
            3) func_2_1_check_env_ready && func_4_2_inference_with_onnx ;;
            4) func_2_1_check_env_ready && func_4_5_generate_golden_interactive ;;
            5) func_2_1_check_env_ready && func_9_1_cpp_menu_entry ;;
            6) func_5_1_clean_project ;;
            7) func_5_1_clean_project 2;;
            8) func_1_1_log "👋 Exiting." "green"; exit 0 ;;
            *) func_1_1_log "Invalid choice." "red" ;;
        esac
    # done
}