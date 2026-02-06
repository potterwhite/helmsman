# ==============================================================================
# Level 1: Helpers & Logging
# ==============================================================================

func_1_1_log() {
    local message="$1"
    local color_name="$2"
    local color_code=""
    case "$color_name" in
        green)  color_code="\033[1;32m" ;;
        yellow) color_code="\033[1;33m" ;;
        red)    color_code="\033[1;31m" ;;
        blue)   color_code="\033[1;34m" ;;
        *)      color_code="\033[0m"    ;;
    esac
    echo -e "${color_code}[Helmsman] ${message}\033[0m"
}

func_1_2_err() {
    func_1_1_log "ERROR: $1" "red"
    exit 1
}

func_1_3_get_current_milliseconds() { date +%s%3N; }

func_1_4_start_time_count() {
    local -n _timer_ref=$1
    _timer_ref="$(func_1_3_get_current_milliseconds)"
}

func_1_5_format_duration_ms() {
    local ms=$1
    local -n _out_ref=$2
    if (( ms < 1000 )); then _out_ref="${ms} ms"; return; fi
    if (( ms < 60000 )); then _out_ref=$(printf "%.2f seconds" "$(awk "BEGIN { print $ms / 1000 }")"); return; fi
    local total_seconds=$(( ms / 1000 ))
    local minutes=$(( total_seconds / 60 ))
    local remaining_ms=$(( ms % 60000 ))
    _out_ref=$(printf "%d minutes %.2f seconds" "$minutes" "$(awk "BEGIN { print $remaining_ms / 1000 }")")
}

func_1_6_elapsed_time_calculation() {
    local start_time=$1
    local end_time=$2
    local -n elapsed_string_ref=$3
    local elapsed_time=$((end_time - start_time))
    func_1_5_format_duration_ms "${elapsed_time}" elapsed_string_ref
}

func_1_7_usage() {
    echo "======================================================="
    echo -e "\n\033[1;33mUsage: $0 <command> [options...]\033[0m"
    echo "Commands: prepare, convert, inference, build, clean, cleanall"
    echo "======================================================="
    func_6_3_cpp_helper_print
}

func_1_8_activate_py_venv(){
    if [ ! -d "$LV1_VENV_DIR" ]; then
        func_1_1_log "❌ Environment not ready. Virtual environment not found at '$LV1_VENV_DIR'." "red"
        exit 1
    else
        source ${LV1_VENV_DIR}/bin/activate
        func_1_1_log "✅ Activate Python venv Successfully." "green"
    fi
}

# ==============================================================================
# Level 2: Environment Variables
# ==============================================================================

func_2_0_setup_env(){
    set -e
    if [[ x"${V}" == x"1" ]];then set -x; DEBUG_MODE=1; fi

    # REPO_TOP_DIR 已经在入口脚本定义了，这里直接用

    # Define Paths
    export PYTHON_Target_VERSION="3.8.10"
    LV1_DOC_DIR="${REPO_TOP_DIR}/docs"
    LV1_ENVS_DIR="${REPO_TOP_DIR}/envs"
    LV1_MEDIA_DIR="${REPO_TOP_DIR}/media"
    LV1_RUNTIME_DIR="${REPO_TOP_DIR}/runtime"
    LV1_3RD_PARTY_DIR="${REPO_TOP_DIR}/third-party"
    LV1_TOOLS_DIR="${REPO_TOP_DIR}/tools"
    LV1_VENV_DIR="${REPO_TOP_DIR}/.venv"
    LV1_BUILD_DIR="${REPO_TOP_DIR}/build"
    LV1_MODELS_DOWNLOAD_DIR="${REPO_TOP_DIR}/modnet-models"

    # 2nd-lv
    LV4_MODNET_SDK_DIR="${LV1_3RD_PARTY_DIR}/sdk/MODNet.git"
    LV4_MODNET_SCRIPTS_DIR="${LV1_3RD_PARTY_DIR}/scripts/modnet"
    LV4_MODNET_MODELS_DIR="${LV1_3RD_PARTY_DIR}/models/modnet"
    LV5_PRETRAINED_DIR="${LV4_MODNET_SDK_DIR}/pretrained"

    DEV_REQUIREMENTS_FILE="${LV1_ENVS_DIR}/requirements.txt"
    LV2_GOLDEN_DIR="${LV1_BUILD_DIR}/golden"
    LV2_GOLDEN_DEBUG_DIR="${LV2_GOLDEN_DIR}/debug"

    PYTHON_BIN="${LV1_VENV_DIR}/bin/python"
    PIP_BIN="${LV1_VENV_DIR}/bin/pip"

    # Pyenv
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv 1>/dev/null 2>&1; then
      eval "$(pyenv init --path)"
      eval "$(pyenv init -)"
    fi

    trap func_5_4_finalize EXIT

    # Setup CPP Env (defined in cpp_build.sh but called here)
    # 因为所有脚本都 source 进来了，所以可以跨文件调用
    func_6_2_cpp_setup_env
}

func_2_1_check_env_ready(){
    func_1_8_activate_py_venv
    if [ ! -f "$PYTHON_BIN" ]; then func_1_2_err "Python binary not found."; fi

    # 简化的检查逻辑，保留核心
    local required_files=(
        "${LV4_MODNET_SDK_DIR}/onnx/export_onnx.py"
        "${LV4_MODNET_SDK_DIR}/pretrained/mobilenetv2_human_seg.ckpt"
    )
    for target_file in "${required_files[@]}"; do
        if [ ! -f "$target_file" ]; then
             func_1_1_log "❌ Missing: $target_file (Run 'prepare')" "yellow"
             exit 1
        fi
    done
}