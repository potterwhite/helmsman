#!/bin/bash


func_1_0_load_env(){
    set -e

    if [[ x"${V}" == x"1" ]];then
        set -x
    fi

    BUILD_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
    BUILD_SCRIPT_DIR="$(dirname ${BUILD_SCRIPT_PATH})"
    REPO_TOP_DIR="${BUILD_SCRIPT_DIR}/../../.."
    # ------------------------------------------
    MODNET_REPO_DIR="${REPO_TOP_DIR}/MODNet.git"
    # ------------------------------------------
    VENV_DIR="${REPO_TOP_DIR}/.venv"
    DEV_REQUIREMENTS_FILE="${MODNET_REPO_DIR}/onnx/requirements.txt"

    # ==========
    PYTHON_CMD="python3"
}

func_1_1_prepare_software(){
    sudo apt update && sudo apt-get install cmake
}

func_1_2_log() {
    local message="$1"
    local color_name="$2"
    local color_code=""
    case "$color_name" in
    green) color_code="\033[0;32m" ;;
    yellow) color_code="\033[0;33m" ;;
    red) color_code="\033[0;31m" ;;
    blue) color_code="\033[0;34m" ;;
    *) color_code="\033[0m" ;;
    esac
    local color_reset="\033[0m"
    echo -e "${color_code}${message}${color_reset}"
}


# --- Main Functions ---
func_2_0_create_venv() {
    func_1_2_log "🛠️  Starting project preparation..." "blue"
    if ! command -v "$PYTHON_CMD" &>/dev/null; then
        func_1_2_log "❌ Error: '$PYTHON_CMD' could not be found." "red"
        exit 1
    fi
    func_1_2_log "✅ Found $($PYTHON_CMD --version)" "green"

    if [ ! -d "$VENV_DIR" ]; then
        func_1_2_log "   Creating virtual environment in './$VENV_DIR'..." "yellow"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    else
        func_1_2_log "   Virtual environment already exists." "green"
    fi

    source "$VENV_DIR/bin/activate"

    func_1_2_log "   Upgrading pip..." "yellow"
    pip install --upgrade pip

    # func_1_2_log "   Installing core dependencies from 'pyproject.toml'..." "yellow"
    # pip install -e .

    if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
        func_1_2_log "   Installing development dependencies from '$DEV_REQUIREMENTS_FILE'..." "yellow"
        pip install -r "$DEV_REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    func_1_2_log "🎉 Preparation complete!" "green"
}

main(){
    func_1_0_load_env || exit
    func_1_1_prepare_software || exit

    func_2_0_create_venv || exit
}

main "$@"