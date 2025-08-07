#!/bin/bash

# ===================================================================================
#
#   Helmsman Project - MODNet Build & Management Script
#
#   Copyright (C) 2025, potterwhite
#   All rights reserved.
#
#   Author: PotterWhite
#   Date: 2025-08-05
#
#   Description:
#   This script automates the setup, model conversion, and inference processes
#   for the MODNet component of Project Helmsman. It handles Python virtual
#   environment creation, dependency installation, conversion of PyTorch .ckpt
#   models to .onnx format, and running inference using the generated .onnx model.
#
#   Usage:
#   ./build.sh [command] [options]
#
# ===================================================================================

# --- Helper Functions ---

func_1_0_load_env(){
    # This function sets up all necessary environment variables and paths.
    set -e
    if [[ x"${V}" == x"1" ]];then
        set -x
    fi

    # ==================== 新增部分 ====================
    # Define the target Python version for this project
    export PYTHON_VERSION="3.8.10"
    # ================================================

    BUILD_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
    BUILD_SCRIPT_DIR="$(dirname ${BUILD_SCRIPT_PATH})"
    REPO_TOP_DIR="${BUILD_SCRIPT_DIR}/../../.."
    MODNET_REPO_DIR="${REPO_TOP_DIR}/MODNet.git"
    PRETRAINED_DIR="${MODNET_REPO_DIR}/pretrained"
    MEDIA_DIR="${REPO_TOP_DIR}/media"
    VENV_DIR="${REPO_TOP_DIR}/.venv"
    DEV_REQUIREMENTS_FILE="${MODNET_REPO_DIR}/onnx/requirements.txt"

    # ==================== 修改部分 ====================
    # Set up pyenv path and initialize it
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv 1>/dev/null 2>&1; then
      eval "$(pyenv init --path)"
      eval "$(pyenv init -)"
    fi
    # Use the python provided by pyenv
    PYTHON_CMD="python"
    # ================================================
}

func_1_1_log() {
    # A utility function for printing colored log messages to the console.
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

func_1_2_usage() {
    # This function prints the help message for the script.
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Helmsman Project - MODNet Build & Management Script"
    echo ""
    echo "Manages the environment, model conversion, and inference for MODNet."
    echo ""
    echo "Commands:"
    echo "  prepare         Set up the complete development environment (system and Python)."
    echo "  convert         Convert a .ckpt model to the .onnx format."
    echo "  inference       Run inference on an image using a .onnx model."
    echo "  <no command>    Enter interactive menu mode."
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message and exit."
    echo "  -v              Enable verbose mode for debugging (set -x)."
    echo ""
}

func_1_3_check_env(){
    # This function performs a quick check to see if the environment is ready.
    if [ ! -d "$VENV_DIR" ]; then
        func_1_1_log "❌ Environment not ready. Virtual environment not found at '$VENV_DIR'." "red"
        func_1_1_log "   Please run './build.sh prepare' first." "yellow"
        exit 1
    fi
    # Activate venv to check for python packages
    source "$VENV_DIR/bin/activate"
    if ! pip show onnx > /dev/null 2>&1; then
        func_1_1_log "❌ Environment not ready. 'onnx' package not found in virtual environment." "red"
        func_1_1_log "   Please run './build.sh prepare' first." "yellow"
        exit 1
    fi
}

# --- Main Functions ---

func_2_0_prepare_env() {
    # This function performs the full, one-time setup of the development environment.
    func_1_1_log "🛠️  Starting full environment preparation..." "blue"

    # --- Step 1: Install system-level dependencies ---
    func_1_1_log "   [1/3] Installing system dependencies..." "yellow"
    # pyenv需要一些编译Python所需的依赖
    sudo apt update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
    cmake libgl1
    func_1_1_log "✅ System dependencies are up to date." "green"

    # --- Step 2: Install and set up pyenv ---
    func_1_1_log "   [2/3] Setting up pyenv for Python version management..." "yellow"
    if [ ! -d "$PYENV_ROOT" ]; then
        func_1_1_log "   Installing pyenv..."
        curl https://pyenv.run | bash
        # The installer message suggests adding lines to .bashrc, which we do programmatically here
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
        func_1_1_log "   pyenv installed. Please restart your shell or run 'source ~/.bashrc' for changes to take full effect." "yellow"
    else
        func_1_1_log "   pyenv already installed." "green"
    fi

    # --- Step 3: Install and set up Python virtual environment ---
    func_1_1_log "   [3/3] Setting up Python ${PYTHON_VERSION} and virtual environment..." "yellow"
    # Install the target Python version if it's not already installed
    if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        func_1_1_log "   Python ${PYTHON_VERSION} not found, installing with pyenv (this may take a while)..."
        pyenv install ${PYTHON_VERSION}
    else
        func_1_1_log "   Python ${PYTHON_VERSION} already installed." "green"
    fi

    # Create virtual environment using the specific python version
    if [ ! -d "$VENV_DIR" ]; then
        func_1_1_log "   Creating virtual environment in '$VENV_DIR'..."
        # Use pyenv's python executable to create the venv
        pyenv local ${PYTHON_VERSION} # Temporarily set local version
        python -m venv "$VENV_DIR"
        pyenv local --unset # Unset local version
    fi
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip
    if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
        func_1_1_log "   Installing Python packages from '$DEV_REQUIREMENTS_FILE'..."
        # 切换回旧的、可靠的requirements
        echo "torch==1.7.1" > "$DEV_REQUIREMENTS_FILE"
        echo "torchvision==0.8.2" >> "$DEV_REQUIREMENTS_FILE"
        echo "onnx==1.8.1" >> "$DEV_REQUIREMENTS_FILE"
        echo "onnxruntime==1.6.0" >> "$DEV_REQUIREMENTS_FILE"
        echo "opencv-python==4.5.1.48" >> "$DEV_REQUIREMENTS_FILE"
        echo "# Pin protobuf to a compatible version to avoid API breaking changes in 4.x" >> "$DEV_REQUIREMENTS_FILE"
        echo "protobuf==3.20.1" >> "$DEV_REQUIREMENTS_FILE"
        echo "# Pin numpy to a version BEFORE np.object was removed" >> "$DEV_REQUIREMENTS_FILE"
        echo "numpy==1.21.6" >> "$DEV_REQUIREMENTS_FILE"
        func_1_1_log "   (Overwrote requirements.txt to a stable legacy version)" "yellow"
        pip install --no-cache-dir -r "$DEV_REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    func_1_1_log "✅ Python virtual environment is ready." "green"
    func_1_1_log "🎉 Preparation complete! You are now in a Python ${PYTHON_VERSION} environment." "green"
}

func_2_1_ckpt_2_onnx(){
    # This function handles the conversion of a .ckpt model to the .onnx format.
    func_1_1_log "🔎 Checking for pre-trained models (.ckpt)..." "blue"
    # mapfile -t ckpt_files < <(find "$PRETRAINED_DIR" -maxdepth 1 -type f -name "*.ckpt")
    mapfile -t ckpt_files < <(find "$PRETRAINED_DIR" -maxdepth 1 -type l -name "*.ckpt")
    if [ ${#ckpt_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .ckpt models found in '$PRETRAINED_DIR'." "red"
        func_1_1_log "   Please download the pre-trained model first." "yellow"
        func_1_1_log "   Link: https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR" "yellow"
        return 1
    fi

    func_1_1_log "   Please choose a .ckpt model to convert to ONNX:" "green"
    select ckpt_path in "${ckpt_files[@]}"; do
        if [[ -n "$ckpt_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done
    local ckpt_filename=$(basename "$ckpt_path")
    local onnx_filename="${ckpt_filename%.ckpt}.onnx"
    local onnx_path="${PRETRAINED_DIR}/${onnx_filename}"

    func_1_1_log "   Selected CKPT: $ckpt_filename" "yellow"
    func_1_1_log "   Output ONNX will be: $onnx_path" "yellow"
    if [ -f "$onnx_path" ]; then
        read -p "   '$onnx_path' already exists. Overwrite? (y/N) " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then func_1_1_log "   Skipping conversion." "yellow"; return 0; fi
    fi
    func_1_1_log "🚀 Starting conversion..." "blue"
    cd "$MODNET_REPO_DIR"
    python -m onnx.export_onnx --ckpt-path="$ckpt_path" --output-path="$onnx_path"
    if [ $? -eq 0 ]; then func_1_1_log "✅ Conversion successful!" "green"; else func_1_1_log "❌ Conversion failed." "red"; return 1; fi
}

func_2_2_inference_with_onnx(){
    # This function runs inference on a single image using a selected .onnx model.
    func_1_1_log "🔎 Checking for ONNX models (.onnx)..." "blue"
    mapfile -t onnx_files < <(find "$PRETRAINED_DIR" -maxdepth 1 -type f,l -name "*.onnx")
    if [ ${#onnx_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .onnx models found in '$PRETRAINED_DIR'." "red";
        func_1_1_log "   Please run the 'convert' command first." "yellow"; return 1;
    fi
    func_1_1_log "   Please choose an ONNX model for inference:" "green"
    select model_path in "${onnx_files[@]}"; do
        if [[ -n "$model_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    func_1_1_log "🔎 Checking for exist pictures ..." "blue"
    mapfile -t images_list < <(find "${MEDIA_DIR}" -type f,l -name "*.jpg" -o -name "*.bmp" -o -name "*.png")
    # read -p "   Please enter the path to the input image: " image_path
    func_1_1_log "   Please choose an picture for inference:" "green"
    select image_path in "${images_list[@]}"; do
        if [[ -n "$image_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done
    if [ ! -f "$image_path" ]; then func_1_1_log "❌ Input image not found at '$image_path'" "red"; return 1; fi

    local input_filename=$(basename "$image_path")
    local output_filename="${input_filename%.*}_matte.png"
    local output_path="$(dirname "$image_path")/${output_filename}"
    func_1_1_log "   Model: $(basename $model_path), Input: $image_path, Output: $output_path" "yellow"
    func_1_1_log "🚀 Starting inference..." "blue"
    cd "$MODNET_REPO_DIR"
    python -m onnx.inference_onnx --model-path="$model_path" --image-path="$image_path" --output-path="$output_path"
    if [ $? -eq 0 ]; then func_1_1_log "✅ Inference successful!" "green"; else func_1_1_log "❌ Inference failed." "red"; return 1; fi
}

func_2_3_clean_project(){
    # This function cleans up generated files and optionally the entire virtual environment.
    func_1_1_log "🧹 Starting project cleanup..." "blue"

    # Provide an interactive menu for safety
    func_1_1_log "   What would you like to clean?" "yellow"
    echo "   1. Generated models (.onnx files)"
    echo "   2. Python virtual environment (.venv directory)"
    echo "   3. All of the above"
    echo "   4. Cancel"
    read -p "   Please enter your choice [1-4]: " clean_choice

    case $clean_choice in
        1)
            func_1_1_log "   Removing generated .onnx files..." "yellow"
            find "$PRETRAINED_DIR" -maxdepth 1 -type f -name "*.onnx" -print -delete
            ;;
        2)
            if [ -d "$VENV_DIR" ]; then
                func_1_1_log "   Removing Python virtual environment ($VENV_DIR)..." "yellow"
                rm -rf "$VENV_DIR"
            else
                func_1_1_log "   Virtual environment not found. Nothing to do." "green"
            fi
            ;;
        3)
            func_1_1_log "   Performing full cleanup..." "yellow"
            func_1_1_log "   - Removing generated .onnx files..."
            find "$PRETRAINED_DIR" -maxdepth 1 -type f -name "*.onnx" -print -delete || true # Use || true to ignore error if no files found
            if [ -d "$VENV_DIR" ]; then
                func_1_1_log "   - Removing Python virtual environment ($VENV_DIR)..."
                rm -rf "$VENV_DIR"
            else
                func_1_1_log "   - Virtual environment not found. Skipping."
            fi
            ;;
        4 | *)
            func_1_1_log "   Cleanup cancelled." "yellow"
            return 0
            ;;
    esac

    func_1_1_log "✅ Cleanup complete." "green"
}

main(){
    # Load environment variables first, regardless of the command
    func_1_0_load_env

    # --- Argument Parsing ---
    # Check for options like -h, -v before processing commands
    for arg in "$@"; do
        case $arg in
        -h | --help)
            func_1_2_usage
            exit 0
            ;;
        -v)
            V=1
            # Re-load env to apply verbose setting
            func_1_0_load_env
            ;;
        esac
    done

    # --- Command Dispatcher ---
    # If a command is provided, execute it directly.
    # Otherwise, enter interactive mode.
    COMMAND=${1:-"menu"} # Default to 'menu' if no command is given

    case $COMMAND in
        prepare)
            func_2_0_prepare_env
            ;;
        convert)
            func_1_3_check_env # Quick check before running
            func_2_1_ckpt_2_onnx
            ;;
        inference)
            func_1_3_check_env # Quick check before running
            func_2_2_inference_with_onnx
            ;;
        clean)
            func_2_3_clean_project
            ;;
        menu)
            while true; do
                func_1_1_log "\n--- Helmsman Build Menu ---" "blue"
                echo "1. Prepare Environment (run this first!)"
                echo "2. Convert .ckpt model to .onnx"
                echo "3. Run inference with .onnx model"
                echo "4. Clean project (remove generated files)"
                echo "5. Exit"
                read -p "Please enter your choice [1-5]: " choice
                case $choice in
                    1) func_2_0_prepare_env ;;
                    2) func_1_3_check_env && func_2_1_ckpt_2_onnx ;;
                    3) func_1_3_check_env && func_2_2_inference_with_onnx ;;
                    4) func_2_3_clean_project ;;
                    5) func_1_1_log "👋 Exiting." "green"; exit 0 ;;
                    *) func_1_1_log "Invalid choice. Please try again." "red" ;;
                esac
            done
            ;;
        *)
            # Handle unknown commands
            if [[ "$COMMAND" != "-v" ]]; then # Ignore if the only arg was -v
                func_1_1_log "❌ Error: Unknown command '$COMMAND'" "red"
                func_1_2_usage
                exit 1
            fi
            ;;
    esac
}

# Execute the main function with all script arguments
main "$@"