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
    BUILD_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
    BUILD_SCRIPT_DIR="$(dirname ${BUILD_SCRIPT_PATH})"
    REPO_TOP_DIR="${BUILD_SCRIPT_DIR}/../../.."
    MODNET_REPO_DIR="${REPO_TOP_DIR}/MODNet.git"
    PRETRAINED_DIR="${MODNET_REPO_DIR}/pretrained"
    VENV_DIR="${REPO_TOP_DIR}/.venv"
    DEV_REQUIREMENTS_FILE="${MODNET_REPO_DIR}/onnx/requirements.txt"
    PYTHON_CMD="python3"
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
    func_1_1_log "   [1/2] Installing system dependencies (e.g., cmake)..." "yellow"
    sudo apt update && sudo apt-get install -y cmake
    func_1_1_log "✅ System dependencies are up to date." "green"

    # --- Step 2: Set up Python virtual environment ---
    func_1_1_log "   [2/2] Setting up Python virtual environment..." "yellow"
    if ! command -v "$PYTHON_CMD" &>/dev/null; then
        func_1_1_log "❌ Error: '$PYTHON_CMD' could not be found." "red"
        exit 1
    fi
    if [ ! -d "$VENV_DIR" ]; then
        func_1_1_log "   Creating virtual environment in '$VENV_DIR'..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
        func_1_1_log "   Installing Python packages from '$DEV_REQUIREMENTS_FILE'..."
        pip install --no-cache-dir -r "$DEV_REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    func_1_1_log "✅ Python virtual environment is ready." "green"
    func_1_1_log "🎉 Preparation complete! You can now run 'convert' or 'inference'." "green"
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
    python -m onnx.export_onnx --ckpt-path="$ckpt_path" --output-path="$onnx_path"
    if [ $? -eq 0 ]; then func_1_1_log "✅ Conversion successful!" "green"; else func_1_1_log "❌ Conversion failed." "red"; return 1; fi
}

func_2_2_inference_with_onnx(){
    # This function runs inference on a single image using a selected .onnx model.
    func_1_1_log "🔎 Checking for ONNX models (.onnx)..." "blue"
    mapfile -t onnx_files < <(find "$PRETRAINED_DIR" -maxdepth 1 -type f -name "*.onnx")
    if [ ${#onnx_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .onnx models found in '$PRETRAINED_DIR'." "red";
        func_1_1_log "   Please run the 'convert' command first." "yellow"; return 1;
    fi
    func_1_1_log "   Please choose an ONNX model for inference:" "green"
    select model_path in "${onnx_files[@]}"; do
        if [[ -n "$model_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done
    read -p "   Please enter the path to the input image: " image_path
    if [ ! -f "$image_path" ]; then func_1_1_log "❌ Input image not found at '$image_path'" "red"; return 1; fi
    local input_filename=$(basename "$image_path")
    local output_filename="${input_filename%.*}_matte.png"
    local output_path="$(dirname "$image_path")/${output_filename}"
    func_1_1_log "   Model: $(basename $model_path), Input: $image_path, Output: $output_path" "yellow"
    func_1_1_log "🚀 Starting inference..." "blue"
    python -m onnx.inference_onnx --model-path="$model_path" --image-path="$image_path" --output-path="$output_path"
    if [ $? -eq 0 ]; then func_1_1_log "✅ Inference successful!" "green"; else func_1_1_log "❌ Inference failed." "red"; return 1; fi
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
        menu)
            # --- Interactive Menu ---
            while true; do
                func_1_1_log "\n--- Helmsman Build Menu ---" "blue"
                echo "1. Prepare Environment (run this first!)"
                echo "2. Convert .ckpt model to .onnx"
                echo "3. Run inference with .onnx model"
                echo "4. Exit"
                read -p "Please enter your choice [1-4]: " choice

                case $choice in
                    1) func_2_0_prepare_env ;;
                    2) func_1_3_check_env && func_2_1_ckpt_2_onnx ;;
                    3) func_1_3_check_env && func_2_2_inference_with_onnx ;;
                    4) func_1_1_log "👋 Exiting." "green"; exit 0 ;;
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