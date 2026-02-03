#!/bin/bash

# ===================================================================================
#
#   Helmsman Project - Development Symlink Setup Script
#
#   Copyright (C) 2025, potterwhite
#   All rights reserved.
#
#   Author: PotterWhite
#   Date: 2025-08-05
#
#   Description:
#   This script creates symbolic links for essential project files, such as
#   configuration files and pre-trained models. It links files from a central
#   source location to the directories where the development tools (e.g.,
#   the MODNet scripts) expect to find them. This avoids duplicating files
#   and keeps a single source of truth.
#
#   Usage:
#   ./setup_links.sh
#
# ===================================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# Get the absolute path of the current script.
BASH_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
BASH_SCRIPT_DIR="$(dirname ${BASH_SCRIPT_PATH})"

# Define paths to key directories.
MODNET_REPO_DIR="${BASH_SCRIPT_DIR}/../../../MODNet.git"
MODNET_MODELS_DIR="${BASH_SCRIPT_DIR}/../../../modnet-models"

# --- Link Definitions ---

# Define the list of symbolic links to create.
# Each line is a single entry with the following space-separated format:
# "SOURCE_FILENAME  SOURCE_DIRECTORY_VARIABLE  DESTINATION_PATH"
declare -a SYMLINKS=(
    "requirements.txt                          ${BASH_SCRIPT_DIR}      ${MODNET_REPO_DIR}/onnx/requirements.txt"
    "export_onnx.py                            ${BASH_SCRIPT_DIR}      ${MODNET_REPO_DIR}/onnx/export_onnx.py"
    "inference_onnx.py                         ${BASH_SCRIPT_DIR}      ${MODNET_REPO_DIR}/onnx/inference_onnx.py"
    "generate_golden_files.py                  ${BASH_SCRIPT_DIR}      ${MODNET_REPO_DIR}/onnx/generate_golden_files.py"
    "mobilenetv2_human_seg.ckpt                ${MODNET_MODELS_DIR}    ${MODNET_REPO_DIR}/pretrained/mobilenetv2_human_seg.ckpt"
    "modnet_photographic_portrait_matting.ckpt ${MODNET_MODELS_DIR}    ${MODNET_REPO_DIR}/pretrained/modnet_photographic_portrait_matting.ckpt"
    "modnet_photographic_portrait_matting.onnx ${MODNET_MODELS_DIR}    ${MODNET_REPO_DIR}/pretrained/modnet_photographic_portrait_matting.onnx"
    "modnet_webcam_portrait_matting.ckpt       ${MODNET_MODELS_DIR}    ${MODNET_REPO_DIR}/pretrained/modnet_webcam_portrait_matting.ckpt"
)


# --- Main Execution Logic ---

echo "🚀 Starting to set up symbolic links..."
# echo "=================================================="

index=0
# Loop through the defined symlinks array.
for entry in "${SYMLINKS[@]}"; do
    echo "=================================================="
    # ((index++))

    # Use 'read' to parse the entry string into three separate variables.
    read -r source_filename source_dir dest_path <<< "$entry"

    # Construct the full, absolute path to the source file.
    full_source_path="${source_dir}/${source_filename}"

    echo "Processing: ${source_filename}"

    # --- Robustness Check 1: Ensure source file exists ---
    if [ ! -f "$full_source_path" ]; then
        echo -e "\033[0;31m   ❌ ERROR: Source file not found: ${full_source_path}. Skipping.\033[0m"
        echo "=================================================="
        continue
    fi

    # Get the directory part of the destination path.
    dest_dir=$(dirname "$dest_path")

    # --- Robustness Check 2: Ensure destination directory exists ---
    if [ ! -d "$dest_dir" ]; then
        echo -e "\033[0;33m   ⚠️  WARN: Destination directory not found. Creating: ${dest_dir}\033[0m"
        mkdir -p "$dest_dir"
    fi

    # echo "   Linking: ${full_source_path}"
    # echo "        ->   ${dest_path}"

    # --- Action: Create the symbolic link ---
    # '-f' option removes the destination file if it exists.
    # '-s' option creates a symbolic link.
    ln -sf "${full_source_path}" "${dest_path}"

    # Verify the link was created successfully.
    # 'ls -l' provides a clear view of the link and its target.
    echo
    ls -l "${dest_path}"

done

echo "=================================================="
echo
echo -e "\033[0;32m✅ All symbolic links processed successfully.\033[0m"


#################################################


# #!/bin/bash

# set -e

# BASH_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
# BASH_SCRIPT_DIR="$(dirname ${BASH_SCRIPT_PATH})"
# MODNET_REPO_DIR="${BASH_SCRIPT_DIR}/../../../MODNet.git"

# # ----- 1st file -----------
# REQUIREMENTS_FILE_NAME="requirements.txt"
# REQUIREMENTS_FILE_PATH="${MODNET_REPO_DIR}/onnx/${REQUIREMENTS_FILE_NAME}"
# # ----- 2nd file 1st ckpt-----------
# SECOND_FILE_NAME="mobilenetv2_human_seg.ckpt"
# SECOND_FILE_PATH="${MODNET_REPO_DIR}/pretrained/${SECOND_FILE_NAME}"
# # ----- 3rd file 2nd ckpt-----------
# THIRD_FILE_NAME="modnet_photographic_portrait_matting.ckpt"
# THIRD_FILE_PATH="${MODNET_REPO_DIR}/pretrained/${THIRD_FILE_NAME}"
# # ----- 4th file 1st onnx -----------
# FOURTH_FILE_NAME="modnet_photographic_portrait_matting.onnx"
# FOURTH_FILE_PATH="${MODNET_REPO_DIR}/pretrained/${FOURTH_FILE_NAME}"
# # ----- 5th file 3rd ckpt-----------
# FIFTH_FILE_NAME="modnet_webcam_portrait_matting.ckpt"
# FIFTH_FILE_PATH="${MODNET_REPO_DIR}/pretrained/${FIFTH_FILE_NAME}"

# for

# rm -f ${REQUIREMENTS_FILE_PATH}
# ln -sf "${BASH_SCRIPT_DIR}/${REQUIREMENTS_FILE_NAME}" ${REQUIREMENTS_FILE_PATH}
# ls -lha ${REQUIREMENTS_FILE_PATH}
# echo
# echo "setup ${REQUIREMENTS_FILE_PATH} done"
# echo
