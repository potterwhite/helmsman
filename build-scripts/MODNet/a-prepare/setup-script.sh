#!/bin/bash

set -e

BASH_SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
BASH_SCRIPT_DIR="$(dirname ${BASH_SCRIPT_PATH})"
MODNET_REPO_DIR="${BASH_SCRIPT_DIR}/../../../MODNet.git"

# Define an array of "structs" (file name and path pairs)
declare -A files=(
    ["requirements.txt,${MODNET_REPO_DIR}/onnx/requirements.txt"]="1"
    ["mobilenetv2_human_seg.ckpt,${MODNET_REPO_DIR}/pretrained/mobilenetv2_human_seg.ckpt"]="1"
    ["modnet_photographic_portrait_matting.ckpt,${MODNET_REPO_DIR}/pretrained/modnet_photographic_portrait_matting.ckpt"]="1"
    ["modnet_photographic_portrait_matting.onnx,${MODNET_REPO_DIR}/pretrained/modnet_photographic_portrait_matting.onnx"]="1"
    ["modnet_webcam_portrait_matting.ckpt,${MODNET_REPO_DIR}/pretrained/modnet_webcam_portrait_matting.ckpt"]="1"
)

# Loop through the files
for entry in "${!files[@]}"; do
    # Split the entry into name and path
    IFS=',' read -r file_name file_path <<< "$entry"
    rm -f "${file_path}"
    ln -sf "${BASH_SCRIPT_DIR}/${file_name}" "${file_path}"
    ls -lha "${file_path}"
    echo
    echo "Setup ${file_name} done"
    echo
done





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
