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
# Level 4: Core Workflows (MODNet & C++)
# ==============================================================================

#------------------------------------------------------------------------------
# Convert PyTorch checkpoint (.ckpt) to ONNX model
# Flow: Select variant -> Select ckpt -> Set output path -> Run conversion
#------------------------------------------------------------------------------
func_4_1_ckpt_2_onnx(){
    # Step 1: Start time counter for performance tracking
    func_1_4_start_time_count __TASK_START_TIME

    # Step 2: Ask user to select which modnet_onnx.py version to use
    #         This determines the ONNX export logic (original or modified)
    func_1_1_log "Select MODNet variant:" "blue"
    echo "1) Original (modnet_onnx.py)"
    echo "2) Modified (modnet_onnx_modified.py) — anti-fusion, for pretrained IBNorm ckpt"
    echo "3) Pure-BN  (export_onnx_pureBN.py)  — for retrained BatchNorm-only ckpt"
    read -p "   Select [1-3]: " variant_choice

    # Step 3: Based on user selection, determine the export script filename
    #         and set a suffix for the output ONNX filename
    local onnx_suffix=""
    local export_module="onnx.export_onnx"
    case $variant_choice in
        2)
            func_1_1_log "   Using: Modified variant (anti-fusion IBNorm)" "yellow"
            onnx_suffix="_modified"
            export_module="onnx.export_onnx_modified"
            ;;
        3)
            func_1_1_log "   Using: Pure-BN variant (retrained BatchNorm-only)" "yellow"
            onnx_suffix="_pureBN"
            export_module="onnx.export_onnx_pureBN"
            ;;
        *)
            func_1_1_log "   Using: Original variant" "yellow"
            ;;
    esac

    # Step 4: Scan the pretrained model directory for available .ckpt files
    func_1_1_log "🔎 Checking for pre-trained models (.ckpt)..." "blue"
    mapfile -t ckpt_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -type l -name "*.ckpt" -o -name "*.ckpt")

    # If no symlink found, try regular files
    if [ ${#ckpt_files[@]} -eq 0 ]; then
        mapfile -t ckpt_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -type f -name "*.ckpt")
    fi

    # Exit if no checkpoint files found
    if [ ${#ckpt_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .ckpt models found in '$LV5_PRETRAINED_DIR'." "red"
        return 1
    fi

    # Step 5: Prompt user to select which checkpoint to convert
    func_1_1_log "   Please choose a .ckpt model to convert to ONNX:" "green"
    select ckpt_path in "${ckpt_files[@]}"; do
        if [[ -n "$ckpt_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    # Step 6: Construct the output ONNX filename
    #         If Modified variant was selected, append "_modified" to filename
    local ckpt_filename=$(basename "$ckpt_path")
    local onnx_filename="${ckpt_filename%.ckpt}${onnx_suffix}.onnx"
    local onnx_path="${LV5_PRETRAINED_DIR}/${onnx_filename}"

    func_1_1_log "   Selected CKPT: $ckpt_filename" "yellow"
    func_1_1_log "   Output ONNX:   $onnx_path" "yellow"

    # Step 7: Check if output file already exists, ask for confirmation to overwrite
    if [ -f "$onnx_path" ]; then
        read -p "   File exists. Overwrite? (y/N) " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return 0; fi
    fi

    # Step 8: Execute the ONNX export using the selected variant
    #         Use the export module determined by user choice (original or modified)
    func_1_1_log "🚀 Starting conversion..." "blue"
    cd "$LV4_MODNET_SDK_DIR"
    # PYTHONPATH=$LV4_MODNET_SDK_DIR ensures local onnx/ package takes priority
    # over the installed onnx pip package (both named 'onnx', local one wins)
    PYTHONPATH="$LV4_MODNET_SDK_DIR" "$PYTHON_BIN" -m ${export_module} --ckpt-path="$ckpt_path" --output-path="$onnx_path"

    # Step 9: Report final result to user
    if [ $? -eq 0 ]; then func_1_1_log "✅ Conversion successful!" "green"; else func_1_2_err "Conversion failed."; fi
}

#------------------------------------------------------------------------------
# Run inference using ONNX model on a single image
# Flow: Select ONNX model -> Select image -> Run inference -> Save output
#------------------------------------------------------------------------------
func_4_2_inference_with_onnx(){
    # Step 1: Start time counter
    func_1_4_start_time_count __TASK_START_TIME

    # Step 2: Scan for available ONNX models in the pretrained directory
    func_1_1_log "🔎 Checking for ONNX models..." "blue"
    mapfile -t onnx_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -name "*.onnx")
    if [ ${#onnx_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .onnx models found. Run 'convert' first." "red"
        return 1
    fi

    # Step 3: Prompt user to select which ONNX model to use
    func_1_1_log "   Choose ONNX model:" "green"
    select model_path in "${onnx_files[@]}"; do
        if [[ -n "$model_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    # Step 4: Scan the media directory for input images (jpg, bmp, png)
    func_1_1_log "🔎 Scanning for images..." "blue"
    mapfile -t images_list < <(find "${LV1_MEDIA_DIR}" -type f \( -name "*.jpg" -o -name "*.bmp" -o -name "*.png" \))

    if [ ${#images_list[@]} -eq 0 ]; then
         func_1_1_log "❌ No images found in ${LV1_MEDIA_DIR}" "red"; return 1;
    fi

    # Step 5: Prompt user to select which image to run inference on
    func_1_1_log "   Choose input image:" "green"
    select image_path in "${images_list[@]}"; do
        if [[ -n "$image_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    # Step 6: Construct output filename (input name + "_matte" suffix)
    local input_filename=$(basename "$image_path")
    local output_filename="${input_filename%.*}_matte.png"
    local output_path="$(dirname "$image_path")/${output_filename}"

    # Step 7: Run inference
    func_1_1_log "🚀 Starting inference..." "blue"
    cd "$LV4_MODNET_SDK_DIR"
    "$PYTHON_BIN" -m onnx.inference_onnx --model-path="$model_path" --image-path="$image_path" --output-path="$output_path"

    # Step 8: Report final result
    if [ $? -eq 0 ]; then func_1_1_log "✅ Inference successful! Saved to: $output_path" "green"; else func_1_2_err "Inference failed."; fi
}

#------------------------------------------------------------------------------
# Generate golden reference files (input image + expected ONNX output)
# Flow: Select image -> Select ONNX model -> Generate output -> Save to golden dir
#------------------------------------------------------------------------------
func_4_5_generate_golden_interactive() {
    # Step 1: Start time counter
    func_1_4_start_time_count __TASK_START_TIME

    # Step 2: Ensure golden output directories exist
    mkdir -p "${LV2_GOLDEN_DIR}" "${LV2_GOLDEN_DEBUG_DIR}"

    # Step 3: Scan media directory for input images
    mapfile -t images < <(find "${LV1_MEDIA_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.bmp" \))

    if [ ${#images[@]} -eq 0 ]; then
        func_1_2_err "No images found under ${LV1_MEDIA_DIR}"
    fi

    # Step 4: Prompt user to select input image
    func_1_1_log "📷 Select input image (from media/):" "green"
    select image_path in "${images[@]}"; do
        [[ -n "$image_path" ]] && break
        func_1_1_log "Invalid selection." "red"
    done

    # Step 5: Scan for available ONNX models
    mapfile -t onnx_models < <(find "${LV1_MODELS_DOWNLOAD_DIR}" -type f -name "*.onnx")

    if [ ${#onnx_models[@]} -eq 0 ]; then
        func_1_2_err "No ONNX model found. Please convert first."
    fi

    # Step 6: Prompt user to select ONNX model
    func_1_1_log "🧠 Select ONNX model:" "green"
    select model_path in "${onnx_models[@]}"; do
        [[ -n "$model_path" ]] && break
        func_1_1_log "Invalid selection." "red"
    done

    # Step 7: Construct output paths for generated files
    local image_name=$(basename "$image_path")
    local stem="${image_name%.*}"

    local output_path="${LV2_GOLDEN_DIR}/${stem}_matte.png"
    local debug_path="${LV2_GOLDEN_DEBUG_DIR}"

    # Step 8: Run golden file generator
    func_1_1_log "🚀 Generating golden files..." "blue"
    func_1_1_log "   Input : ${image_path}" "yellow"
    func_1_1_log "   Output: ${output_path}" "yellow"

    cd "${LV4_MODNET_SDK_DIR}"

    "${PYTHON_BIN}" onnx/generate_golden_files.py \
        --image-path "${image_path}" \
        --model-path "${model_path}" \
        --output-path "${output_path}" \
        --debug-file-path "${debug_path}"

    # Step 9: Report final result
    func_1_1_log "✅ Golden files generated successfully." "green"
}
