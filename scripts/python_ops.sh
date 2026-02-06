
# ==============================================================================
# Level 4: Core Workflows (MODNet & C++)
# ==============================================================================

func_4_1_ckpt_2_onnx(){
    # (From A)
    func_1_4_start_time_count __TASK_START_TIME

    func_1_1_log "🔎 Checking for pre-trained models (.ckpt)..." "blue"
    mapfile -t ckpt_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -type l -name "*.ckpt" -o -name "*.ckpt")

    if [ ${#ckpt_files[@]} -eq 0 ]; then
        # Try finding regular files if symlinks fail
        mapfile -t ckpt_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -type f -name "*.ckpt")
    fi

    if [ ${#ckpt_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .ckpt models found in '$LV5_PRETRAINED_DIR'." "red"
        return 1
    fi

    func_1_1_log "   Please choose a .ckpt model to convert to ONNX:" "green"
    select ckpt_path in "${ckpt_files[@]}"; do
        if [[ -n "$ckpt_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    local ckpt_filename=$(basename "$ckpt_path")
    local onnx_filename="${ckpt_filename%.ckpt}.onnx"
    local onnx_path="${LV5_PRETRAINED_DIR}/${onnx_filename}"

    func_1_1_log "   Selected CKPT: $ckpt_filename" "yellow"
    func_1_1_log "   Output ONNX:   $onnx_path" "yellow"

    if [ -f "$onnx_path" ]; then
        read -p "   File exists. Overwrite? (y/N) " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return 0; fi
    fi

    func_1_1_log "🚀 Starting conversion..." "blue"
    cd "$LV4_MODNET_SDK_DIR"
    # Use the venv python explicitly
    "$PYTHON_BIN" -m onnx.export_onnx --ckpt-path="$ckpt_path" --output-path="$onnx_path"

    if [ $? -eq 0 ]; then func_1_1_log "✅ Conversion successful!" "green"; else func_1_2_err "Conversion failed."; fi
}

func_4_2_inference_with_onnx(){
    # (From A)
    func_1_4_start_time_count __TASK_START_TIME

    func_1_1_log "🔎 Checking for ONNX models..." "blue"
    # mapfile -t onnx_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -type f -name "*.onnx")
    mapfile -t onnx_files < <(find "$LV5_PRETRAINED_DIR" -maxdepth 1 -name "*.onnx")
    if [ ${#onnx_files[@]} -eq 0 ]; then
        func_1_1_log "❌ No .onnx models found. Run 'convert' first." "red"
        return 1
    fi

    func_1_1_log "   Choose ONNX model:" "green"
    select model_path in "${onnx_files[@]}"; do
        if [[ -n "$model_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    func_1_1_log "🔎 Scanning for images..." "blue"
    mapfile -t images_list < <(find "${LV1_MEDIA_DIR}" -type f \( -name "*.jpg" -o -name "*.bmp" -o -name "*.png" \))

    if [ ${#images_list[@]} -eq 0 ]; then
         func_1_1_log "❌ No images found in ${LV1_MEDIA_DIR}" "red"; return 1;
    fi

    func_1_1_log "   Choose input image:" "green"
    select image_path in "${images_list[@]}"; do
        if [[ -n "$image_path" ]]; then break; else func_1_1_log "Invalid selection." "red"; fi
    done

    local input_filename=$(basename "$image_path")
    local output_filename="${input_filename%.*}_matte.png"
    local output_path="$(dirname "$image_path")/${output_filename}"

    func_1_1_log "🚀 Starting inference..." "blue"
    cd "$LV4_MODNET_SDK_DIR"
    "$PYTHON_BIN" -m onnx.inference_onnx --model-path="$model_path" --image-path="$image_path" --output-path="$output_path"

    if [ $? -eq 0 ]; then func_1_1_log "✅ Inference successful! Saved to: $output_path" "green"; else func_1_2_err "Inference failed."; fi
}

func_4_5_generate_golden_interactive() {
    func_1_4_start_time_count __TASK_START_TIME

    mkdir -p "${LV2_GOLDEN_DIR}" "${LV2_GOLDEN_DEBUG_DIR}"

    # --------------------------------------------------
    # 1. Scan media directory
    # --------------------------------------------------
    mapfile -t images < <(find "${LV1_MEDIA_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.bmp" \))

    if [ ${#images[@]} -eq 0 ]; then
        func_1_2_err "No images found under ${LV1_MEDIA_DIR}"
    fi

    func_1_1_log "📷 Select input image (from media/):" "green"
    select image_path in "${images[@]}"; do
        [[ -n "$image_path" ]] && break
        func_1_1_log "Invalid selection." "red"
    done

    # --------------------------------------------------
    # 2. Select ONNX model
    # --------------------------------------------------
    mapfile -t onnx_models < <(find "${LV1_MODELS_DOWNLOAD_DIR}" -type f -name "*.onnx")

    if [ ${#onnx_models[@]} -eq 0 ]; then
        func_1_2_err "No ONNX model found. Please convert first."
    fi

    func_1_1_log "🧠 Select ONNX model:" "green"
    select model_path in "${onnx_models[@]}"; do
        [[ -n "$model_path" ]] && break
        func_1_1_log "Invalid selection." "red"
    done

    # --------------------------------------------------
    # 3. Auto-generate output paths
    # --------------------------------------------------
    local image_name=$(basename "$image_path")
    local stem="${image_name%.*}"

    local output_path="${LV2_GOLDEN_DIR}/${stem}_matte.png"
    # local debug_path="${LV2_GOLDEN_DEBUG_DIR}/${stem}_debug.npy"
    local debug_path="${LV2_GOLDEN_DEBUG_DIR}"

    func_1_1_log "🚀 Generating golden files..." "blue"
    func_1_1_log "   Input : ${image_path}" "yellow"
    func_1_1_log "   Output: ${output_path}" "yellow"

    # --------------------------------------------------
    # 4. Run MODNet golden generator
    # --------------------------------------------------
    cd "${LV4_MODNET_SDK_DIR}"

    "${PYTHON_BIN}" onnx/generate_golden_files.py \
        --image-path "${image_path}" \
        --model-path "${model_path}" \
        --output-path "${output_path}" \
        --debug-file-path "${debug_path}"

    func_1_1_log "✅ Golden files generated successfully." "green"
}
