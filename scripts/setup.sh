
# ==============================================================================
# Level 3: Environment Preparation Logic
# ==============================================================================

func_3_0_setup_modnet_softlinks() {
    func_1_1_log "🔗 Setting up symbolic links and assets..." "blue"

    # Ensure the model source directory exists
    if [ ! -d "${LV1_MODELS_DOWNLOAD_DIR}" ]; then
        mkdir -p "${LV1_MODELS_DOWNLOAD_DIR}"
    fi

    # =========================================================
    # Part 1: Local Script Links (Source: BUILD_SCRIPT_DIR)
    # =========================================================

    #   1ST field                       2nd Field                     3rd field
    #   file name                       source dir                    destination dir
    declare -a SCRIPT_LINKS=(
        "requirements.txt          ${LV1_ENVS_DIR}              ${LV4_MODNET_SDK_DIR}/onnx/requirements.txt"
        "export_onnx.py            ${LV4_MODNET_SCRIPTS_DIR}    ${LV4_MODNET_SDK_DIR}/onnx/export_onnx.py"
        "inference_onnx.py         ${LV4_MODNET_SCRIPTS_DIR}    ${LV4_MODNET_SDK_DIR}/onnx/inference_onnx.py"
        "generate_golden_files.py  ${LV4_MODNET_SCRIPTS_DIR}    ${LV4_MODNET_SDK_DIR}/onnx/generate_golden_files.py"
    )

    for entry in "${SCRIPT_LINKS[@]}"; do
        read -r filename src_dir dest_path <<< "$entry"
        # func_1_1_log "+++++ filename=${filename}; src_dir=${src_dir} ; dest_path=${dest_path}"
        local full_source_path="${src_dir}/${filename}"
        local dest_dir=$(dirname "$dest_path")

        # Robustness: Check source
        if [ ! -f "$full_source_path" ]; then
            func_1_1_log "   ❌ Script ${filename} not found in ${src_dir}" "red"
            continue
        fi

        # Robustness: Create dest dir
        if [ ! -d "$dest_dir" ]; then mkdir -p "$dest_dir"; fi

        # Action: Link
        ln -sf "${full_source_path}" "${dest_path}"

        # Check Link and Echo
        ln -sf "${full_source_path}" "${dest_path}"
        if [ -L "${dest_path}" ]; then
            func_1_1_log "   ✅ Linked: ${filename}" "green"
        fi
    done

    # =========================================================
    # Part 2: Model Links with Auto-Download
    # =========================================================

    # --- [CONFIG] HARDCODE YOUR URLs HERE ---
    local URL_MOBILENET="https://huggingface.co/PotterWhite/MODNet/resolve/main/mobilenetv2_human_seg.ckpt"
    local URL_PHOTO_CKPT="https://huggingface.co/PotterWhite/MODNet/resolve/main/modnet_photographic_portrait_matting.ckpt"
    local URL_PHOTO_ONNX="https://huggingface.co/PotterWhite/MODNet/resolve/main/modnet_photographic_portrait_matting.onnx"
    local URL_WEBCAM="https://huggingface.co/PotterWhite/MODNet/resolve/main/modnet_webcam_portrait_matting.ckpt"
    # ----------------------------------------

    # Internal helper to handle the logic: Check -> Wget -> Link
    _process_model() {
        local filename="$1"
        local url="$2"
        local dest_path="$3"
        local full_source_path="${LV1_MODELS_DOWNLOAD_DIR}/${filename}"
        local dest_dir=$(dirname "$dest_path")

        # 1. Check if source exists; if not, download
        if [ ! -f "$full_source_path" ]; then
            func_1_1_log "   📥 Model missing: ${filename}" "yellow"

            if [[ "$url" == "YOUR_URL_HERE" || -z "$url" ]]; then
                func_1_1_log "   ⚠️  URL not set. Skipping download." "yellow"
                return
            fi

            func_1_1_log "      Downloading from: $url" "blue"
            if ! command -v wget > /dev/null; then
                func_1_2_err "wget not found. Please install it."
            fi

            wget -O "$full_source_path" "$url"
            if [ $? -ne 0 ]; then
                func_1_1_log "      ❌ Download failed." "red"
                rm -f "$full_source_path" # Remove empty/corrupt file
                return
            fi
            func_1_1_log "   ✅ Download success." "green"
        fi

        # 2. Create Destination Directory
        if [ ! -d "$dest_dir" ]; then mkdir -p "$dest_dir"; fi

        # 3. Create Symlink
        if [ -f "$full_source_path" ]; then
            ln -sf "${full_source_path}" "${dest_path}"
            if [ -L "${dest_path}" ]; then
                func_1_1_log "   ✅ Linked: ${filename}" "green"
            fi
        fi
    }

    # --- Execute Process (Explicit Calls) ---

    _process_model \
        "mobilenetv2_human_seg.ckpt" \
        "$URL_MOBILENET" \
        "${LV4_MODNET_SDK_DIR}/pretrained/mobilenetv2_human_seg.ckpt"

    _process_model \
        "modnet_photographic_portrait_matting.ckpt" \
        "$URL_PHOTO_CKPT" \
        "${LV4_MODNET_SDK_DIR}/pretrained/modnet_photographic_portrait_matting.ckpt"

    _process_model \
        "modnet_photographic_portrait_matting.onnx" \
        "$URL_PHOTO_ONNX" \
        "${LV4_MODNET_SDK_DIR}/pretrained/modnet_photographic_portrait_matting.onnx"

    _process_model \
        "modnet_webcam_portrait_matting.ckpt" \
        "$URL_WEBCAM" \
        "${LV4_MODNET_SDK_DIR}/pretrained/modnet_webcam_portrait_matting.ckpt"

    func_1_1_log "✅ Link setup and model verification complete." "green"
}

func_3_1_check_opencv_compatibility() {
    # (Transplanted from B)
    # Checks if OpenCV works. If not (common in servers), switches to headless.
    func_1_1_log "   Verifying OpenCV environment inside the venv..." "blue"

    # First try with whatever is currently installed
    if "${PYTHON_BIN}" -c "import cv2" &> /dev/null; then
        func_1_1_log "   ✅ OpenCV import succeeded." "green"
        return 0
    fi

    func_1_1_log "   ⚠️ OpenCV import failed (likely missing GUI libs). Switching to opencv-python-headless..." "yellow"

    # Switch to headless
    "${PIP_BIN}" uninstall -y opencv-python || true
    "${PIP_BIN}" install "opencv-python-headless==4.5.1.48" || func_1_2_err "Failed to install opencv-python-headless."

    # Re-check
    if "${PYTHON_BIN}" -c "import cv2" &> /dev/null; then
        func_1_1_log "   ✅ OpenCV (headless) import succeeded." "green"
        return 0
    fi

    func_1_2_err "OpenCV environment check failed even after installing headless version."
}

func_3_2_setup_dep_before_build() {
    # (Merged A & B)
    func_1_1_log "🛠️  Starting full environment preparation..." "blue"

    # ********************************************
    # --- Step 1: System Dependencies (From A) ---
    func_1_1_log "   [1/7] Installing system dependencies..." "yellow"
    sudo apt update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
    cmake libgl1
    func_1_1_log "   ✅ System dependencies installed." "green"

    # ********************************************
    # --- Step 2: Pyenv Setup (From A & B) ---
    func_1_1_log "   [2/7] Setting up pyenv..." "yellow"
    if [ ! -d "$PYENV_ROOT" ]; then
        curl https://pyenv.run | bash
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    else
        func_1_1_log "   Pyenv already installed." "green"
    fi

    # Install Python Version if missing
    if ! pyenv versions --bare | grep -q "^${PYTHON_Target_VERSION}$"; then
        func_1_1_log "   Installing Python ${PYTHON_Target_VERSION}..."
        pyenv install ${PYTHON_Target_VERSION}
    else
        func_1_1_log "   Python ${PYTHON_Target_VERSION} found." "green"
    fi

    # ********************************************
    # --- Step 3: Virtual Environment (From A, enhanced) ---
    func_1_1_log "   [3/7] Setting up Virtual Environment..." "yellow"
    if [ ! -d "$LV1_VENV_DIR" ]; then
        func_1_1_log "   Creating venv in '$LV1_VENV_DIR'..."
        pyenv local ${PYTHON_Target_VERSION}
        python -m venv "$LV1_VENV_DIR"
        pyenv local --unset
    fi

    # Ensure pip is up to date
    "${PIP_BIN}" install --upgrade pip

    # Handle Requirements (Load from file only)
    if [ -f "$DEV_REQUIREMENTS_FILE" ]; then
        func_1_1_log "   Installing Python packages from $DEV_REQUIREMENTS_FILE..."
        # "${PIP_BIN}" install --no-cache-dir -r "$DEV_REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
        "${PIP_BIN}" install -r "$DEV_REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
        func_1_1_log "❌ Requirements file not found at: $DEV_REQUIREMENTS_FILE" "red"
        func_1_1_log "   Please create the file manually before running prepare." "yellow"
        exit 1
    fi

    # Auto-fix OpenCV if needed (From B)
    # This remains critical: even if requirements.txt installed opencv-python,
    # this function will swap it for headless if the GUI version fails to load.
    func_3_1_check_opencv_compatibility

    func_1_1_log "   ✅ Python environment ready." "green"

    func_1_8_activate_py_venv

    # ********************************************
    # --- Step 4: Conan (From A) ---
    func_1_1_log "   [4/7] Setting up Conan..." "yellow"

    # Processing -- 4.1 install conan in venv
    if ! command -v conan &> /dev/null; then
        "${PIP_BIN}" install conan
    fi
    # Check if profile exists to avoid error
    if command -v conan &> /dev/null; then
         conan profile detect --force 2>/dev/null || true
         func_1_1_log "   ✅ Conan ready." "green"
    fi

    # Preparation -- 4.2 extract the ort version from current venv
    func_3_4_detect_onnxruntime_version

    # Processing -- 4.3 Manually install ort into conan
    func_3_5_build_onnxruntime_conan_package

    # ********************************************
    # --- Step 5: build dir is temporary for restoring temp files and folders
    if [ ! -d "${LV1_BUILD_DIR}" ];then
        mkdir -p "${LV1_BUILD_DIR}"
        func_1_1_log "   [5/7] Setting up build dir successfully..." "yellow"
    fi

    # ********************************************
    # --- Step 6: Sync MODNet sdk as needed

    func_1_1_log "   [6/7] Target Submodule: ${LV4_MODNET_SDK_DIR}" "yellow"
    func_3_3_rebuild_sdk modnet









    # if [ ! -d "${LV4_MODNET_SDK_DIR}" ]; then
    #     cd "${REPO_TOP_DIR}"
    #     git submodule update
    #     func_1_1_log "   [6/7] Synchronized MODNet SDK Complete." "yellow"
    # else
    #     func_1_1_log "   [6/7] MODNet SDK Exist, Skip Downloading." "yellow"
    # fi

    # ********************************************
    # --- Step 7: MODNet Links for furthur executing
    func_1_1_log "   [7/7] Setting up Softlinks for MODNet SDK..." "yellow"
    func_3_0_setup_modnet_softlinks

    # ********************************************
    # --- Finilization
    func_1_1_log "🎉 Preparation complete!" "green"
}

func_3_3_rebuild_sdk(){
    if [ x"$1" == x"modnet" ]; then
        # ********************************************
        # --- Step 6: Force Re-install MODNet sdk
        cd "${REPO_TOP_DIR}"

        # 定义变量 (确保完全匹配)
        REL_PATH="third-party/sdk/MODNet.git"
        REMOTE_URL="https://github.com/ZHKKKe/MODNet.git"

        # ==========================================
        # Phase 1: 彻底清理所有残留 (Nuclear Clean)
        # ==========================================

        # 1. 尝试从 git 注册表中反初始化
        git submodule deinit -f -- "${REL_PATH}" 2>/dev/null || true

        # 2. 从 Git 索引(Index)中强制移除 (这一步解决 pathspec error)
        # 即使它不在索引里，加了 || true 也不会报错
        git rm --cached -f "${REL_PATH}" 2>/dev/null || true

        # 3. 物理删除工作目录
        rm -rf "${REL_PATH}"

        # 4. 【关键】清理 .git/modules 缓存
        # 根据你提供的 tree 信息，这里可能有残留，我们把相关的全删了
        rm -rf ".git/modules/third-party/sdk/MODNet.git"
        # 你截图里多出来的那个奇怪目录也删掉，防止干扰
        rm -rf ".git/modules/MODNet.git"

        func_1_1_log "   Cleanup complete. Re-adding from scratch..." "yellow"

        # ==========================================
        # Phase 2: 全新添加 (Re-Add)
        # ==========================================

        # 使用 'git submodule add' 而不是 'update'
        # 'add' 命令会自动做三件事：
        # 1. 下载代码
        # 2. 写入 .gitmodules
        # 3. 将 submodule 信息写入 Git 索引 (修复你的核心问题)
        # --force 允许我们在该目录被 git 认为是 ignored 或有残留时强制执行

        if git submodule add --force "${REMOTE_URL}" "${REL_PATH}"; then
            func_1_1_log "   Submodule added successfully." "green"
        else
            func_1_1_log "   'git submodule add' failed. Checking if it's already there..." "red"
            # 如果 add 失败，可能是因为清理不彻底，或者网络问题，这里做一个兜底尝试
            git submodule update --init --recursive --force -- "${REL_PATH}"
        fi

        # 最后的确认
        if [ -f "${REL_PATH}/.git" ]; then
            func_1_1_log " [6/7] MODNet SDK force synced OK." "yellow"
        else
            func_1_1_log " [6/7] Force sync failed!" "red"
            exit 1
        fi
    fi
}

func_3_4_detect_onnxruntime_version() {
    func_1_1_log "🔍 Detecting ONNX Runtime version from Python venv..." "blue"

    # Sanity check: pip/python must exist
    if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
        func_1_2_err "$PYTHON_BIN is not set or not executable. Cannot detect onnxruntime version."
    fi

    # Use the python behind pip to query onnxruntime version
    local ort_version
    ort_version=$("${PYTHON_BIN}" -c "import onnxruntime as ort; print(ort.__version__)" 2>/dev/null)

    if [ -z "${ort_version}" ]; then
        func_1_2_err "Failed to detect onnxruntime version from venv. Is onnxruntime installed?"
    fi

    # global env (caller decides where/how it's consumed)
    ONNXRUNTIME_VERSION="${ort_version}"
    export ONNXRUNTIME_VERSION

    func_1_1_log "   ✔ ONNX Runtime version detected: ${ONNXRUNTIME_VERSION}" "green"
}

func_3_5_build_onnxruntime_conan_package() {
    func_1_1_log "📦 Building ONNX Runtime Conan package..." "blue"

    # Sanity check: version must exist
    if [ -z "${ONNXRUNTIME_VERSION}" ]; then
        func_1_2_err "ONNXRUNTIME_VERSION is not set. Cannot build ONNX Runtime Conan package."
    fi

    local recipe_dir="${REPO_TOP_DIR}/runtime/cpp/third_party/conan_recipes/onnxruntime"

    if [ ! -f "${recipe_dir}/conanfile.py" ]; then
        func_1_2_err "conanfile.py not found in ${recipe_dir}"
    fi

    func_1_1_log "   Recipe: ${recipe_dir}" "yellow"
    func_1_1_log "   Version: ${ONNXRUNTIME_VERSION}" "yellow"

    pushd "${recipe_dir}" >/dev/null || func_1_2_err "Failed to enter recipe directory."

    # Ensure Conan profile exists (idempotent)
    conan profile detect --force 2>/dev/null || true

    # Build & register package into local Conan cache
    conan create . \
        --name=onnxruntime \
        --version="${ONNXRUNTIME_VERSION}" \
        --build=missing

    if [ $? -ne 0 ]; then
        popd >/dev/null
        func_1_2_err "Failed to create ONNX Runtime Conan package."
    fi

    popd >/dev/null

    func_1_1_log "   ✔ ONNX Runtime ${ONNXRUNTIME_VERSION} successfully installed into Conan cache." "green"
}

func_3_6_echo_ort_root_for_cmake(){
    # ONNXRUNTIME_ROOT="$(conan cache path onnxruntime/${ONNXRUNTIME_VERSION})"
    ONNXRUNTIME_ROOT=$(conan cache path onnxruntime/${ONNXRUNTIME_VERSION} | sed 's#/e$#/s#')
    # func_1_1_log "   ✔ ONNX Runtime root detected: ${ONNXRUNTIME_ROOT}" "green"
    echo "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
}


# ==============================================================================
# Level 5: Cleanup & Finalization
# ==============================================================================

func_5_1_clean_project(){
    # # (Merged A & B)
    # func_1_1_log "🧹 Cleanup Options:" "blue"
    # echo "   1. Generated models (.onnx)"
    # echo "   2. Virtual Environment (.venv)"
    # echo "   3. Full Clean (All of above)"
    # echo "   4. Cancel"
    # read -p "   Choice [1-4]: " clean_choice
    _clean_build(){
        func_7_1_cpp_core_clean_target

        if [ -d "${LV1_BUILD_DIR}" ]; then
            rm -rf "${LV1_BUILD_DIR}"
            func_1_1_log "✅ build dir Has been deleted." "green"
        # else
        #     func_1_1_log "✅ build dir not exist. Remove Skipped." "green"
        fi

        func_1_1_log "✅ 1st Lv Clean has done." "green"
    }

    _clean_all(){
        _clean_build

        if [ -d "${LV1_MODELS_DOWNLOAD_DIR}" ]; then
            rm -rf "${LV1_MODELS_DOWNLOAD_DIR}"
            func_1_1_log "✅ MODNet Pretrained Models Has been deleted." "green"
        # else
        #     func_1_1_log "✅ MODNet Pretrained Models not exist. Remove Skipped." "green"
        fi

        if [ -d "${LV1_VENV_DIR}" ]; then
            rm -rf "${LV1_VENV_DIR}"
            func_1_1_log "✅ Python Virtual Environment Has been deleted." "green"
        # else
        #     func_1_1_log "✅ Python Virtual Environment not exist. Remove Skipped." "green"
        fi

        if [ -d "${LV4_MODNET_SDK_DIR}" ]; then
            rm -rf "${LV4_MODNET_SDK_DIR}"
            func_1_1_log "✅ MODNet SDK Has been deleted." "green"
        # else
        #     func_1_1_log "✅ MODNet SDK not exist. Remove Skipped." "green"
        fi

        func_1_1_log "✅ 2nd Lv Clean has done." "green"
    }

    local clean_choice="$1"
    # func_1_1_log "clean_choice=${clean_choice}"
    case $clean_choice in
        2)
            _clean_all
            ;;
        *)
            _clean_build
            ;;
    esac
}

func_5_4_finalize() {
    # (From B) - The "magic" reporting at the end
    local exit_code=$?
    local final_time=$(func_1_3_get_current_milliseconds)

    echo "" # New line for visual separation
    if [[ -n "${__START_TIME:-}" ]]; then
        # Total script execution time
        local whole_duration_human
        func_1_6_elapsed_time_calculation "${__START_TIME}" "${final_time}" whole_duration_human
        func_1_1_log "Total Session Time: ${whole_duration_human}." "blue"
    fi
    # Exit with the code of the last executed command
    exit "$exit_code"
}

