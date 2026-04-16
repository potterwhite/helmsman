# Helmsman — Codebase Map (AI Agent Quick Reference)

> **⚠️ FOR AI AGENTS — READ THIS FIRST**
> This document is the single source of truth for codebase structure.
> **Do NOT do a full repo scan** — read this file instead.
>
> **Maintenance rule:** Any AI agent that modifies a file listed here MUST update
> the relevant section in this document in the same commit/session.
>
> Last updated: 2026-03-30 (initial creation from project scan; added full pipeline + script function tables)
>
> **中文版 →** [../../zh/1-for-ai/codebase_map.md](../../zh/1-for-ai/codebase_map.md)

---

## Legend

| Symbol | Meaning |
|---|---|
| ★ | Active development / primary file |
| 🔒 | Critical invariant — understand before touching |
| 📖 | Documentation / knowledge only |
| 🗄️ | Archived / historical |

---

## Repository Root Layout

```
helmsman.git/
├── helmsman                    ← ★ Unified CLI entry point (THE main Bash script)
├── scripts/                    ← Bash function modules (sourced by helmsman, not standalone)
│   ├── common.sh               ← Logging, env vars, timers (func_1_x, func_2_x)
│   ├── setup.sh                ← Env setup & cleanup (func_3_x, func_5_x)
│   ├── python_ops.sh           ← Python workflows: ckpt→onnx, infer, golden (func_4_x)
│   ├── cpp_build.sh            ← C++ build dispatch (func_6_x – func_8_x)
│   └── menus.sh                ← Interactive TUI menus (func_9_x)
│
├── runtime/cpp/                ← C++ inference engine (main deliverable)
│   ├── CMakeLists.txt          ← Root CMake; extracts version from CHANGELOG.md
│   ├── CMakePresets.json       ← 12 build presets (native/rk3588s/rv1126bp × debug/release × static/shared)
│   ├── .env (gitignored) 🔒    ← SDK root paths; REQUIRED before any cross-compile
│   ├── apps/
│   │   └── matting/client/     ← ★ PRIMARY APP: Helmsman_Matting_Client binary
│   ├── libs/
│   │   ├── cvkit/              ← OpenCV wrappers (load, color-convert, resize, dumpBinary)
│   │   ├── utils/              ← Logger, FileUtils, MathUtils, OtherUtils
│   │   ├── runtime/            ← ONNX Runtime session wrapper
│   │   └── network/            ← TCP socket client/server
│   ├── cmake/
│   │   ├── ArcFunctions.cmake  ← Custom CMake macros (arc_init_*, arc_extract_version_*)
│   │   └── toolchains/         ← Cross-compile toolchains: rk3588s.cmake, rv1126bp.cmake
│   └── third_party/            ← FetchContent CMake scripts for deps
│       ├── librknnrt/          ← Rockchip NPU runtime (downloaded at configure time)
│       └── onnxruntime/        ← ONNX Runtime (downloaded at configure time)
│
├── third-party/
│   ├── sdk/MODNet.git/ 🔒      ← Git submodule: ZHKKKe/MODNet. DO NOT EDIT DIRECTLY.
│   └── scripts/modnet/         ← ★ Custom scripts symlinked into MODNet.git/
│       ├── src/models/modnet.py       ← ★ MODIFIED: Pure-BN architecture (IBNorm→BatchNorm)
│       ├── onnx/export_onnx_modified.py  ← ONNX export for modified model
│       ├── onnx/modnet_onnx_modified.py  ← Modified ONNX-compatible architecture
│       ├── onnx/generate_golden_files.py ← Dumps intermediate tensors for C++ validation
│       └── train_modnet_block1_2.py   ← ★ ACTIVE: Fine-tune training script
│
├── envs/
│   └── requirements.txt 🔒     ← Python deps (torch 2.0.1+cu118, onnx 1.8.1, etc.)
│
├── modnet-models/ (gitignored) ← Downloaded model weights
├── tools/                      ← Deployment & benchmarking shell scripts
├── docs/
│   ├── architecture/           ← ★ AI-first documentation system (this directory)
│   │   ├── 00_INDEX.md         ← Navigation hub
│   │   ├── 1-for-ai/           ← AI agent docs
│   │   ├── 2-progress/         ← Phase status + backlog
│   │   └── 3-highlights/       ← Architecture vision + archived decisions
│   ├── MASTER_PLAN.md          ← Project roadmap (4 phases)
│   ├── blocks/                 ← Per-block execution documentation
│   └── knowledge/              ← Self-study KB (KB-001 through KB-009)
│
├── archived/                   ← Historical/experimental code (do not edit)
├── CHANGELOG.md 🔒             ← Semantic versioning source of truth
├── release-please-config.json  ← Automated release configuration
└── .github/workflows/          ← CI: release-please.yml, license-check.yml
```

---

## `helmsman` — CLI Entry Point

Single Bash script at repo root. Sources all `scripts/*.sh` on startup. Dispatches to:

| Command | Handler | Notes |
|---|---|---|
| `prepare` | `func_3_2_setup_dep_before_build` | Full 7-step env setup |
| `convert` | `func_4_1_ckpt_2_onnx` | Interactive .ckpt → .onnx |
| `inference` | `func_4_2_inference_with_onnx` | Python ONNX inference |
| `golden` | `func_4_5_generate_golden_interactive` | Dump golden binary files |
| `build cpp <...>` | `func_8_1_cpp_build_dispatch` | C++ build orchestration |
| `clean` / `cleanall` | `func_5_1_clean_project` | arg 1 = clean, arg 2 = cleanall |
| `menu` | `func_9_1_main_menu_entry` | Interactive TUI |

---

## `scripts/common.sh`

| Function | Level | Purpose |
|---|---|---|
| `func_1_1_log()` | 1 | Colored console logging (green/yellow/red/blue) |
| `func_1_2_err()` | 1 | Error + exit |
| `func_1_3_get_current_milliseconds()` | 1 | Returns `date +%s%3N` |
| `func_1_4_start_time_count()` | 1 | Start a timer (nameref) |
| `func_1_5_format_duration_ms()` | 1 | Format milliseconds → human readable |
| `func_1_6_elapsed_time_calculation()` | 1 | Calculate elapsed duration |
| `func_1_7_usage()` | 1 | Print help text |
| `func_1_8_activate_py_venv()` | 1 | Activate `.venv/`, exit if missing |
| `func_2_0_setup_env()` | 2 | Set ALL path variables (LV1_*, LV4_*, etc.) + call `func_6_2_cpp_setup_env` |
| `func_2_1_check_env_ready()` | 2 | Verify venv + required MODNet files exist |

**Key variables set by `func_2_0_setup_env()`**:
```
PYTHON_Target_VERSION=3.8.10
LV1_DOC_DIR, LV1_ENVS_DIR, LV1_MEDIA_DIR, LV1_RUNTIME_DIR
LV1_3RD_PARTY_DIR, LV1_TOOLS_DIR, LV1_VENV_DIR, LV1_BUILD_DIR
LV1_MODELS_DOWNLOAD_DIR (→ modnet-models/)
LV4_MODNET_SDK_DIR      (→ third-party/sdk/MODNet.git)
LV4_MODNET_SCRIPTS_DIR  (→ third-party/scripts/modnet)
LV5_PRETRAINED_DIR      (→ third-party/sdk/MODNet.git/pretrained)
DEV_REQUIREMENTS_FILE   (→ envs/requirements.txt)
PYTHON_BIN, PIP_BIN     (→ .venv/bin/python, pip)
```

---

## `scripts/setup.sh`

| Function | Level | Purpose |
|---|---|---|
| `func_3_0_setup_modnet_softlinks()` | 3 | Create symlinks `scripts/modnet/` → `MODNet.git/`; auto-download models from HuggingFace |
| `func_3_1_check_opencv_compatibility()` | 3 | Try `import cv2`; swap to `opencv-python-headless` if fail |
| `func_3_2_setup_dep_before_build()` | 3 | Full 7-step env setup (apt, pyenv, venv, pip, conan, MODNet submodule, symlinks) |
| `func_3_3_rebuild_sdk()` | 3 | Reset MODNet submodule: `reset` = git reset+clean; `nuclear` = full deinit+re-add |
| `func_5_1_clean_project()` | 5 | `clean` (arg 1): C++ builds + `build/`; `cleanall` (arg 2): + models + `.venv` + MODNet reset |
| `func_5_4_finalize()` | 5 | EXIT trap: print total session duration |

**Symlinks created by `func_3_0_setup_modnet_softlinks()`**:
```
third-party/scripts/modnet/onnx/inference_onnx.py         → MODNet.git/onnx/
third-party/scripts/modnet/onnx/generate_golden_files.py  → MODNet.git/onnx/
third-party/scripts/modnet/onnx/export_onnx.py            → MODNet.git/onnx/
third-party/scripts/modnet/onnx/modnet_onnx.py            → MODNet.git/onnx/
third-party/scripts/modnet/onnx/export_onnx_modified.py   → MODNet.git/onnx/
third-party/scripts/modnet/onnx/modnet_onnx_modified.py   → MODNet.git/onnx/
third-party/scripts/modnet/src/models/modnet.py           → MODNet.git/src/models/
third-party/scripts/modnet/train_modnet_mvp.py            → MODNet.git/
third-party/scripts/modnet/train_modnet_block1_2.py       → MODNet.git/
envs/requirements.txt                                     → MODNet.git/onnx/requirements.txt
```

---

## `scripts/python_ops.sh`

| Function | Level | Purpose |
|---|---|---|
| `func_4_1_ckpt_2_onnx()` | 4 | Interactive: select variant (original/modified), select .ckpt → run `export_onnx[_modified].py` |
| `func_4_2_inference_with_onnx()` | 4 | Interactive: select ONNX model, select image → run `inference_onnx.py` |
| `func_4_5_generate_golden_interactive()` | 4 | Interactive: select image + ONNX model → run `generate_golden_files.py` → dump to `build/golden/` |

---

## `scripts/cpp_build.sh`

| Function | Level | Purpose |
|---|---|---|
| `func_6_2_cpp_setup_env()` | 6 | Set all C++ path variables; load `runtime/cpp/.env` file |
| `func_6_1_cpp_check_preset_existence()` | 6 | Verify preset exists in CMakePresets.json |
| `func_6_3_cpp_helper_print()` | 6 | Print C++ build help text |
| `func_7_1_cpp_core_clean_target()` | 7 | `rm -rf` build dir + install dir |
| `func_7_2_cpp_core_configure()` | 7 | `cmake --preset <name>` + optional extra args |
| `func_7_3_cpp_core_build()` | 7 | `cmake --build <dir> -j$(nproc)` |
| `func_7_4_cpp_core_install()` | 7 | `cmake --install <dir>` |
| `func_7_5_cpp_core_test()` | 7 | `ctest --test-dir <dir> --output-on-failure` |
| `func_8_2_cpp_arguments_parsing()` | 8 | Parse: action, platform, build_type, lib_type → construct `CPP_PRESET_NAME` |
| `func_8_3_cpp_validate_platform_env()` | 8 | Check `ARC_<PLATFORM>_SDK_ROOT` env var is set and points to valid dir |
| `func_8_4_cpp_dispatch()` | 8 | Orchestrator: Phase 1 (pre-clean) → Phase 2 (Conan for native) → Phase 3 (execute) |
| `func_8_5_cpp_run_native_conan_install()` | 8 | Run `conan install` in build dir for native platform |
| `func_8_1_cpp_build_dispatch()` | 8 | Top-level entry from `helmsman build` command |

---

## `runtime/cpp/` — C++ Inference Engine

### Root Files

| File | Purpose |
|---|---|
| `CMakeLists.txt` 🔒 | Project root; extracts version via `arc_extract_version_from_changelog()`; includes third_party → libs → apps |
| `CMakePresets.json` | 12 build presets with dependency URLs and SHA hashes |
| `conanfile.txt` | Conan package manifest (used for native builds) |
| `.env` (gitignored) 🔒 | `ARC_RK3588S_SDK_ROOT` and `ARC_RV1126BP_SDK_ROOT`; required before any cross-compile |

### `cmake/ArcFunctions.cmake`

Custom CMake macros:
- `arc_init_global_settings()` — C++17, warnings, PIC
- `arc_init_project_metadata()` — project name, version
- `arc_extract_version_from_changelog(CHANGELOG_FILE VERSION_VAR)` — parses `## [X.Y.Z]` from CHANGELOG.md

### CMake Build Presets

| Preset Name | Platform | Build | Libs | Key Deps |
|---|---|---|---|---|
| `native-release` | x86_64 | Release | Shared | ONNX Runtime 1.16.3 |
| `native-debug` | x86_64 | Debug | Shared | ONNX Runtime 1.16.3 |
| `native-release-static` | x86_64 | Release | Static | ONNX Runtime 1.16.3 |
| `rk3588s-release` | aarch64 | Release | Shared | librknnrt 2.3.2 + ORT 1.16.3 |
| `rk3588s-debug` | aarch64 | Debug | Shared | librknnrt 2.3.2 |
| `rv1126bp-release` | aarch64 | Release | Shared | librknnrt 2.3.2, sherpa-onnx |
| `rv1126bp-release-static` | aarch64 | Release | Static | librknnrt 2.3.2 |

---

## `runtime/cpp/libs/` — Shared Libraries

### `libs/cvkit/` — OpenCV Wrapper
**CMake target**: `Helmsman::Lib::CVKit`

| File | Key Classes/Functions |
|---|---|
| `include/CVKit/base/base.h` | `arcforge::cvkit::Base`: `loadImage()`, `bgrToRgb()`, `ensure3Channel()`, `hwcToNchw()`, `dumpBinary()` |
| `src/base/base.cpp` | Implementation |
| `src/base/impl/base-impl.{h,cpp}` | Pimpl pattern |

### `libs/utils/` — Utilities
**CMake target**: `Helmsman::Lib::Utils`

| File | Key Classes/Functions |
|---|---|
| `include/Utils/logger/logger.h` | `Logger::GetInstance()`, `.Info()`, `.Warning()`, `.Debug()`, `.setLevel()`, `.AddSink()`, `.ClearSinks()` |
| `include/Utils/logger/worker/consolesink.h` | `ConsoleSink` — stdout sink |
| `include/Utils/logger/worker/filesink.h` | `FileSink` — file sink |
| `include/Utils/math/math-utils.h` | `MathUtils::GetInstance()`, `getScaleFactor()` → `ScaleFactor` struct |
| `include/Utils/file/file-utils.h` | `FileUtils::GetInstance()`, `dumpBinary()` |

### `libs/runtime/` — ONNX Runtime Wrapper
**CMake target**: `Helmsman::Lib::Runtime`

| File | Key Classes/Functions |
|---|---|
| `include/Runtime/onnx/onnx.h` | `arcforge::runtime::RuntimeONNX::GetInstance()` |
| `src/onnx/onnx.cpp` | ONNX Runtime session management |
| `src/onnx/impl/impl.{h,cpp}` | Pimpl: create session, run inference, `GetInputNameAllocated()` |

### `libs/network/` — TCP Socket Library
**CMake target**: `Helmsman::Lib::Network`

| File | Key Classes |
|---|---|
| `include/Network/client/client.h` | `Network::Client` — TCP client |
| `include/Network/server/server.h` | `Network::Server` — TCP server |
| `include/Network/base/exception.h` | Network exceptions |

---

## `apps/matting/client/` ★ — Primary Application

**Binary**: `Helmsman_Matting_Client`
**Usage**: `Helmsman_Matting_Client <image> <model> <output_dir> [background]`
**Install path**: `runtime/cpp/install/<platform>/release/bin/`

### Matting Pipeline (C++ Data Flow)

```
[Input: Image Path + RKNN/ONNX Model Path]
        |
        ▼
┌─────────────────┐
│  ImageFrontend  │  → pipeline/frontend/frontend.cpp
│  ::preprocess() │
│                 │  1. cv::imread (BGR)
│                 │  2. BGR → RGB
│                 │  3. Ensure 3-channel
│                 │  4. convertTo CV_32FC3 (keep 0–255 range, NO normalize)
│                 │  5. Letterbox resize to model_W × model_H
│                 │     - scale preserving aspect ratio
│                 │     - cv::copyMakeBorder (black padding)
│                 │  6. Copy HWC float32 memory → tensor_data.data
│                 │  7. Populate TensorData metadata (orig_w/h, pad_*)
└────────┬────────┘
         │ TensorData {data, shape[NHWC], orig_w, orig_h, pad_top/bottom/left/right}
         ▼
┌─────────────────────────────┐
│   InferenceEngine (abstract)│  → pipeline/inference-engine/base/inference-engine.h
│   .load(model_path)         │
│   .infer(TensorData)        │
│                             │
│  RKNN path (ENABLE_RKNN_BACKEND):
│  ├─ InferenceEngineRKNNZeroCP  ← PRIMARY for production
│  │   - rknn_create_mem (NPU-visible buffers)
│  │   - memcpy input into input_mem_->virt_addr
│  │   - rknn_run()
│  └─ InferenceEngineRKNN       ← Non-zero-copy (backup)
│
│  ONNX path (native/x86 default):
│  └─ InferenceEngineONNX       ← ONNX Runtime 1.16.3
└────────┬────────────────────┘
         │ TensorData {data, shape[NCHW], same metadata}
         ▼
┌─────────────────┐
│  MattingBackend │  → pipeline/backend/backend.cpp
│  ::postprocess()│
│                 │  1. NCHW → HWC conversion (loop over C,H,W)
│                 │  2. Clamp to [0,1]
│                 │  3. Crop letterbox padding (using pad_* metadata)
│                 │  4. Resize cropped mask → (orig_w, orig_h) via INTER_LINEAR
│                 │  5. convertTo CV_8UC1, scale ×255 → save cpp_11_result.png
│                 │  6. Optional: alpha composite → cpp_12_composed.jpg
└─────────────────┘
```

**Key invariant**: Frontend data range is **0–255 float32 in HWC**. RKNN driver handles quantization internally via calibration.

### Pipeline Files

| File | Purpose |
|---|---|
| `src/main-client.cpp` | Entry point; sets up logger, signal handler, calls `Pipeline::init()` + `run()` |
| `include/common-define.h` | `kcurrent_module_name = "main-client"` |
| `include/pipeline/pipeline.h` | `Pipeline` singleton: `init()`, `run()` |
| `src/pipeline/pipeline.cpp` 🔒 | Orchestrates: load model → preprocess → infer (×10 bench) → postprocess |
| `include/pipeline/core/data_structure.h` 🔒 | `TensorData` struct (data, shape, orig_w/h, pad_*) |
| `include/pipeline/frontend/frontend.h` | `ImageFrontend`: `preprocess(image_path, model_w, model_h)` |
| `src/pipeline/frontend/frontend.cpp` 🔒 | BGR→RGB→float32→letterbox→HWC tensor (0–255 range) |
| `include/pipeline/inference-engine/base/inference-engine.h` 🔒 | `InferenceEngine` ABC: `load()`, `infer()`, `setOutputBinPath()`, `getInputHeight()`, `getInputWidth()` |
| `include/pipeline/inference-engine/rknn/rknn-zero-copy.h` | `InferenceEngineRKNNZeroCP` |
| `src/pipeline/inference-engine/rknn/rknn-zero-copy.cpp` | Alloc NPU buffers via `rknn_create_mem`, bind, memcpy, `rknn_run`, read output |
| `include/pipeline/inference-engine/rknn/rknn-non-zero-copy.h` | `InferenceEngineRKNN` (backup) |
| `include/pipeline/inference-engine/onnx/onnx.h` | `InferenceEngineONNX` |
| `src/pipeline/inference-engine/onnx/onnx.cpp` | `OrtSession::Run`, tensor I/O via `GetInputNameAllocated()` |
| `include/pipeline/backend/backend.h` | `MattingBackend`: `postprocess(TensorData)` → `cv::Mat` |
| `src/pipeline/backend/backend.cpp` 🔒 | NCHW→HWC, clamp, crop letterbox, resize to original, save PNG/JPG |

### `TensorData` Contract 🔒

```cpp
typedef struct {
    std::vector<float> data;      // Flat float array
    std::vector<int64_t> shape;   // {1,H,W,C} (NHWC) for input; {1,C,H,W} (NCHW) for output
    int orig_width;   // Original image width before letterbox
    int orig_height;  // Original image height before letterbox
    int pad_top;      // Pixels of black padding added to top
    int pad_bottom;   // Pixels of black padding added to bottom
    int pad_left;     // Pixels of black padding added to left
    int pad_right;    // Pixels of black padding added to right
} TensorData;
```

### Debug Binary Dumps (validation against Python golden files)

| File | Stage |
|---|---|
| `cpp_01_loadimage.bin` | After `cv::imread` |
| `cpp_02_bgrToRgb.bin` | After BGR→RGB |
| `cpp_03_ensure3Channel.bin` | After channel check |
| `cpp_04_converted_float.bin` | After uint8→float32 |
| `cpp_05_resized.bin` | After letterbox resize |
| `cpp_06-07_hwc_direct.bin` | Final input tensor (HWC) |
| `cpp_09_backend_input.bin` | Raw model output |
| `cpp_10_clamped.bin` | After [0,1] clamp |
| `cpp_11_result.png` | Final alpha matte (visual) |
| `cpp_12_composed.jpg` | Alpha-composited result (if background provided) |

### C++ Namespaces & Logger Pattern

```cpp
// Logging (use everywhere)
auto& logger = arcforge::embedded::utils::Logger::GetInstance();
logger.Info("message", kcurrent_module_name);
logger.Warning("message", kcurrent_module_name);
logger.Debug("message", kcurrent_module_name);  // only shown in Debug builds

// kcurrent_module_name defined in include/common-define.h per app
constexpr std::string_view kcurrent_module_name = "main-client";
```

---

## `third-party/scripts/modnet/` ★ — MODNet Custom Scripts

**Do NOT edit files in `third-party/sdk/MODNet.git/` directly.** Edit here; symlinks propagate to the submodule.

| File | Purpose |
|---|---|
| `src/models/modnet.py` ★ | **Modified MODNet architecture** — `Conv2dIBNormRelu` uses `nn.BatchNorm2d` only (Pure-BN). No `IBNorm`, no `InstanceNorm2d`. Classes: `MODNet`, `LRBranch`, `HRBranch`, `FusionBranch` |
| `onnx/modnet_onnx_modified.py` ★ | Graph-rewritten model: `IBNorm.forward()` uses `Var(x) = E[x²] − (E[x])²` primitives (anti-fusion prevents RKNN from reconstructing InstanceNorm) |
| `onnx/export_onnx_modified.py` ★ | Export modified model to ONNX (opset 11, dynamic axes); handles `DataParallel` state_dict stripping; CPU/GPU auto-detect |
| `onnx/modnet_onnx.py` | Original ONNX model class (no modification) |
| `onnx/export_onnx.py` | Export original model to ONNX |
| `onnx/inference_onnx.py` | Python ONNX inference on a single image |
| `onnx/generate_golden_files.py` | Dumps intermediate tensor `.bin` files for C++ bit-level validation |
| `train_modnet_block1_2.py` ★ | Full fine-tune training: `P3MDataset` class with dynamic trimap, color jitter, 15 epochs, `StepLR(step=5, gamma=0.1)` |
| `train_modnet_mvp.py` | Minimal smoke test (50 batches) |
| `VERSIONS.md` | Changelog for script modifications |

### Training Config (edit at top of `train_modnet_block1_2.py`)

```python
DATASET_ROOT = "/path/to/P3M-10k/train/blurred_image"
MASK_ROOT    = "/path/to/P3M-10k/train/mask"
VAL_IMAGE_DIR= "/path/to/P3M-10k/validation/P3M-500-P/blurred_image"
VAL_MASK_DIR = "/path/to/P3M-10k/validation/P3M-500-P/mask"
BATCH_SIZE   = 8          # Reduce to 4 if OOM on 24GB VRAM
LR           = 0.01
EPOCHS       = 15
INPUT_SIZE   = 512
```

---

## `tools/` — Deployment & Benchmarking

| File | Purpose |
|---|---|
| `deploy_and_test.sh` | Build `rk3588s` → `rsync` to `evboard` → `ssh` run inference |
| `deploy_and_benchmark.sh` | Build + deploy + multi-pass benchmark |
| `compare_int8_versions.sh` | Compare INT8 quantization variants on board |
| `compare_opt_levels.sh` | Compare RKNN optimization level 1/2/3 |
| `test_384.sh` / `test_384_final.sh` | Test at 384×384 input resolution |
| `test_opt3.sh` | Test with RKNN opt level 3 |
| `MODNet/verify_golden_tensor.py` | Compare C++ `.bin` outputs vs Python golden files |
| `MODNet/reconstruct_from_bin.py` | Reconstruct image from `.bin` binary dump |

---

## `modnet-models/` (gitignored, auto-downloaded)

| File | Description |
|---|---|
| `modnet_photographic_portrait_matting.ckpt` | Official pretrained (IBNorm/InstanceNorm inside) |
| `modnet_photographic_portrait_matting.onnx` | Official ONNX (has `InstanceNormalization` nodes) |
| `modnet_photographic_portrait_matting_in_folded.onnx` | Graph-rewritten (anti-fusion, no InstanceNorm) |
| `modnet_webcam_portrait_matting.ckpt` | Official webcam model checkpoint |

**Current active model** (in MODNet submodule, not here):
- `third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt` ← retrained Pure-BN (Block 1.2 output)

---

## `archived/` — Historical Code 🗄️

| File | Historical Purpose |
|---|---|
| `conan_recipes/onnxruntime/conanfile.py` | Old Conan packaging for ONNX Runtime (replaced by FetchContent in v0.5.0) |
| `helmsman` | Old monolithic CLI script |
| `generate_golden_files-v1-successful.py` | Initial golden file generator |
| `generate_golden_files-v2-instrumentation-successful.py` | Instrumented version |
| `v1.2-golden-generator.cpp` · `v1.3.x-*.cpp` | Early C++ experiments |

---

## Dependency Graph (C++ CMake)

```
Helmsman_Matting_Client
├── Helmsman::Lib::CVKit        (OpenCV wrappers)
│   └── OpenCV
├── Helmsman::Lib::Utils        (Logger, Math, File)
├── Helmsman::Lib::Runtime      (ONNX Runtime wrapper)
│   └── ThirdParty::ONNXRuntime (FetchContent 1.16.3)
├── [ENABLE_RKNN_BACKEND only]
│   └── ThirdParty::LibRKNNRT   (FetchContent 2.3.2, aarch64 only)
└── arc_base_settings           (C++17, warnings, PIC)

Helmsman_ASR_Server
├── Helmsman::Lib::ASREngine    (Sherpa-ONNX recognizer + VAD)
│   └── ThirdParty::SherpaONNX (FetchContent)
├── Helmsman::Lib::Network      (TCP socket)
└── Helmsman::Lib::Utils
```

---

## Key Architectural Patterns

1. **Sources-not-executes**: All `scripts/*.sh` are sourced by `helmsman`, never run standalone. They share environment via exported Bash variables.
2. **Pimpl everywhere in C++**: `libs/cvkit`, `libs/utils`, `libs/runtime` all use Pimpl (`base-impl.h`) to hide implementation details.
3. **Singleton loggers**: `Logger::GetInstance()`, `RuntimeONNX::GetInstance()`, `MathUtils::GetInstance()` — single instance per process.
4. **FetchContent for all C++ deps**: No submodules or vendored code in `runtime/cpp/`. Everything downloaded at CMake configure time with pinned URLs and SHA hashes.
5. **Symlink injection for MODNet**: Custom Python scripts live in `third-party/scripts/modnet/` and are symlinked into the read-only submodule by `func_3_0_setup_modnet_softlinks()`.
6. **Compile-time backend selection**: RKNN vs ONNX is decided by `ENABLE_RKNN_BACKEND` CMake define — no runtime branching.
7. **Anti-fusion for RKNN**: InstanceNorm is replaced by arithmetic primitives `Var(x) = E[x²] − (E[x])²` to prevent the RKNN compiler from reconstructing `InstanceNormalization` and causing CPU fallback.
