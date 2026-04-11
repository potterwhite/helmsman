# Helmsman — 代码库地图（AI Agent 快速参考）

> **⚠️ AI AGENT 必读**
> 本文档是代码库结构的唯一权威来源。
> **不要做全仓库扫描** — 请直接读本文件。
>
> **维护规则：** 任何修改本文件中所列文件的 AI Agent，必须在同一 commit/会话中更新本文档的对应章节。
>
> 最后更新：2026-04-11（Phase-3 C++ 集成：InferenceEngine N→M tensor、RecurrentStateManager、Pipeline RVM 路径）
>
> **English →** [../../en/1-for-ai/codebase_map.md](../../en/1-for-ai/codebase_map.md)

---

## 图例

| 符号 | 含义 |
|---|---|
| ★ | 活跃开发 / 主要文件 |
| 🔒 | 关键不变量——修改前必须理解 |
| 📖 | 文档 / 仅作知识参考 |
| 🗄️ | 已归档 / 历史文件 |

---

## 仓库根目录结构

```
helmsman.git/
├── helmsman                    ← ★ 统一 CLI 入口（主 Bash 脚本）
├── scripts/                    ← Bash 功能模块（由 helmsman source，不单独执行）
│   ├── common.sh               ← 日志、环境变量、计时器（func_1_x, func_2_x）
│   ├── setup.sh                ← 环境配置与清理（func_3_x, func_5_x）
│   ├── python_ops.sh           ← Python 工作流：ckpt→onnx、推理、golden（func_4_x）
│   ├── cpp_build.sh            ← C++ 构建调度（func_6_x – func_8_x）
│   └── menus.sh                ← 交互式 TUI 菜单（func_9_x）
│
├── runtime/cpp/                ← C++ 推理引擎（主交付物）
│   ├── CMakeLists.txt          ← 根 CMake；从 CHANGELOG.md 提取版本号
│   ├── CMakePresets.json       ← 12 个构建预设（native/rk3588s/rv1126bp × debug/release × static/shared）
│   ├── .env (gitignored) 🔒    ← SDK 根路径；交叉编译前必须存在
│   ├── apps/
│   │   ├── matting/client/     ← ★ 主程序：Helmsman_Matting_Client 二进制
│   │   └── asr/                ← 次程序：ASR 服务端/客户端（Sherpa-ONNX）
│   ├── libs/
│   │   ├── cvkit/              ← OpenCV 封装（加载、颜色转换、缩放、二进制转储）
│   │   ├── utils/              ← Logger、FileUtils、MathUtils、OtherUtils
│   │   ├── runtime/            ← ONNX Runtime 会话封装
│   │   ├── network/            ← TCP socket 客户端/服务端
│   │   └── asr_engine/         ← Sherpa-ONNX 识别器 + VAD
│   ├── cmake/
│   │   ├── ArcFunctions.cmake  ← 自定义 CMake 宏（arc_init_*、arc_extract_version_*）
│   │   └── toolchains/         ← 交叉编译工具链：rk3588s.cmake、rv1126bp.cmake
│   └── third_party/            ← 各依赖的 FetchContent CMake 脚本
│       ├── librknnrt/          ← Rockchip NPU 运行时（配置时下载）
│       ├── onnxruntime/        ← ONNX Runtime（配置时下载）
│       └── sherpa-onnx/        ← ASR 运行时（配置时下载）
│
├── third-party/
│   ├── sdk/MODNet.git/ 🔒      ← Git 子模块：ZHKKKe/MODNet。禁止直接修改。
│   └── scripts/modnet/         ← ★ 自定义脚本，通过符号链接注入 MODNet.git/
│       ├── src/models/modnet.py       ← ★ 已修改：纯 BN 架构（IBNorm→BatchNorm）
│       ├── onnx/export_onnx_modified.py  ← 修改版模型 ONNX 导出脚本
│       ├── onnx/modnet_onnx_modified.py  ← 修改版 ONNX 兼容架构
│       ├── onnx/generate_golden_files.py ← 转储中间张量供 C++ 验证
│       └── train_modnet_block1_2.py   ← ★ 活跃：微调训练脚本
│
├── envs/
│   ├── requirements.txt 🔒     ← Python 依赖——CPU 变体（默认；torch 2.0.1 CPU，ONNX 导出 / golden 生成用）
│   │                              注：onnx==1.14.1（从 1.8.1 升级，torch 2.0.1 需要 ≥1.13）；onnxruntime==1.15.1
│   └── requirements-gpu.txt    ← Python 依赖——GPU 变体（torch 2.0.1+cu118，RTX 3090 训练用；需 PyTorch WHL 源）
│
├── modnet-models/ (gitignored) ← 下载的模型权重
├── tools/                      ← 部署与基准测试 shell 脚本
├── docs/
│   ├── zh/                     ← ★ AI-first 文档系统（中文，权威版本）
│   │   ├── 00_INDEX.md         ← 导航中枢
│   │   ├── 1-for-ai/           ← AI Agent 专用文档
│   │   ├── 2-progress/         ← 阶段状态 + 任务待办
│   │   └── 3-highlights/       ← 架构愿景 + 设计决策
│   └── en/                     ← zh/ 的英文镜像翻译
│
├── archived/                   ← 历史/实验性代码（不要修改）
├── CHANGELOG.md 🔒             ← 语义版本号唯一来源
├── release-please-config.json  ← 自动化发版配置
└── .github/workflows/          ← CI：release-please.yml、license-check.yml
```

---

## `helmsman` — CLI 入口

单一 Bash 脚本，位于仓库根目录。启动时 source 所有 `scripts/*.sh`。调度到：

| 命令 | 处理函数 | 备注 |
|---|---|---|
| `prepare` | `func_3_2_setup_dep_before_build` | 完整 7 步环境配置 |
| `convert` | `func_4_1_ckpt_2_onnx` | 交互式 .ckpt → .onnx |
| `inference` | `func_4_2_inference_with_onnx` | Python ONNX 推理 |
| `golden` | `func_4_5_generate_golden_interactive` | 转储 golden 二进制文件 |
| `build cpp <...>` | `func_8_1_cpp_build_dispatch` | C++ 构建编排 |
| `clean` / `cleanall` | `func_5_1_clean_project` | 参数 1 = clean，参数 2 = cleanall |
| `menu` | `func_9_1_main_menu_entry` | 交互式 TUI |

---

## `scripts/common.sh`

| 函数 | 层级 | 用途 |
|---|---|---|
| `func_1_1_log()` | 1 | 彩色控制台日志（绿/黄/红/蓝） |
| `func_1_2_err()` | 1 | 报错 + 退出 |
| `func_1_3_get_current_milliseconds()` | 1 | 返回 `date +%s%3N` |
| `func_1_4_start_time_count()` | 1 | 启动计时器（nameref） |
| `func_1_5_format_duration_ms()` | 1 | 毫秒格式化为人类可读时长 |
| `func_1_6_elapsed_time_calculation()` | 1 | 计算已用时长 |
| `func_1_7_usage()` | 1 | 打印帮助文字 |
| `func_1_8_activate_py_venv()` | 1 | 激活 `.venv/`，若不存在则退出 |
| `func_2_0_setup_env()` | 2 | 设置所有路径变量（LV1_*、LV4_* 等）+ 调用 `func_6_2_cpp_setup_env` |
| `func_2_1_check_env_ready()` | 2 | 验证 venv + 必需的 MODNet 文件存在 |

**`func_2_0_setup_env()` 设置的关键变量**：
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

| 函数 | 层级 | 用途 |
|---|---|---|
| `func_3_0_setup_modnet_softlinks()` | 3 | 创建符号链接 `scripts/modnet/` → `MODNet.git/`；从 HuggingFace 自动下载模型 |
| `func_3_1_check_opencv_compatibility()` | 3 | 尝试 `import cv2`；失败则换用 `opencv-python-headless` |
| `func_3_2_setup_dep_before_build()` | 3 | 完整 7 步环境配置（apt、pyenv、venv、pip、conan、MODNet 子模块、符号链接） |
| `func_3_3_rebuild_sdk()` | 3 | 重置 MODNet 子模块：`reset` = git reset+clean；`nuclear` = 完整 deinit+重新添加 |
| `func_5_1_clean_project()` | 5 | `clean`（参数 1）：C++ 构建 + `build/`；`cleanall`（参数 2）：+ 模型 + `.venv` + MODNet 重置 |
| `func_5_4_finalize()` | 5 | EXIT trap：打印会话总用时 |

**`func_3_0_setup_modnet_softlinks()` 创建的符号链接**：
```
third-party/scripts/modnet/onnx/inference_onnx.py         → MODNet.git/onnx/
third-party/scripts/modnet/onnx/generate_golden_files.py  → MODNet.git/onnx/
third-party/scripts/modnet/onnx/export_onnx.py            → MODNet.git/onnx/
third-party/scripts/modnet/onnx/modnet_onnx.py            → MODNet.git/onnx/
third-party/scripts/modnet/onnx/export_onnx_modified.py   → MODNet.git/onnx/
third-party/scripts/modnet/onnx/modnet_onnx_modified.py   → MODNet.git/onnx/
third-party/scripts/modnet/onnx/export_onnx_pureBN.py     → MODNet.git/onnx/
third-party/scripts/modnet/src/models/modnet.py           → MODNet.git/src/models/
third-party/scripts/modnet/train_modnet_mvp.py            → MODNet.git/
third-party/scripts/modnet/train_modnet_block1_2.py       → MODNet.git/
envs/requirements.txt                                     → MODNet.git/onnx/requirements.txt
```

---

## `scripts/python_ops.sh`

| 函数 | 层级 | 用途 |
|---|---|---|
| `func_4_1_ckpt_2_onnx()` | 4 | 交互式：选择变体（原始/修改版），选择 .ckpt → 运行 `export_onnx[_modified].py` |
| `func_4_2_inference_with_onnx()` | 4 | 交互式：选择 ONNX 模型，选择图片 → 运行 `inference_onnx.py` |
| `func_4_5_generate_golden_interactive()` | 4 | 交互式：选择图片 + ONNX 模型 → 运行 `generate_golden_files.py` → 转储到 `build/golden/` |

---

## `scripts/cpp_build.sh`

| 函数 | 层级 | 用途 |
|---|---|---|
| `func_6_2_cpp_setup_env()` | 6 | 设置所有 C++ 路径变量；加载 `runtime/cpp/.env` 文件 |
| `func_6_1_cpp_check_preset_existence()` | 6 | 验证预设是否存在于 CMakePresets.json |
| `func_6_3_cpp_helper_print()` | 6 | 打印 C++ 构建帮助文字 |
| `func_7_1_cpp_core_clean_target()` | 7 | `rm -rf` 构建目录 + 安装目录 |
| `func_7_2_cpp_core_configure()` | 7 | `cmake --preset <name>` + 可选额外参数 |
| `func_7_3_cpp_core_build()` | 7 | `cmake --build <dir> -j$(nproc)` |
| `func_7_4_cpp_core_install()` | 7 | `cmake --install <dir>` |
| `func_7_5_cpp_core_test()` | 7 | `ctest --test-dir <dir> --output-on-failure` |
| `func_8_2_cpp_arguments_parsing()` | 8 | 解析：action、platform、build_type、lib_type → 构造 `CPP_PRESET_NAME` |
| `func_8_3_cpp_validate_platform_env()` | 8 | 检查 `ARC_<PLATFORM>_SDK_ROOT` 环境变量是否设置且指向有效目录 |
| `func_8_4_cpp_dispatch()` | 8 | 编排：阶段1（预清理）→ 阶段2（native 的 Conan）→ 阶段3（执行） |
| `func_8_5_cpp_run_native_conan_install()` | 8 | 在构建目录中为 native 平台运行 `conan install` |
| `func_8_1_cpp_build_dispatch()` | 8 | `helmsman build` 命令的顶层入口 |

---

## `runtime/cpp/` — C++ 推理引擎

### 根文件

| 文件 | 用途 |
|---|---|
| `CMakeLists.txt` 🔒 | 项目根；通过 `arc_extract_version_from_changelog()` 提取版本；包含 third_party → libs → apps |
| `CMakePresets.json` | 12 个构建预设，含依赖 URL 和 SHA 哈希 |
| `conanfile.txt` | Conan 包清单（用于 native 构建） |
| `.env` (gitignored) 🔒 | `ARC_RK3588S_SDK_ROOT` 和 `ARC_RV1126BP_SDK_ROOT`；交叉编译前必须存在 |

### `cmake/ArcFunctions.cmake`

自定义 CMake 宏：
- `arc_init_global_settings()` — C++17、警告、PIC
- `arc_init_project_metadata()` — 项目名、版本
- `arc_extract_version_from_changelog(CHANGELOG_FILE VERSION_VAR)` — 从 CHANGELOG.md 解析 `## [X.Y.Z]`

### CMake 构建预设

| 预设名称 | 平台 | 构建类型 | 库类型 | 关键依赖 |
|---|---|---|---|---|
| `native-release` | x86_64 | Release | 动态库 | ONNX Runtime 1.16.3 |
| `native-debug` | x86_64 | Debug | 动态库 | ONNX Runtime 1.16.3 |
| `native-release-static` | x86_64 | Release | 静态库 | ONNX Runtime 1.16.3 |
| `rk3588s-release` | aarch64 | Release | 动态库 | librknnrt 2.3.2 + ORT 1.16.3 |
| `rk3588s-debug` | aarch64 | Debug | 动态库 | librknnrt 2.3.2 |
| `rv1126bp-release` | aarch64 | Release | 动态库 | librknnrt 2.3.2, sherpa-onnx |
| `rv1126bp-release-static` | aarch64 | Release | 静态库 | librknnrt 2.3.2 |

---

## `runtime/cpp/libs/` — 共享库

### `libs/cvkit/` — OpenCV 封装
**CMake 目标**：`Helmsman::Lib::CVKit`

| 文件 | 关键类/函数 |
|---|---|
| `include/CVKit/base/base.h` | `arcforge::cvkit::Base`: `loadImage()`, `bgrToRgb()`, `ensure3Channel()`, `hwcToNchw()`, `dumpBinary()` |
| `src/base/base.cpp` | 实现 |
| `src/base/impl/base-impl.{h,cpp}` | Pimpl 模式 |

### `libs/utils/` — 工具类
**CMake 目标**：`Helmsman::Lib::Utils`

| 文件 | 关键类/函数 |
|---|---|
| `include/Utils/logger/logger.h` | `Logger::GetInstance()`, `.Info()`, `.Warning()`, `.Debug()`, `.setLevel()`, `.AddSink()`, `.ClearSinks()` |
| `include/Utils/logger/worker/consolesink.h` | `ConsoleSink` — 标准输出 sink |
| `include/Utils/logger/worker/filesink.h` | `FileSink` — 文件 sink |
| `include/Utils/math/math-utils.h` | `MathUtils::GetInstance()`, `getScaleFactor()` → `ScaleFactor` 结构体 |
| `include/Utils/file/file-utils.h` | `FileUtils::GetInstance()`, `dumpBinary()` |

### `libs/runtime/` — ONNX Runtime 封装
**CMake 目标**：`Helmsman::Lib::Runtime`

| 文件 | 关键类/函数 |
|---|---|
| `include/Runtime/onnx/onnx.h` | `arcforge::runtime::RuntimeONNX::GetInstance()` |
| `src/onnx/onnx.cpp` | ONNX Runtime 会话管理 |
| `src/onnx/impl/impl.{h,cpp}` | Pimpl：创建会话、运行推理、`GetInputNameAllocated()` |

### `libs/network/` — TCP Socket 库
**CMake 目标**：`Helmsman::Lib::Network`

| 文件 | 关键类 |
|---|---|
| `include/Network/client/client.h` | `Network::Client` — TCP 客户端 |
| `include/Network/server/server.h` | `Network::Server` — TCP 服务端 |
| `include/Network/base/exception.h` | 网络异常 |

### `libs/asr_engine/` — 语音识别引擎
**CMake 目标**：`Helmsman::Lib::ASREngine`

| 文件 | 关键类 |
|---|---|
| `include/ASREngine/recognizer/recognizer.h` | `ASREngine::Recognizer`（Sherpa-ONNX） |
| `include/ASREngine/recognizer/recognizer-config.h` | `RecognizerConfig` 结构体 |
| `include/ASREngine/vad/vad.h` | `ASREngine::VAD` — 语音活动检测 |
| `include/ASREngine/wav-reader/wav-reader.h` | WAV 文件读取器 |

---

## `apps/matting/client/` ★ — 主应用程序

**二进制**：`Helmsman_Matting_Client`
**用法**：`Helmsman_Matting_Client <image> <model> <output_dir> [background] [--rvm]`
**安装路径**：`runtime/cpp/install/<platform>/release/bin/`

**CLI 标志**：
- `--rvm` — 使用 RVM 模式（递归状态 + 5 帧循环）
- `--modnet` — 使用 MODNet 模式（默认，单帧 + 10× benchmark）

### 抠图流水线（C++ 数据流）

```
[输入：图片路径 + RKNN/ONNX 模型路径]
        |
        ▼
┌─────────────────┐
│  ImageFrontend  │  → pipeline/frontend/frontend.cpp
│  ::preprocess() │
│                 │  1. cv::imread (BGR)
│                 │  2. BGR → RGB
│                 │  3. 确保 3 通道
│                 │  4. convertTo CV_32FC3（保持 0–255 范围，不归一化）
│                 │  5. Letterbox 缩放至 model_W × model_H
│                 │     - 保持宽高比缩放
│                 │     - cv::copyMakeBorder（黑色填充）
│                 │  6. 将 HWC float32 内存复制到 tensor_data.data
│                 │  7. 填充 TensorData 元数据（orig_w/h, pad_*）
└────────┬────────┘
         │ vector<TensorData>:
         │   MODNet: [src]
         │   RVM:    [src, r1i, r2i, r3i, r4i] (由 RecurrentStateManager 注入)
         ▼
┌─────────────────────────────┐
│   InferenceEngine（抽象）   │  → pipeline/inference-engine/base/inference-engine.h
│   .load(model_path)         │
│   .infer(vector<in>,        │  ★ N-input / M-output 泛化接口
│          vector<out>)       │    MODNet: 1→1,  RVM: 5→6
│                             │
│  RKNN 路径（ENABLE_RKNN_BACKEND）:
│  ├─ InferenceEngineRKNNZeroCP  ← 生产环境主选
│  │   - 多组 rknn_create_mem（N 输入 + M 输出缓冲区）
│  │   - 遍历 inputs 逐个 memcpy 到 input_mems_[i]->virt_addr
│  │   - rknn_run()
│  └─ InferenceEngineRKNN       ← 非零拷贝（备选）
│
│  ONNX 路径（native/x86 默认）:
│  └─ InferenceEngineONNX       ← ONNX Runtime 1.16.3
└────────┬────────────────────┘
         │ vector<TensorData>:
         │   MODNet: [pha]
         │   RVM:    [fgr, pha, r1o, r2o, r3o, r4o]
         ▼
┌─────────────────────────────┐
│  RecurrentStateManager      │  → pipeline/core/recurrent-state-manager.h
│  (仅 RVM 路径)               │
│  .update(outputs)           │  保存 r1o~r4o → 下一帧的 r1i~r4i
│  .inject(inputs)            │  将 r1i~r4i 追加到 inputs
└────────┬────────────────────┘
         │ vector<TensorData> outputs
         ▼
┌─────────────────┐
│  MattingBackend │  → pipeline/backend/backend.cpp
│  ::postprocess()│
│   (vector<TD>)  │  ★ 按名称选取 pha tensor
│                 │  1. NCHW → HWC 转换（遍历 C,H,W）
│                 │  2. 截断到 [0,1]
│                 │  3. 裁剪 letterbox 填充（使用 pad_* 元数据）
│                 │  4. 将裁剪后遮罩缩放到 (orig_w, orig_h)，INTER_LINEAR
│                 │  5. convertTo CV_8UC1，×255 缩放 → 保存 cpp_11_result.png
│                 │  6. 可选：透明度合成 → cpp_12_composed.jpg
└─────────────────┘
```

**关键不变量**：前端数据范围为 **0–255 float32 HWC 格式**。RKNN 驱动通过校准数据在内部处理量化。

### 流水线文件

| 文件 | 用途 |
|---|---|
| `src/main-client.cpp` | 入口点；配置 logger、信号处理，调用 `Pipeline::init()` + `run()` |
| `include/common-define.h` | `kcurrent_module_name = "main-client"` |
| `include/pipeline/pipeline.h` | `Pipeline` 单例：`init()`, `run()`, `runMODNet()`, `runRVM()` + `ModelType` 枚举 |
| `src/pipeline/pipeline.cpp` 🔒 | 编排：MODNet 路径（1→1 + 10× 基准）/ RVM 路径（5→6 + 递归状态 + 5 帧循环）|
| `include/pipeline/core/data_structure.h` 🔒 | `TensorData` 结构体（name, data, shape, orig_w/h, pad_*） |
| `include/pipeline/core/recurrent-state-manager.h` ★ | `RecurrentStateManager`：RVM 递归状态持久化（init/reset/inject/update） |
| `src/pipeline/core/recurrent-state-manager.cpp` ★ | RecurrentStateManager 实现 |
| `include/pipeline/frontend/frontend.h` | `ImageFrontend`: `preprocess(image_path, model_w, model_h)` |
| `src/pipeline/frontend/frontend.cpp` 🔒 | BGR→RGB→float32→letterbox→HWC 张量（0–255 范围） |
| `include/pipeline/inference-engine/base/inference-engine.h` 🔒 | `InferenceEngine` 抽象基类：`load()`, `infer(vector<in>, vector<out>)` — N→M 泛化接口 |
| `include/pipeline/inference-engine/rknn/rknn-zero-copy.h` | `InferenceEngineRKNNZeroCP` — 多组 zero-copy buffer |
| `src/pipeline/inference-engine/rknn/rknn-zero-copy.cpp` | N 输入 × M 输出 zero-copy：遍历 input_mems_/output_mems_，自适应 INT8/FP16/FP32 |
| `include/pipeline/inference-engine/rknn/rknn-non-zero-copy.h` | `InferenceEngineRKNN`（备选）|
| `src/pipeline/inference-engine/rknn/rknn-non-zero-copy.cpp` | N→M non-zero-copy：rknn_inputs_set 多组 + rknn_outputs_get 多组 |
| `include/pipeline/inference-engine/onnx/onnx.h` | `InferenceEngineONNX` |
| `src/pipeline/inference-engine/onnx/onnx.cpp` | N→M ORT Run：inputs[0] NHWC→NCHW+归一化，inputs[1..N] 直通 |
| `include/pipeline/backend/backend.h` | `MattingBackend`: `postprocess(vector<TensorData>)` → `cv::Mat` — 按名选 pha |
| `src/pipeline/backend/backend.cpp` 🔒 | NCHW→HWC，截断，裁剪 letterbox，缩放至原始尺寸，保存 PNG/JPG |

### `TensorData` 契约 🔒

```cpp
typedef struct {
    std::string            name;  // tensor 名称 ("src", "r1i", "pha", "r1o" 等)
    std::vector<float>     data;  // 展平的 float 数组
    std::vector<int64_t>   shape; // {1,H,W,C}（NHWC）用于输入；{1,C,H,W}（NCHW）用于输出
    int orig_width;   // letterbox 前的原始图像宽度
    int orig_height;  // letterbox 前的原始图像高度
    int pad_top;      // 顶部黑色填充像素数
    int pad_bottom;   // 底部黑色填充像素数
    int pad_left;     // 左侧黑色填充像素数
    int pad_right;    // 右侧黑色填充像素数
} TensorData;
```

### 调试二进制转储（与 Python golden 文件对比验证）

| 文件 | 阶段 |
|---|---|
| `cpp_01_loadimage.bin` | `cv::imread` 之后 |
| `cpp_02_bgrToRgb.bin` | BGR→RGB 之后 |
| `cpp_03_ensure3Channel.bin` | 通道检查之后 |
| `cpp_04_converted_float.bin` | uint8→float32 之后 |
| `cpp_05_resized.bin` | letterbox 缩放之后 |
| `cpp_06-07_hwc_direct.bin` | 最终输入张量（HWC） |
| `cpp_09_backend_input.bin` | 模型原始输出 |
| `cpp_10_clamped.bin` | [0,1] 截断之后 |
| `cpp_11_result.png` | 最终透明遮罩（可视化） |
| `cpp_12_composed.jpg` | 透明度合成结果（提供背景图时生成） |

### C++ 命名空间与 Logger 模式

```cpp
// 日志（全局使用）
auto& logger = arcforge::embedded::utils::Logger::GetInstance();
logger.Info("message", kcurrent_module_name);
logger.Warning("message", kcurrent_module_name);
logger.Debug("message", kcurrent_module_name);  // 仅在 Debug 构建中显示

// kcurrent_module_name 在每个 app 的 include/common-define.h 中定义
constexpr std::string_view kcurrent_module_name = "main-client";
```

---

## `third-party/scripts/modnet/` ★ — MODNet 自定义脚本

**禁止直接修改 `third-party/sdk/MODNet.git/` 中的文件。** 在此目录修改；符号链接会自动传播到子模块。

| 文件 | 用途 |
|---|---|
| `src/models/modnet.py` ★ | **修改版 MODNet 架构** — `Conv2dIBNormRelu` 仅使用 `nn.BatchNorm2d`（纯 BN），无 `IBNorm`，无 `InstanceNorm2d`。类：`MODNet`, `LRBranch`, `HRBranch`, `FusionBranch` |
| `onnx/modnet_onnx_modified.py` ★ | 图改写模型：`IBNorm.forward()` 使用 `Var(x) = E[x²] − (E[x])²` 原语（反融合，防止 RKNN 重构 InstanceNorm） |
| `onnx/export_onnx_modified.py` ★ | 将修改版模型导出为 ONNX（opset 11，动态轴）；处理 `DataParallel` state_dict 剥离；自动检测 CPU/GPU |
| `onnx/export_onnx_pureBN.py` ★ | **Pure-BN ONNX 导出脚本** — 专为 `modnet_bn_best.ckpt` 设计；内含 sys.path 手术防止本地 `onnx/` 遮蔽 pip 包；opset 11，动态轴 |
| `onnx/modnet_onnx.py` | 原始 ONNX 模型类（未修改） |
| `onnx/export_onnx.py` | 将原始模型导出为 ONNX |
| `onnx/inference_onnx.py` | 对单张图片做 Python ONNX 推理 |
| `onnx/generate_golden_files.py` | 转储中间张量 `.bin` 文件供 C++ 位级验证 |
| `train_modnet_block1_2.py` ★ | 完整微调训练：`P3MDataset` 类含动态 trimap、色彩扰动、15 轮、`StepLR(step=5, gamma=0.1)` |
| `train_modnet_mvp.py` | 最小冒烟测试（50 个 batch） |
| `VERSIONS.md` | 脚本修改变更日志 |

### 训练配置（在 `train_modnet_block1_2.py` 顶部修改）

```python
DATASET_ROOT = "/path/to/P3M-10k/train/blurred_image"
MASK_ROOT    = "/path/to/P3M-10k/train/mask"
VAL_IMAGE_DIR= "/path/to/P3M-10k/validation/P3M-500-P/blurred_image"
VAL_MASK_DIR = "/path/to/P3M-10k/validation/P3M-500-P/mask"
BATCH_SIZE   = 8          # 24GB VRAM OOM 时改为 4
LR           = 0.01
EPOCHS       = 15
INPUT_SIZE   = 512
```

---

## `tools/` — 部署与基准测试

| 文件 | 用途 |
|---|---|
| `deploy_and_test.sh` | 构建 `rk3588s` → `rsync` 到 `evboard` → `ssh` 远程推理 |
| `deploy_and_benchmark.sh` | 构建 + 部署 + 多次基准测试 |
| `compare_int8_versions.sh` | 在板端对比不同 INT8 量化方案 |
| `compare_opt_levels.sh` | 对比 RKNN 优化级别 1/2/3 |
| `test_384.sh` / `test_384_final.sh` | 测试 384×384 输入分辨率 |
| `test_opt3.sh` | 使用 RKNN opt level 3 测试 |
| `MODNet/verify_golden_tensor.py` | 对比 C++ `.bin` 输出与 Python golden 文件 |
| `MODNet/reconstruct_from_bin.py` | 从 `.bin` 二进制转储重建图像 |

---

## `modnet-models/`（gitignored，自动下载）

| 文件 | 描述 |
|---|---|
| `modnet_photographic_portrait_matting.ckpt` | 官方预训练（含 IBNorm/InstanceNorm） |
| `modnet_photographic_portrait_matting.onnx` | 官方 ONNX（含 `InstanceNormalization` 节点） |
| `modnet_photographic_portrait_matting_in_folded.onnx` | 图改写版（反融合，无 InstanceNorm） |
| `modnet_webcam_portrait_matting.ckpt` | 官方摄像头模型 checkpoint |

**当前活跃模型**（在 MODNet 子模块中，不在此处）：
- `third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt` ← 重训纯 BN 模型（Block 1.2 输出）

---

## `archived/` — 历史代码 🗄️

| 文件 | 历史用途 |
|---|---|
| `conan_recipes/onnxruntime/conanfile.py` | 旧版 Conan 打包 ONNX Runtime（v0.5.0 已换用 FetchContent） |
| `helmsman` | 旧版单文件 CLI 脚本 |
| `generate_golden_files-v1-successful.py` | 初版 golden 文件生成器 |
| `generate_golden_files-v2-instrumentation-successful.py` | 带插桩的版本 |
| `v1.2-golden-generator.cpp` · `v1.3.x-*.cpp` | 早期 C++ 实验 |

---

## 依赖图（C++ CMake）

```
Helmsman_Matting_Client
├── Helmsman::Lib::CVKit        （OpenCV 封装）
│   └── OpenCV
├── Helmsman::Lib::Utils        （Logger, Math, File）
├── Helmsman::Lib::Runtime      （ONNX Runtime 封装）
│   └── ThirdParty::ONNXRuntime （FetchContent 1.16.3）
├── [仅 ENABLE_RKNN_BACKEND]
│   └── ThirdParty::LibRKNNRT   （FetchContent 2.3.2, aarch64 only）
└── arc_base_settings           （C++17, 警告, PIC）

Helmsman_ASR_Server
├── Helmsman::Lib::ASREngine    （Sherpa-ONNX 识别器 + VAD）
│   └── ThirdParty::SherpaONNX （FetchContent）
├── Helmsman::Lib::Network      （TCP socket）
└── Helmsman::Lib::Utils
```

---

## 关键架构模式

1. **source 不执行**：所有 `scripts/*.sh` 由 `helmsman` source 引入，从不单独运行。通过导出的 Bash 变量共享环境。
2. **C++ 全面 Pimpl**：`libs/cvkit`、`libs/utils`、`libs/runtime` 均使用 Pimpl（`base-impl.h`）隐藏实现细节。
3. **单例 Logger**：`Logger::GetInstance()`、`RuntimeONNX::GetInstance()`、`MathUtils::GetInstance()` — 每个进程一个实例。
4. **所有 C++ 依赖使用 FetchContent**：`runtime/cpp/` 中无子模块或源码内置。所有内容在 CMake 配置时通过固定 URL 和 SHA 哈希下载。
5. **MODNet 符号链接注入**：自定义 Python 脚本位于 `third-party/scripts/modnet/`，由 `func_3_0_setup_modnet_softlinks()` 符号链接到只读子模块中。
6. **编译期后端选择**：RKNN 与 ONNX 由 `ENABLE_RKNN_BACKEND` CMake 定义决定——无运行时分支。
7. **RKNN 反融合**：将 InstanceNorm 替换为算术原语 `Var(x) = E[x²] − (E[x])²`，防止 RKNN 编译器重构 `InstanceNormalization` 导致 CPU 回退。
