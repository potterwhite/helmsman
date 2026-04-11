# Helmsman — AI Agent 工作指南

> **目标读者**：AI 编码 Agent（Claude Code、Cursor 等）
> **碰任何代码之前请先读本文档。**
> **English →** [../../en/1-for-ai/guide.md](../../en/1-for-ai/guide.md)

---

## 1. 每次会话的阅读顺序

1. **本文档** — 了解如何在本仓库工作
2. **[`codebase_map.md`](codebase_map.md)** — 完整代码库结构（代替文件扫描）
3. **[`../2-progress/progress.md`](../2-progress/progress.md)** — 当前阶段与活跃任务
4. **相关参考文档** — 仅在任务需要时读取（API 规范、领域规则等）

---

## 2. 不可违反的规则

### 代码
- 所有源代码和注释必须使用**英文**
- 与人类沟通时使用**中文**
- **不得**擅自结束会话——始终等待用户的下一步指令

### Commit
- **每个步骤一个 commit** — 不要堆积多个改动后一次性提交
- 严格遵循下方 commit 消息格式
- 永远不要提交损坏的代码或失败的测试

### 文档
- 修改任何在 `codebase_map.md` 中列出的文件后，必须在同一个 commit 中更新该文档
- 某个阶段 Block 完成后，在同一 commit 中更新 `../2-progress/progress.md` 中的状态
- 开始新工作时，在 `../2-progress/NEED_TO_DO.md` 中添加日期组条目

### AI 角色定位（重要！）

**AI 是教练，用户是学员**。本项目的核心目标是让用户通过实践掌握：
量化（INT8/INT4）、剪枝（Pruning）、知识蒸馏（Distillation）、微调（Fine-tune）的能力。

因此 AI 必须：
- **不要把所有步骤都替用户执行**——提出步骤建议，让用户自己运行命令
- **等用户反馈结果**——用户执行后，AI 检查输出并给出建议
- **解释原理**——每个关键操作都简要说明"为什么这样做"
- **PKB 记录员**——把推演过程、错误和修复即时记录到 PKB（`1.4.md` 等文件）

当用户说"我来操作"时，AI 切换到教练模式：只给命令和预期输出，不自己执行。

当用户说"**你尽可能袖手旁观，但要对一切成竹于胸**"时，AI 进入**纯观察-教练模式**：
- 完全不主动修改代码或执行命令
- 只在用户提问时回答，或在用户执行后分析输出结果
- 重点工作：记录用户每一步操作的参数和结果，保持对全局状态的掌握
- 这一状态持续到用户明确解除（例如说"帮我做"或"你来"）
- **2026-04-01 起本项目处于此模式**：C++ native 构建与调试由用户主导，AI 陪同观察并在被问到时给出分析

---

## 3. Commit 消息格式

```
<type>: <subject>

<body>

<footer>
```

**类型**（必填）：`feat` · `fix` · `docs` · `refactor` · `perf` · `test` · `build` · `chore`

**主题**（必填）：英文，≤70 字符，现在时
- ✅ `feat(rknn): implement zero-copy inference and INT8 quantization pipeline`
- ✅ `fix: handle non-GPU environments in requirements.txt`
- ❌ `Fixed bugs and updated stuff`

**正文**（推荐）：条目式说明做了什么、为什么这样做

**尾部**（推荐）：`Phase X BlockY.Z complete.`

> **重要**：本项目使用 **release-please** 自动发版。合并到 `main` 的 commit 中，`feat:` 前缀触发 minor 版本升级；`fix:` 触发 patch；`feat!:`（破坏性变更）触发 major。

---

## 4. 如何处理用户请求

### "构建新功能"
1. 提出澄清问题（目标、涉及文件、是否有破坏性变更）
2. 在 `docs/` 中写计划——**此时不写代码**
3. 等待用户批准
4. 分步实施，每步一个 commit

### "有 bug"
1. 复现并理解根本原因
2. 修复
3. 用 `fix:` 前缀 commit

### "重构 / 优化"
1. 在 `docs/architecture/` 中写重构方案
2. 等待用户批准
3. 分步执行

---

## 5. 常见陷阱

| ❌ 错误做法 | ✅ 正确做法 |
|---|---|
| 修改 10 个文件后做一次大 commit | 每完成一个逻辑步骤就 commit |
| 没读 codebase_map 就开始写代码 | 先读 codebase_map |
| 改完代码，忘记更新 codebase_map | 始终在同一 commit 里同步 codebase_map |
| 模糊的 commit 消息"fix bugs" | 具体消息，说明根本原因 |
| 发明新的架构模式 | 遵循 codebase_map 中已有的模式 |
| 直接修改 `third-party/sdk/MODNet.git/` 中的文件 | 在 `third-party/scripts/modnet/` 中修改（会被符号链接注入） |
| 硬编码 Python 版本或 pip URL | `common.sh` 中的 `func_2_0_setup_env()` 是唯一真实来源 |
| 在 C++ 前端将输入归一化到 [-1,1] | 数据范围是 0–255 float32；RKNN 量化通过校准数据内部处理归一化 |
| 认为 C++ 的 letterbox/INTER_LINEAR/0-255 是"Bug"并试图改成与 Python golden 一致 | **这些差异是故意的！** C++ 预处理（letterbox padding、`cv::INTER_LINEAR`、0-255 float32）是针对 ArcFoundry RKNN pipeline 专门调好的配置，与 Python golden（`cv2.INTER_AREA`、`/127.5-1`、无 padding）不同是正常且正确的行为。**永远不要**为了让 native ONNX 对比结果一致而修改这些参数——那会破坏 RKNN 推理 |
| 使用旧版 ONNX Runtime API `GetInputName()` | 使用 `GetInputNameAllocated()`（v0.5.0 之后已更新） |
| 没有 `.env` 就运行 `./helmsman build cpp build rk3588s` | 先从 `.env.example` 创建 `runtime/cpp/.env` |

---

## 6. 关键架构事实

- **CLI 入口是 `helmsman`**（根目录 Bash 脚本）。`scripts/` 下的所有子脚本均通过 source 引入，从不单独执行。永远不要单独运行 `scripts/*.sh`。
- **Python 目标版本是 3.8.10** — 硬编码在 `common.sh`（`PYTHON_Target_VERSION`）中。`envs/requirements.txt` 中的所有 pip 版本号均针对此版本校准。
- **C++ 后端在编译期选定** — CMake 中的 `ENABLE_RKNN_BACKEND` 定义。native（x86）= ONNX Runtime。`rk3588s`/`rv1126bp` 预设 = RKNN。永远不要为此添加运行时分支逻辑。
- **`TensorData` 是流水线契约** — 前端填充 `orig_width/height` 和 `pad_top/bottom/left/right`。后端必须使用这些字段裁剪 letterbox 并恢复原始尺寸。创建 `TensorData` 时不得遗漏任何字段。
- **输入数据范围是 0–255 float32 HWC 格式** — 前端不做归一化到 [-1,1]。RKNN 量化流水线通过校准数据处理归一化。
- **版本号在 `CHANGELOG.md` 中** — CMake 通过 `arc_extract_version_from_changelog()` 读取。永远不要直接在 `CMakeLists.txt` 中修改版本号。
- **两个工作分支**：`main`（仅用于发版，不要直接在这里工作）和 `retrain/modnet`（当前所有工作在此）。
- **MODNet 子模块是临时性的** — `./helmsman cleanall` 会重置它。所有永久改动归属于 `third-party/scripts/modnet/`；符号链接由 `./helmsman prepare` 重新创建。
- **反融合对 RKNN 至关重要** — 不要在模型任何地方引入 `InstanceNormalization`。RKNN 编译器会检测并重构它，导致 CPU 回退和约 40% 的延迟回归。
- **`runtime/cpp/.env` 在 gitignore 中且为必需** — `cpp_build.sh` 若它不存在会立即退出。必须手动从 `.env.example` 创建。

---

## 7⁺. CMake 版本漂移警告（重要！）

### 问题
`conanfile.txt` 中写的是 `cmake/[>=3.18]`（浮动范围），Conan 会从 conancenter 解析到当前最新版（2026-04 = cmake/4.3.0）。

**cmake 4.x 改变了 `--preset` + 工具链注入的行为**：
- cmake 3.28.x：`-DCMAKE_TOOLCHAIN_FILE=...` 作为额外参数追加到 `cmake --preset` 后面，**能生效**
- cmake 4.x：`--preset` 初始化阶段提前，命令行 `-D` 无法在正确时机插入工具链
- 症状：`conan_toolchain.cmake` 未被加载 → `CMAKE_PREFIX_PATH` 未设置 → `find_package(OpenCV)` 失败

### 现象
```
CMake Error at libs/cvkit/CMakeLists.txt:84 (find_package):
Could not find a package configuration file provided by "OpenCV"
```

### 根本原因
`scripts/cpp_build.sh` 用命令行 `-DCMAKE_TOOLCHAIN_FILE=...` 注入工具链，这在 cmake 4.x 里不够早。

### 修复方案（二选一）

**方案 A（推荐）：** 在 `runtime/cpp/conanfile.txt` 的 `[tool_requires]` 中固定 cmake 版本：
```ini
[tool_requires]
cmake/3.28.6
```
这样 Conan 始终使用 3.28.6，不受 conancenter 更新影响，行为可复现。

**方案 B：** 在 `runtime/cpp/CMakePresets.json` 的 `native-release` preset 中加入：
```json
"toolchainFile": "${sourceDir}/build/native-release/conan_toolchain.cmake",
```
这样工具链由 preset 直接加载，不依赖命令行 `-D` 注入。

### 环境一致性建议
- **不要**把 Docker 容器锁定到 Ubuntu 22 — cmake 版本问题与 Ubuntu 版本无关
- **要**在 `conanfile.txt` 中 pin cmake 版本 — 这是 Conan 管理的依赖，应该像其他包一样固定
- 用户本机的 `cmake 3.28.3`（`apt` 安装）与 Conan 下载的 `cmake 4.3.0` 是两个独立的 cmake，不冲突；Helmsman 构建只用 Conan 管理的那个

---

## 7. 开发命令

```bash
# Python 环境配置与模型操作
./helmsman prepare              # 完整配置：pyenv + Python 3.8 + venv + 依赖 + MODNet 子模块
./helmsman convert              # .ckpt → .onnx（交互式：选原始或修改版变体）
./helmsman inference            # 对图片运行 Python ONNX 推理（交互式）
./helmsman golden               # 生成 golden 参考二进制文件（用于 C++ 验证）
./helmsman clean                # 清理构建产物和 build/
./helmsman cleanall             # 彻底清理：+ .venv、模型、MODNet 子模块重置

# C++ 构建命令
./helmsman build cpp build              # 增量构建（native, release, shared）
./helmsman build cpp cb                 # 清理 + 构建 + 安装（native）
./helmsman build cpp build rk3588s      # 交叉编译（RK3588, release）
./helmsman build cpp cb rv1126bp debug  # 全新构建（RV1126BP, debug）
./helmsman build cpp list               # 列出所有可用 CMake 预设
./helmsman build cpp clean rk3588s      # 仅清理 RK3588 构建
./helmsman build cpp test               # 构建并运行测试（native）

# 直接使用 CMake（绕过 helmsman）
cd runtime/cpp
cmake --preset rk3588s-release
cmake --build build/rk3588s-release -j$(nproc)
cmake --install build/rk3588s-release

# 部署到开发板
./tools/deploy_and_test.sh              # 构建 rk3588s → rsync → 远程推理
./tools/deploy_and_benchmark.sh         # 在板端运行扩展基准测试

# 验证 C++ 与 Python golden 文件对比
python3 tools/MODNet/verify_golden_tensor.py
python3 tools/MODNet/reconstruct_from_bin.py

# C++ 二进制用法（安装后）
# runtime/cpp/install/<platform>/release/bin/Helmsman_Matting_Client
Helmsman_Matting_Client <image_path> <model_path> <output_dir> [background_path] [--rvm]
```

---

## 8. PKB（个人知识库）路径 — AI Agent RAG 参考

> 这些路径是用户的 Obsidian 知识库。AI Agent 可以读取这些文件获取更丰富的项目上下文。

| 目录 | 用途 |
|---|---|
| `/home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/2025.18_Project_Helmsman/` | ⭐ **Helmsman 项目专属 PKB**（最重要）|
| `/home/james/syncthing/ObsidianVault/PARA-Vault/3_RESOURCE/03_Tech_Stacks_and_Domains/03.2_Artificial_Intelligence/` | AI 技术通用知识库（RAG 参考） |
| `/home/james/syncthing/ObsidianVault/PARA-Vault/2_AREA/03-Area-Career/Project_ArcFoundry/` | ArcFoundry 项目笔记（相关项目）|

**Helmsman PKB 关键文件**：
```
1_PROJECT/2025.18_Project_Helmsman/
├── 2025.18_Project_Helmsman.md          ← 项目主文件
└── development/
    └── stage-2/
        └── 任务2.4-performance-optimization/
            └── 重新训练-BN代替IN/
                ├── roadmap.md            ← ⭐ 项目路线图（4 个阶段）
                └── Phrase-1_Retrain/
                    ├── 1.4.md            ← ⭐ Block 1.4 执行计划（当前活跃！）
                    ├── Block 1.0_数据集与环境对齐.md
                    ├── ✅Block 1.1 网络结构外科手术.md
                    └── Block 1.2  粗调训练 (Fine-tuning).md
```

**规则**：不得在允许路径之外读取 PKB 的父目录。
