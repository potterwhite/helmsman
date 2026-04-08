# Helmsman — 架构愿景

> **用途**：解释项目为什么这样构建。
> 战略定位、设计哲学，以及关键决策背后的思考。
> **English →** [../../en/3-highlights/architecture_vision.md](../../en/3-highlights/architecture_vision.md)

---

## 我们要解决的问题

**人像抠图**（背景消除，精确到发丝级透明度）计算量极大。
现有方案均存在问题：
- 云端方案（高延迟 + 隐私顾虑）
- 依赖桌面 GPU（无法用于嵌入式部署）
- 嵌入式硬件上速度过慢（>500ms 导致实时性不可行）

**Helmsman** 的解法：将训练好的深度学习模型部署在 **Rockchip RK3588 NPU**（6 TOPS）上，目标达到 **30 FPS @ 1080P**——快到足以支撑实时视频会议、视频制作和边缘 AI 应用。

---

## 设计哲学

### 1. NPU 优先，拒绝 CPU 回退

每项架构决策都经过以下问题的审视：
> *"这会导致 RKNN 编译器将计算回退到 CPU 执行吗？"*

RK3588 上每次 CPU 回退约带来 ~40% 的延迟增加。因此：
- `IBNorm` 已从模型架构中手术级剔除（它包含 `InstanceNorm2d`，RKNN 编译器无法将其映射到 NPU 硬件，会静默回退 CPU 执行）
- 对预训练模型采用「反融合」技术（将 InstanceNorm 替换为编译器无法识别为 InstanceNorm 的算术原语）
- 通过零拷贝 RKNN 缓冲区消除 CPU 与 NPU 之间的内存拷贝

### 2. 高分辨率训练，低分辨率部署

模型以 **512×512** 训练以保留梯度中的细发丝细节，但在 NPU 上以 **256×256** 部署以最大化吞吐量。Phase 3 通过 **Guided Filter** 后处理弥补分辨率差距——使用原始 1080P 图像将低分辨率透明遮罩锐化回高质量结果。

这种「高分辨率训练，低分辨率部署」策略避免了训练质量与推理速度之间的权衡。

### 3. 纯 BN 模型优于图改写

从 MODNet 中去除 InstanceNorm 有两种方式：
- **方案 A**（v0.8.0 采用）：利用 `Var(x) = E[x²] − (E[x])²` 恒等式对现有 ONNX checkpoint 做图改写（「反融合」）
- **方案 B**（Phase 1 采用）：将 `IBNorm` 替换为 `BatchNorm2d` 后从头重训

方案 A 是权宜之计——有效，但若编译器变得更能识别折叠后的模式则会失效。方案 B 产出的模型从设计之初就是 NPU 原生的。方案 B 是长期策略。方案 A 的代码保留在 `modnet_onnx_modified.py` 中，以向后兼容已部署的模型。

### 4. 模块化 C++ 流水线

C++ 推理引擎分为三个阶段，以干净的契约（`TensorData` 结构体）相衔接：
- **前端**（CPU）：图像加载、颜色转换、letterbox 缩放
- **推理引擎**（NPU 或 CPU）：模型执行，编译期选定
- **后端**（CPU）：输出解码、letterbox 裁剪还原、图像合成

这种分离使得交换推理后端（RKNN 零拷贝、RKNN 标准、ONNX Runtime）无需改动前端或后端代码，同时让每个阶段均可通过二进制转储文件独立验证。

### 5. CLI 作为开发接口

`helmsman` CLI 将所有工作流（Python 环境配置、ONNX 导出、C++ 构建、golden 文件生成）统一抽象到单一接口之后，确保：
- 新贡献者有明确的单一起点
- AI Agent 只需学习一套命令
- 复杂的编排（pyenv → venv → pip → conan → cmake）具有确定性和可复现性

---

## 技术选型

| 决策 | 考虑过的替代方案 | 选择理由 |
|---|---|---|
| MODNet (ZHKKKe) | Background Matting V2、RobustVideoMatting | 骨干网络更轻量（MobileNetV2）；质量与速度平衡好；子模块活跃维护 |
| RK3588 NPU (RKNN) | Coral TPU、Hailo-8 | 客户需求；该价位提供 6 TOPS |
| CMake FetchContent | Conan、vcpkg、源码内置 | 零外部包管理器依赖；固定 URL 和哈希保证可复现 |
| ONNX opset 11 | opset 12+、TensorRT | 最大化 RKNN Toolkit 2 兼容性；低 opset = 不支持算子更少 |
| Python 3.8 | Python 3.10+ | 满足 RKNN Toolkit 2 宿主机要求；onnxruntime 1.6.0 兼容性 |
| pyenv + .venv | Docker、conda | 轻量；训练时无需 Docker 额外开销 |
| P3M-10k 数据集 | PPM-100、DIM-481 | 最大的公开可用人像抠图数据集；包含经过隐私处理的真实人脸图像 |

---

## 「完成」的定义

项目「完成」的条件：

1. ✅ ~~模型在 RK3588 上运行无 CPU 回退~~ (v0.8.0 — InstanceNorm 已消除)
2. ✅ ~~纯 BN 模型重训完成~~ (Phase 1 R1 Block 1.2)
3. ✅ ~~纯 BN 模型已导出并验证~~ (Phase 1 R1 Block 1.4)
4. ✅ ~~Guided Filter 后处理在完整 1080P 分辨率下恢复发丝细节~~ (Phase 2 R1 GF Exp-08)
5. ✅ ~~Pure-BN FP16 RKNN 板端 benchmark 完成，BN 提速 30-45%~~ (Phase 2 R1 FP16)
6. 🔥 **INT8 量化达到目标延迟** (Phase 2 R1 INT8 — 当前，288×512 优先)
7. ⏳ C++ 流水线产品化（IPC 集成、多线程）(Phase 3)
8. ⏳ EMA 时序平滑消除视频输出中的抖动 (Phase 4)
9. ⏳ 完整流水线在 RK3588 上基准测试达到 30 FPS @ 1080P

---

## 非目标

- **设备端实时训练** — 训练在 x86 GPU 服务器上离线完成
- **多人抠图** — MODNet 专为单人像抠图设计
- **Windows 支持** — 工具链和部署目标仅限 Linux
- **Web 部署** — 这是嵌入式/边缘系统，不是 SaaS 产品
- **YOLO 风格目标检测集成** — 纯背景消除；不含目标检测
