# Helmsman — Progress

> Last updated: 2026-04-06
> **English →** [../../en/2-progress/progress.md](../../en/2-progress/progress.md)

---

## Overall Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Infrastructure & ONNX baseline | ✅ Done (v0.1.0 – v0.3.0) |
| **Phase 1** | Model Retraining (Pure-BN MODNet) | 🔄 In Progress (~80%) |
| **Phase 2** | INT8 Quantization (RKNN Toolkit 2) | 🔜 当前（质量门槛：中规中矩即可） |
| **Phase 3** | AI+CV Hybrid Pipeline (Guided Filter) | ✅ Block 3.1 Done |
| **Phase 4** | Temporal Smoothing (Video EMA) | ⏳ Pending |

**Currently active:** Phase 2 — INT8 Quantization (288×512 优先) | Phase 3 Block 3.1 GF ✅ converged | Phase 1 Block 1.4 ✅

---

## Phase 0 — Infrastructure & ONNX Baseline

| Step | Description | Commit |
|---|---|---|
| **0.1** | Initial commit; pyenv setup; `.ckpt → ONNX` pipeline working | v0.1.0 |
| **0.2** | MIT license headers; release-please automation | v0.2.0 |
| **0.3** | Refactored to full-featured CLI; bit-exact C++ NumPy preprocessing | v0.3.0 |
| **0.4** | C++ runtime restructured (libs/apps); CMakePresets; cross-compile | v0.4.0 |
| **0.5** | Replaced Conan-based ORT with CMake FetchContent; `ScaleFactor` struct | v0.5.0 |
| **0.6** | Zero-copy RKNN inference; INT8 manual quantization; 287ms @ 512×512 | v0.6.0 |
| **0.7** | Dynamic input shape + letterbox metadata; output matches original resolution | v0.7.0 |
| **0.8** | InstanceNorm elimination via graph rewrite (anti-fusion); ~40% latency improvement | v0.8.0 |

---

## Phase 1 — Model Retraining (Pure-BN) (In Progress)

| Block | Description | Status | Commit |
|---|---|---|---|
| **1.0** | DataLoader + Dynamic Trimap (`P3MDataset`) | ✅ Done | 2026-03-19 |
| **1.1** | IBNorm → BatchNorm Surgery in `modnet.py` | ✅ Done | 2026-03-13 |
| **1.2** | Fine-tune Training (15 epochs, P3M-10k) | ✅ Done | 2026-03-19 |
| **1.3** | Knowledge Distillation | ⏳ Canceled | Not needed; Block 1.2 quality sufficient |
| **1.4** | ONNX Export + Verification | 🟡 部分完成 | — |

### Block 1.0 — 数据加载器 + 动态 Trimap ✅ DONE (2026-03-19)

`train_modnet_block1_2.py` 中的 `P3MDataset` 类：
- 加载 P3M-10k 数据集（9,421 个训练样本）
- 通过对真值遮罩做形态学膨胀/腐蚀生成动态 trimap
- 色彩扰动增强（亮度/对比度/饱和度/色调）
- 每个样本输出：`(image_tensor, trimap_tensor, gt_matte_tensor)`

**预期数据集结构**：
```
P3M-10k/
  train/
    blurred_image/*.jpg
    mask/*.png
  validation/P3M-500-P/
    blurred_image/*.jpg
    mask/*.png
```

### Block 1.1 — IBNorm → BatchNorm 手术 ✅ DONE (2026-03-13)

修改文件：`third-party/scripts/modnet/src/models/modnet.py`

**修改前** — `IBNorm` 分割通道：一半 BatchNorm + 一半 InstanceNorm
**修改后** — `Conv2dIBNormRelu` 仅使用 `nn.BatchNorm2d`（纯 BN）。模型中无任何 `InstanceNorm2d`。

验证：MVP 冒烟测试（50 个 batch）通过，无 NaN——架构健康。

### Block 1.2 — 微调训练 ✅ DONE (2026-03-19)

```
Epochs: 15  |  BS: 8  |  LR: 0.01 → StepLR(step=5, gamma=0.1)
优化器: SGD(momentum=0.9)  |  输入: 512×512
骨干网络: pretrained=True (ImageNet MobileNetV2, 冻结)
验证集: P3M-500-P (500 张图), 每轮验证一次
```

训练结果：
```
Epoch 1:  Train Loss 0.5410  Val L1 0.0264  → 新最优 ✅
Epoch 2:  Train Loss 0.3054  Val L1 0.0175  → 新最优 ✅
Epoch 3:  Train Loss 0.2433  Val L1 0.0146  → 新最优 ✅
Epoch 4:  Train Loss 0.2187  Val L1 0.0132  → 新最优 ✅
Epochs 5-15: 已完成（训练健康，无过拟合信号）
```

输出：`third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt`

**决策**：Block 1.3（知识蒸馏）已取消——训练质量已满足直接进入下一步的要求。

### Block 1.4 — ONNX 导出 + 验证 🟡 部分完成（2026-03-31）

**目标**：将 `modnet_bn_best.ckpt` 导出为 `.onnx`，验证不含 InstanceNorm 节点，并与 C++ golden 文件对比验证。

**已完成步骤**：
1. ✅ 创建 `export_onnx_pureBN.py`（`third-party/scripts/modnet/onnx/`，符号链接到 `MODNet.git/onnx/`）
2. ✅ ONNX 导出成功：`modnet_bn_best.ckpt` → `modnet_bn_best_pureBN.onnx`（25M，opset 11）
3. ✅ 节点验证通过：0 个 `InstanceNormalization` 节点；BN 节点已 fold 进 Conv（正常）

**修复的 Bug（2026-03-31）**：
- `func_3_3_rebuild_sdk`：submodule 未初始化时 reset 到错误 HEAD（commit `1a88cbe`）
- `onnx==1.8.1` 不兼容 PyTorch 2.0.1 → 升级至 `onnx==1.14.1`，`onnxruntime==1.6.0` → `1.15.1`（commit `554ca64`）
- 本地 `MODNet.git/onnx/` 目录遮蔽 pip `onnx` 包 → 在 `export_onnx_pureBN.py` 中做 sys.path 手术（commit `554ca64`）

**剩余步骤**（需要测试图片放入 `helmsman.git/media/`）：
4. 🔜 把测试图片（jpg/png）放入 `helmsman.git/media/`
5. 🔜 生成 golden 文件：`./helmsman golden`（选 `modnet_bn_best_pureBN.onnx` + 测试图片）
6. 🔜 构建 C++ 原生：`./helmsman build cpp cb`
7. 🔜 运行 C++ 推理：`Helmsman_Matting_Client <img> checkpoints/modnet_bn_best_pureBN.onnx <output_dir>`
8. 🔜 对比：`python3 tools/MODNet/verify_golden_tensor.py`

**成功标准**：
- C++ 输出与 Python golden 在浮点误差范围内一致
- 视觉透明遮罩边缘清晰，发丝细节可见

---

## Phase 2 — INT8 量化（🔜 当前）

> **质量门槛**（2026-04-06 更新）：中规中矩即可 — 不出大面积全黑/全噪点级别退化。
> 原"95~99% 满意度才启动"已放宽：实际需求是追求速度，FP16 质量已确认可接受。

| Block | 描述 | 状态 |
|---|---|---|
| **Q.1** | 校准数据集（P3M-10k 200 张 + 20 张复杂背景） | ⏳ 待开始 |
| **Q.2** | RKNN Toolkit 2 转换（`modnet_bn_best_pureBN.onnx → .rknn`，INT8 288×512） | ⏳ 待 Q.1 |
| **Q.3** | 板端性能基准（10轮 benchmark，INT8 vs FP16 对比） | ⏳ 待 Q.2 |
| **Q.4** | 质量评估（INT8 vs FP16，退化容忍标准） | ⏳ 待 Q.3 |
| **Q.5** | 256×256 INT8 对比（速度甜点探索） | ⏳ 待 Q.2 |
| **Q.6** | 混合精度回退（若 INT8 质量下降） | ⏳ 备用 |
| **Q.7** | 最终结论与推荐配置 | ⏳ 待 Q.3~Q.5 |

**目标**：INT8 288×512 < 35ms（配合 GF 后处理 ~42ms 端到端 ≈ 24fps）；INT8 256×256 目标 30fps+。

---

## Phase 3 — AI+CV 混合流水线

| Block | 描述 | 状态 |
|---|---|---|
| **3.1** | Guided Filter 集成（发丝细节恢复） | ✅ 完成（Exp-08 最优：r=2, ε=1e-4, thr=0.2, erode=1） |
| **3.2** | 异构流水线（CPU/RGA + NPU + Mali GPU） | ⏳ 待 3.1 |

**设计依据**：以 512×512 训练但以 256×256 部署。Guided Filter 使用原始 1080P 图像恢复低分辨率 NPU 推理丢失的发丝细节。

**Block 3.1 收敛结论（2026-04-02，Exp-01 ~ Exp-11）**：
- 最终参数：`GuidedFilterPostProcessor(r=2, ε=1e-4, threshold=0.2, erode_iters=1)`
- 流程：`threshold(0.2) → erode(3×3, iter=1) → GF(r=2, ε=1e-4)`
- 效果：光晕消除 ✅，语义轮廓正确 ✅，躯干无空洞 ✅，发丝有结构 ✅
- 剩余轻微锯齿为 GF 精确吸附 1080P 物理边界的自然结果，不可通过参数调优消除
- 进一步改善需依赖更高质量的模型输出（Phase 1 重训或 INT8 后处理）

---

## Phase 4 — 视频时序稳定（待处理）

| Block | 描述 | 状态 |
|---|---|---|
| **4.1** | EMA 时序平滑（`alpha = 0.3·当前帧 + 0.7·前一帧`） | ⏳ 待 Phase 3 |

---

## 性能基准（历史数据）

| 日期 | 模型 | 分辨率 | 延迟 | 硬件 | 备注 |
|---|---|---|---|---|---|
| 2026-03-04 | ONNX（含 InstanceNorm） | 512×512 | 287.66ms | x86 原生 | v0.6.0 基准 |
| 2026-03-12 | RKNN INT8 反融合 | 576×1024 | ~430ms | RK3588 NPU | v0.8.0；比含 InstanceNorm 版本快 ~40% |
| 2026-04-03 | RKNN FP16 PureBN | 256×256 | ~30ms | RK3588 3NPU | BN 融合后 ~44% faster than IN |
| 2026-04-03 | RKNN FP16 PureBN | 288×512 | ~61ms | RK3588 3NPU | BN 融合后 ~45% faster than IN |
| 2026-04-03 | RKNN FP16 PureBN | 384×384 | ~66ms | RK3588 3NPU | BN 融合后 ~39% faster than IN |
| 2026-04-03 | RKNN FP16 PureBN | 576×1024 | ~261ms | RK3588 3NPU | BN 融合后 ~34% faster than IN |

---

## 架构决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-03-04 | RKNN 零拷贝推理 | 消除 CPU→NPU 内存拷贝开销 |
| 2026-03-04 | C++ 中手动 INT8 量化 | 绕过 NPU 归一化开销；掌控数据范围 |
| 2026-03-07 | Letterbox + 填充元数据传递 | 修复输入宽高比≠模型输入比时的输出尺寸不匹配 |
| 2026-03-07 | 动态读取模型输入尺寸 | 不再硬编码 512；支持任意模型输入维度 |
| 2026-03-12 | 反融合：`Var(x) = E[x²] − (E[x])²` | 防止 RKNN 编译器重构 InstanceNorm → CPU 回退 |
| 2026-03-13 | 训练架构中 IBNorm → BatchNorm | 训练原生纯 BN 模型（而非仅对预训练模型做图改写） |
| 2026-03-19 | 512×512 训练，256×256 部署 | 发丝细节在训练中得以保留；Phase 3 GF 在部署时恢复 |
| 2026-03-19 | backbone_pretrained=True | 收敛速度快 5–10 倍；MobileNetV2 骨干不做修改 |

---

## 已知问题 / 阻塞项

| 问题 | 状态 | 备注 |
|---|---|---|
| `modnet_bn_best.ckpt` 视觉质量尚未验证 | 🔜 待完成 | ONNX 导出 ✅；待 golden 生成 + C++ 对比（需测试图片放入 `media/`） |
| P3M-10k 数据集路径在训练脚本中硬编码 | ⚠️ 已知 | 手动修改 `train_modnet_block1_2.py` 顶部的路径 |
| `evboard` 主机名 + 凭证在部署脚本中硬编码 | ⚠️ 已知 | 见 `tools/deploy_and_test.sh` |
