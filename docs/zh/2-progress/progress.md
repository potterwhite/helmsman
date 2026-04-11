# Helmsman — Progress

> Last updated: 2026-04-11
> **English →** [../../en/2-progress/progress.md](../../en/2-progress/progress.md)

---

## 你现在在哪？

**→ MR2 Phase-3 C++ 集成启动 ← 你在这里**

---

## Roadmap 全景

本项目有两个**迭代域**和两个**线性阶段**。迭代域之间可以回环；线性阶段依次推进。

```
                    ┌──────────────────────────────────────────┐
                    │         迭代域（做到满意为止）              │
                    │                                          │
                    │   模型域                    部署域         │
                    │   (训练 & 导出)              (量化 & 调优)  │
                    │                                          │
                    │   R1: Pure-BN ✅ ────→ R1: FP16 ✅       │
                    │                            GF  ✅       │
                    │                            INT8 ⏳       │
                    │                                          │
                    │   R2: RVM Phase-2 ✅   R2: Phase-3 C++   │
                    │                            ← 你在这里     │
                    │                                          │
                    └──────────────────────────────────────────┘
                                        │
                                        │ Phase-3/4 完成后 ↓
                                        │
                    ┌──────────────────────────────────────────┐
                    │         线性阶段（依次推进）               │
                    │                                          │
                    │   产品化 → 实时视频                        │
                    │   (IPC/多线程/RGA)   (EMA 时序平滑)        │
                    └──────────────────────────────────────────┘
```

**回环规则**：部署域评估后若模型质量不足 → 回到模型域开新 Round → 再回部署域重新量化。直到满意后退出迭代域，进入线性阶段。

---

## 基础设施（已完成，v0.1.0 – v0.8.0）

CLI、Python 环境、ONNX 导出、C++ 推理引擎、零拷贝 RKNN、反融合。详见 CHANGELOG.md。

---

## 模型域

> **关注什么**：训练什么模型——架构、数据集、loss、导出格式
> **PKB 入口**：`model/retrain_modnet_roadmap.md`

### R1：Pure-BN 重训 ✅

| Block | 一句话 | 日期 |
|---|---|---|
| 1.0 | P3MDataset 数据加载器 + 动态 Trimap | 2026-03-19 |
| 1.1 | IBNorm → BatchNorm 手术 | 2026-03-13 |
| 1.2 | 15 epoch 微调，Val L1=0.0086 | 2026-03-19 |
| 1.4 | ONNX 导出（25MB, opset 11, 0 InstanceNorm） | 2026-03-31 |

输出：`modnet_bn_best.ckpt` → `modnet_bn_best_pureBN.onnx`

### R2：增强数据集 + 重训 🔜

> 触发条件：部署域 R1 评估后若识别多样性不够（复杂背景/侧面等）。
> 计划：P3M-10k + PPM-100 + 自有采集，更多 epoch，解冻骨干网络部分层。

---

## 部署域

> **关注什么**：怎么部署——量化类型、分辨率、GF 参数、板端 benchmark
> **PKB 入口**：`deploy/` 目录（gf/ rknn/ int8/）

### R2：RVM 视频抠图 ← current

> Gate-0 GO 2026-04-09，Phase-1 完成 2026-04-10，Phase-2 完成 2026-04-11。
> PKB 入口：`model/round2-rvm/plan-MR2-P3-cpp-design.md`（Phase-3 设计文档）

| Block | 描述 | 状态 |
|---|---|---|
| **1.0** | submodule 添加 + 环境搭建 + MattingNetwork 可导入验证 | ✅ 2026-04-10 |
| **1.1** | ONNX 导出精调：去除 refiner、固化 downsample_ratio | ✅ 2026-04-10 |
| **1.2** | PC 端 Python 推理验证（逐帧 + 递归状态 shape） | ✅ 2026-04-10 |
| **1.3** | ONNX 图简化（onnxsim） | ✅（ArcFoundry 内置） |
| **1.4** | ArcFoundry FP16 配置创建 | ✅ 2026-04-10 |
| **2.1** | ArcFoundry FP16 转换 → 9.3MB .rknn | ✅ 2026-04-10 |
| **2.3** | 板端单帧 benchmark：45.7ms @288×512（目标<80ms） | ✅ 2026-04-11 |
| **2.4** | 板端5帧递归验证：全 has_nan=False | ✅ 2026-04-11 |
| **2.5** | 256×256 build（9.0MB）+ 板测：20.5ms（理论48fps） | ✅ 2026-04-11 |
| **Phase-3** | C++ 集成 — Block 3.0~3.6 接口泛化 + RecurrentStateManager + Pipeline RVM 路径 | ✅ native 编译通过 ← 你在这里（待板测）|

---

### R1：FP16 + GF + INT8 🔥（并行推进）

**GF 后处理 ✅**：r=2, ε=1e-4, threshold=0.2, erode=1（Exp-08 收敛）

**FP16 RKNN Benchmark ✅**：BN 提速 30-45%

| 分辨率 | FP16 延迟 | 硬件 |
|---|---|---|
| 256×256 | ~30ms | RK3588 3NPU |
| 288×512 | ~61ms | RK3588 3NPU |
| 384×384 | ~66ms | RK3588 3NPU |
| 576×1024 | ~261ms | RK3588 3NPU |

**INT8 量化（R1 MODNet，暂停）**

> 质量门槛：中规中矩即可 — 不出大面积全黑/全噪点。

| Block | 描述 | 状态 |
|---|---|---|
| **Q.1** | 校准数据集（P3M-10k 200 张 + 20 张复杂背景） | ⏳ 待开始 |
| **Q.2** | RKNN Toolkit 2 转换（INT8 288×512） | ⏳ 待 Q.1 |
| **Q.3** | 板端性能基准（INT8 vs FP16） | ⏳ 待 Q.2 |
| **Q.4** | 质量评估 | ⏳ 待 Q.3 |
| **Q.5** | 256×256 INT8 对比（速度甜点） | ⏳ 待 Q.2 |
| **Q.6** | 混合精度回退 | ⏳ 备用 |
| **Q.7** | 最终结论 | ⏳ 待 Q.3~Q.5 |

目标：INT8 288×512 < 35ms；INT8 256×256 目标 30fps+。

**回环决策点**：Q.7 完成后——
- 质量 OK → 退出迭代域，进入产品化
- 质量不足 → 模型域 R2 → 部署域 R2

### R2：re-quantize 🔜（待模型域 R2 后触发）

---

## 产品化（待迭代域满意后进入）

> **PKB 参考**：`top/ipc-architecture-analysis.md`

| Block | 描述 | 状态 |
|---|---|---|
| **P.1** | IPC 架构：推理服务与导播台集成（dma_buf / shm） | ⏳ |
| **P.2** | 多线程流水线（推理线程 + I/O 线程解耦） | ⏳ |
| **P.3** | 异构加速（RGA resize、Mali GPU 后处理） | ⏳ |
| **P.4** | 内存优化（内存池、缓冲区复用） | ⏳ |

---

## 实时视频（待产品化后进入）

| Block | 描述 | 状态 |
|---|---|---|
| **V.1** | EMA 时序平滑 | ⏳ |

---

## 架构决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-03-04 | RKNN 零拷贝推理 | 消除 CPU→NPU 内存拷贝 |
| 2026-03-12 | 反融合 `Var(x)=E[x²]−(E[x])²` | 防 RKNN 重构 InstanceNorm → CPU 回退 |
| 2026-03-13 | IBNorm → BatchNorm 重训 | 原生 NPU 友好，不依赖图改写 |
| 2026-03-19 | 512 训练 / 256 部署 + GF | 训练保细节，GF 恢复 1080P 发丝 |
| 2026-04-06 | Roadmap 重构为双域迭代 + 线性阶段 | 实际工作流是 train→quantize→evaluate 循环 |

---

## 已知问题

| 问题 | 备注 |
|---|---|
| P3M-10k 路径硬编码 | 手动改 `train_modnet_block1_2.py` 顶部 |
| `evboard` 凭证硬编码 | 见 `tools/deploy_and_test.sh` |
