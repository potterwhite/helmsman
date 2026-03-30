# Helmsman — Progress

> Last updated: 2026-03-30
> **中文版 →** [../../zh/2-progress/progress.md](../../zh/2-progress/progress.md)

---

## Overall Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Infrastructure & ONNX baseline | ✅ Done (v0.1.0 – v0.3.0) |
| **Phase 1** | Model Retraining (Pure-BN MODNet) | 🔄 In Progress (~80%) |
| **Phase 2** | INT8 Quantization (RKNN Toolkit 2) | ⏳ Pending |
| **Phase 3** | AI+CV Hybrid Pipeline (Guided Filter) | ⏳ Pending |
| **Phase 4** | Temporal Smoothing (Video EMA) | ⏳ Pending |

**Currently active:** Phase 1 Block 1.4 — ONNX Export + Verification of retrained Pure-BN checkpoint

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
| **1.4** | ONNX Export + Verification | 🔥 In progress | — |

### Block 1.0 — DataLoader + Dynamic Trimap ✅ DONE (2026-03-19)

`P3MDataset` class in `train_modnet_block1_2.py`:
- Loads P3M-10k dataset (9,421 training samples)
- Dynamic trimap generation via morphological dilation/erosion on ground truth mask
- Color jitter augmentation (brightness/contrast/saturation/hue)
- Output: `(image_tensor, trimap_tensor, gt_matte_tensor)` per sample

**Dataset structure expected**:
```
P3M-10k/
  train/
    blurred_image/*.jpg
    mask/*.png
  validation/P3M-500-P/
    blurred_image/*.jpg
    mask/*.png
```

### Block 1.1 — IBNorm → BatchNorm Surgery ✅ DONE (2026-03-13)

Modified `third-party/scripts/modnet/src/models/modnet.py`:

**Before** — `IBNorm` splits channels: half BatchNorm + half InstanceNorm
**After** — `Conv2dIBNormRelu` uses only `nn.BatchNorm2d` (Pure-BN). No `InstanceNorm2d` anywhere in the model.

Verification: MVP smoke test (50 batches) passed without NaN — architecture healthy.

### Block 1.2 — Fine-Tune Training ✅ DONE (2026-03-19)

```
Epochs: 15  |  BS: 8  |  LR: 0.01 → StepLR(step=5, gamma=0.1)
Optimizer: SGD(momentum=0.9)  |  Input: 512×512
Backbone: pretrained=True (ImageNet MobileNetV2, frozen)
Val: P3M-500-P (500 images), every epoch
```

Training results:
```
Epoch 1:  Train Loss 0.5410  Val L1 0.0264  → new best ✅
Epoch 2:  Train Loss 0.3054  Val L1 0.0175  → new best ✅
Epoch 3:  Train Loss 0.2433  Val L1 0.0146  → new best ✅
Epoch 4:  Train Loss 0.2187  Val L1 0.0132  → new best ✅
Epochs 5-15: Completed (HEALTHY, no overfitting)
```

Output: `third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt`

**Decision**: Block 1.3 (Knowledge Distillation) CANCELED — training quality sufficient to proceed directly.

### Block 1.4 — ONNX Export + Verification 🔥 IN PROGRESS

**Goal**: Export `modnet_bn_best.ckpt` → `.onnx`, verify no InstanceNorm nodes, validate against C++ golden files.

**Steps**:
1. Activate venv: `source .venv/bin/activate`
2. Navigate: `cd third-party/sdk/MODNet.git`
3. Export: `python3 -m onnx.export_onnx_modified --ckpt-path=checkpoints/modnet_bn_best.ckpt --output-path=checkpoints/modnet_bn_best.onnx`
4. Verify in Netron: confirm no `InstanceNormalization` nodes
5. Generate golden files: `./helmsman golden` (select `modnet_bn_best.onnx` + test image)
6. Build C++ native: `./helmsman build cpp cb`
7. Run C++ inference: `Helmsman_Matting_Client <img> checkpoints/modnet_bn_best.onnx <output_dir>`
8. Compare: `python3 tools/MODNet/verify_golden_tensor.py`

**Success criteria**:
- No `InstanceNormalization` in Netron
- C++ output matches Python golden within float tolerance
- Visual alpha matte shows clean edges with hair detail

---

## Phase 2 — INT8 Quantization (Pending)

| Block | Description | Status |
|---|---|---|
| **2.1** | Calibration Dataset (200 images from P3M-10k) | ⏳ Pending Block 1.4 |
| **2.2** | RKNN Toolkit 2 conversion (`modnet_bn_best.onnx → .rknn`) | ⏳ Pending 2.1 |
| **2.3** | Board Profiling (latency on RK3588) | ⏳ Pending 2.2 |
| **2.4** | Mixed Precision Fallback (if INT8 quality degrades) | ⏳ Standby |

**Target**: < 25ms @ 256×256 (scales to ~33ms @ 1080P with preprocessing).

---

## Phase 3 — AI+CV Hybrid Pipeline (Pending)

| Block | Description | Status |
|---|---|---|
| **3.1** | Guided Filter Integration (hair detail recovery) | ⏳ Pending Phase 2 |
| **3.2** | Heterogeneous Pipeline (CPU/RGA + NPU + Mali GPU) | ⏳ Pending 3.1 |

**Rationale**: Train at 512×512 but deploy at 256×256. Guided Filter uses the original 1080P image to recover hair details that low-resolution NPU inference loses.

---

## Phase 4 — Video Temporal Stability (Pending)

| Block | Description | Status |
|---|---|---|
| **4.1** | EMA Temporal Smoothing (`alpha = 0.3·model + 0.7·previous`) | ⏳ Pending Phase 3 |

---

## Performance Benchmarks (Historical)

| Date | Model | Resolution | Latency | Hardware | Notes |
|---|---|---|---|---|---|
| 2026-03-04 | ONNX w/ InstanceNorm | 512×512 | 287.66ms | x86 native | v0.6.0 baseline |
| 2026-03-12 | RKNN INT8 anti-fusion | 576×1024 | ~430ms | RK3588 NPU | v0.8.0; ~40% better than w/ InstanceNorm |

---

## Architecture Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-04 | Zero-Copy RKNN inference | Eliminates CPU→NPU memory copy overhead |
| 2026-03-04 | Manual INT8 quantization in C++ | Bypasses NPU normalization overhead; controls data range |
| 2026-03-07 | Letterbox + padding metadata propagation | Fixes output size mismatch when input aspect ≠ model aspect |
| 2026-03-07 | Dynamic model input size reading | No more hard-coded 512; supports any model input dimension |
| 2026-03-12 | Anti-fusion via `Var(x) = E[x²] − (E[x])²` | Prevents RKNN compiler from reconstructing InstanceNorm → CPU fallback |
| 2026-03-13 | IBNorm → BatchNorm in training architecture | Train a native Pure-BN model (not just graph-rewrite a pretrained one) |
| 2026-03-19 | 512×512 training, 256×256 deployment | Hair detail survives training; Phase 3 GF recovers at deployment |
| 2026-03-19 | backbone_pretrained=True | 5–10× faster convergence; MobileNetV2 backbone unchanged |

---

## Known Issues / Blockers

| Issue | Status | Notes |
|---|---|---|
| `modnet_bn_best.ckpt` visual quality not yet verified | 🔥 Active | Pending Block 1.4 ONNX export + visual check |
| P3M-10k dataset path hardcoded in training script | ⚠️ Known | Edit paths manually at top of `train_modnet_block1_2.py` |
| `evboard` hostname + credentials hardcoded in deploy scripts | ⚠️ Known | See `tools/deploy_and_test.sh` |
