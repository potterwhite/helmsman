# Master Plan — RK3588 Edge Portrait Matting at 30 FPS

> **Project**: helmsman / MODNet Pure-BN Deployment
> **Target**: RK3588 NPU, 1080P input, < 33ms per frame, hair-level edges
> **Last Updated**: 2026-03-19 (training in progress)
> **English →** [../../en/2-progress/MASTER_PLAN.md](../../en/2-progress/MASTER_PLAN.md)

---

## Global Pipeline

```
Input: 1080P video stream
  -> [Phase 1] Retrained Pure-BN MODNet
  -> [Phase 2] INT8 Quantized .rknn model
  -> [Phase 3] NPU inference + Guided Filter post-processing
  -> [Phase 4] Temporal smoothing for video stability
Output: 1080P alpha matte at 30 FPS with hair-level edges
```

---

## Phase 1: Model Retraining

| Block | Task | Status | Date | Notes |
|-------|------|--------|------|-------|
| 1.0 | DataLoader + Dynamic Trimap | ✅ DONE | 2026-03-19 | P3MDataset class, 9421 samples |
| 1.1 | IBNorm -> BatchNorm Surgery | ✅ DONE | 2026-03-13 | modnet.py modified, MVP 50-batch passed |
| 1.2 | Fine-tune Training | ✅ DONE | 2026-03-19 | 15ep, pretrained BB, 512x512 |
| 1.3 | [Fallback] Knowledge Distillation | ⏳ Canceled | - | Only if 1.2 quality is poor |
| 1.4 | ONNX Export + Verification | **⏳ IN PROGRESS** | - | Fixed 512x512, check Netron |

### Block 1.2 Training Progress

```
Strategy: Fast Validation
Config:   15 epochs, BS=8, backbone_pretrained=True, 512x512, StepLR

Epoch  Train Loss  Val L1    Best?   LR
  1     0.5410     0.0264    YES    0.01
  2     0.3054     0.0175    YES    0.01
  3     0.2433     0.0146    YES    0.01
  4     0.2187     0.0132    YES    0.01
  5     (running)   -         -     0.01  <- last epoch at LR=0.01
  6      -          -         -     0.001 <- LR drops (StepLR kicks in)
  ...
  15     -          -         -     0.0001

Status: HEALTHY - all epochs setting new best, no overfitting signal.
```

### Decision Point After Block 1.2

```
IF visual quality is good (body intact, edges reasonable):
  -> Block 1.4 (ONNX Export)
  -> Skip Block 1.3

IF quality is poor (holes, noise, worse than original MODNet):
  -> Block 1.3 (Knowledge Distillation using original as teacher)
  -> Then Block 1.4
```

---

## Phase 2: INT8 Quantization

| Block | Task | Status | Notes |
|-------|------|--------|-------|
| 2.1 | Calibration dataset (200 images) | ⏳ Pending | Random subset of P3M-10k |
| 2.2 | RKNN Toolkit conversion | ⏳ Pending | asymmetric INT8 |
| 2.3 | Board profiling (speed + quality) | ⏳ Pending | Target: < 25ms at 256x256 |
| 2.4 | [Fallback] Mixed precision | ⏳ Standby | If INT8 quality crashes |

---

## Phase 3: AI + CV Hybrid Pipeline

| Block | Task | Status | Notes |
|-------|------|--------|-------|
| 3.1 | Guided Filter integration | ⏳ Pending | OpenCV ximgproc or custom OpenCL |
| 3.2 | Heterogeneous pipeline assembly | ⏳ Pending | CPU/RGA resize + NPU + GPU filter |

---

## Phase 4: Video Temporal Stability

| Block | Task | Status | Notes |
|-------|------|--------|-------|
| 4.1 | EMA temporal smoothing | ⏳ Pending | alpha=0.3 current + 0.7 previous |

---

## Key Files

| File | Purpose |
|------|---------|
| `MODNet.git/src/models/modnet.py` | Pure-BN model architecture |
| `MODNet.git/train_modnet_block1_2.py` | Block 1.2 training script |
| `MODNet.git/checkpoints/modnet_bn_best.ckpt` | Best trained model |
| `MODNet.git/output/epoch_XX_val.png` | Visual validation results |
| `MODNet.git/onnx/export_onnx_modified.py` | ONNX export script |
| `runtime/cpp/apps/matting/` | C++ inference pipeline |
| `docs/knowledge/` | Knowledge base for self-study |
| `docs/blocks/` | Per-block execution documentation |
