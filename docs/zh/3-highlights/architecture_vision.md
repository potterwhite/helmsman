# Helmsman — Architecture Vision

> **Purpose**: Explains *why* the project is built the way it is.
> Strategic positioning, design philosophy, and the reasoning behind key decisions.
> **English →** [../../en/3-highlights/architecture_vision.md](../../en/3-highlights/architecture_vision.md)

---

## The Problem We're Solving

**Portrait matting** (background removal with hair-level alpha) is computationally intensive.
Existing solutions are either:
- Cloud-based (latency + privacy concerns)
- Desktop GPU-dependent (not embeddable)
- Slow on embedded hardware (>500ms makes real-time impossible)

**Helmsman** solves this by deploying a trained deep learning model on the **Rockchip RK3588 NPU** (6 TOPS), targeting **30 FPS @ 1080P** — fast enough for real-time video conferencing, video production, and live edge AI applications.

---

## Design Philosophy

### 1. NPU-First, Not CPU-Fallback

Every architectural decision is evaluated against one question:
> *"Will this cause the RKNN compiler to fall back to CPU execution?"*

CPU fallback on the RK3588 adds ~40% latency per offloaded operation. This is why:
- `IBNorm` was surgically removed from the model architecture (it contains `InstanceNorm2d`, which the RKNN compiler cannot map to NPU hardware and silently offloads to CPU)
- The "anti-fusion" technique is used for pretrained models (replacing InstanceNorm with arithmetic primitives that the compiler cannot identify as InstanceNorm)
- Memory copies between CPU and NPU are eliminated via zero-copy RKNN buffers

### 2. Train High, Deploy Low

The model trains at **512×512** to preserve fine hair detail in gradients, but is deployed at **256×256** on the NPU for maximum throughput. The resolution gap is bridged in Phase 3 by **Guided Filter** post-processing, which uses the original 1080P source image to sharpen the low-res alpha matte back to high quality.

This "train high, deploy low" strategy avoids the trade-off between training quality and inference speed.

### 3. Pure-BN Over Graph Rewriting

There are two ways to remove InstanceNorm from MODNet:
- **Option A** (applied in v0.8.0): Graph rewrite on existing ONNX checkpoint using the `Var(x) = E[x²] − (E[x])²` identity ("anti-fusion")
- **Option B** (applied in Phase 1): Retrain from scratch with `IBNorm` replaced by `BatchNorm2d`

Option A is a workaround — effective but fragile if the compiler becomes smarter at detecting folded patterns. Option B produces a model that was *designed* to be NPU-native from the start. Option B is the long-term strategy. Option A's code is kept in `modnet_onnx_modified.py` for backward compatibility with already-deployed models.

### 4. Modular C++ Pipeline

The C++ inference engine is split into three stages with a clean contract (`TensorData` struct):
- **Frontend** (CPU): image loading, color conversion, letterbox resize
- **InferenceEngine** (NPU or CPU): model execution, compile-time selected
- **Backend** (CPU): output decoding, letterbox crop removal, compositing

This separation makes it possible to swap inference backends (RKNN zero-copy, RKNN standard, ONNX Runtime) without touching frontend or backend code. It also makes testing easier — each stage can be validated independently via binary dump files.

### 5. CLI as a Development Interface

The `helmsman` CLI abstracts all workflows (Python setup, ONNX export, C++ build, golden file generation) behind a single unified interface. This ensures:
- New contributors have one place to start
- AI agents have one command set to learn
- Complex orchestration (pyenv → venv → pip → conan → cmake) is deterministic and reproducible

---

## Technology Choices

| Decision | Alternative Considered | Reason Chosen |
|---|---|---|
| MODNet (ZHKKKe) | Background Matting V2, RobustVideoMatting | Lighter backbone (MobileNetV2); good balance of quality and speed; active submodule |
| RK3588 NPU (RKNN) | Coral TPU, Hailo-8 | Customer requirement; 6 TOPS for the price point |
| CMake FetchContent | Conan, vcpkg, vendored | Zero external package manager dependency; pinned URLs and hashes for reproducibility |
| ONNX opset 11 | opset 12+, TensorRT | Maximum RKNN Toolkit 2 compatibility; lower opset = fewer unsupported ops |
| Python 3.8 | Python 3.10+ | Matches RKNN Toolkit 2 host requirements; onnxruntime 1.6.0 compatibility |
| pyenv + .venv | Docker, conda | Lightweight; no Docker overhead for training on host GPU |
| P3M-10k dataset | PPM-100, DIM-481 | Largest publicly available portrait matting dataset; real privacy-safe blurred faces |

---

## What "Done" Looks Like

The project is "done" when:

1. ✅ ~~Model runs on RK3588 without CPU fallback~~ (v0.8.0 — InstanceNorm eliminated)
2. ✅ ~~Pure-BN model retrained~~ (Phase 1 Block 1.2 complete)
3. 🔥 **Pure-BN model exported and verified** (Phase 1 Block 1.4 — current)
4. ⏳ INT8 quantized RKNN model achieves <25ms @ 256×256 on RK3588
5. ⏳ Guided Filter post-processing recovers hair detail at full 1080P resolution
6. ⏳ EMA temporal smoothing eliminates flicker in video output
7. ⏳ Full pipeline benchmarked at 30 FPS @ 1080P on RK3588

---

## Non-Goals

- **Real-time training on device** — training is done offline on an x86 GPU server
- **Multi-person matting** — MODNet is designed for single-subject portrait matting
- **Windows support** — the toolchain and deployment targets are Linux only
- **Web deployment** — this is an embedded/edge system, not a SaaS product
- **YOLO-style detection integration** — pure background removal only; no object detection
