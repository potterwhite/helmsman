# KB-007: ONNX, RKNN, TensorRT — Model Format Conversion Chain

> **Target Audience**: Edge deployment engineers targeting RK3588 NPU
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `deployment`, `onnx`, `rknn`, `tensorrt`, `model-format`, `npu`

---

## 1. Why Can't NPU Directly Run PyTorch Models?

### The Language Barrier Analogy

```
PyTorch:  Speaks "Python + CUDA"  (GPU language)
RK3588 NPU: Speaks "RKNN"        (proprietary NPU language)
NVIDIA GPU: Speaks "TensorRT"     (NVIDIA-specific optimization)

They're different processors with different instruction sets.
It's like trying to run a Windows .exe on a Mac — it won't work.
You need a TRANSLATOR.
```

### The Translation Pipeline

```
PyTorch (.ckpt)          -- Source code (Python)
     |
     v  [torch.onnx.export()]
ONNX (.onnx)             -- Universal assembly language
     |
     +------+------+
     |      |      |
     v      v      v
  RKNN   TensorRT  OpenVINO    -- Platform-specific machine code
(.rknn)  (.engine)  (.xml/.bin)
  |        |         |
  v        v         v
RK3588   NVIDIA     Intel
  NPU     GPU       iGPU/CPU
```

---

## 2. ONNX — The Universal Translator

**ONNX = Open Neural Network Exchange**

### What Is It?

An open standard format that can represent ANY neural network as a
computational graph. Think of it as a "PDF for AI models" — any
framework can export to it, any runtime can read it.

```
ONNX file contains:
  1. Graph structure: which operations happen in what order
     Input -> Conv -> BN -> ReLU -> Conv -> BN -> ... -> Output

  2. Weight values: the actual numbers for each operation
     Conv1.weight = [0.03, -0.12, 0.45, ...]

  3. Shape information: tensor dimensions at each step
     Input: (1, 3, 512, 512)
     After Conv1: (1, 32, 256, 256)
     ...
```

### ONNX Operators (Ops)

Each operation in the network becomes an ONNX "operator":

| PyTorch Code              | ONNX Operator        | NPU Support |
|---------------------------|----------------------|-------------|
| `nn.Conv2d`               | `Conv`               | ✅ Native   |
| `nn.BatchNorm2d`          | `BatchNormalization` | ✅ Fused    |
| `nn.ReLU`                 | `Relu`               | ✅ Native   |
| `F.interpolate`           | `Resize`             | ✅ Native   |
| `torch.sigmoid`           | `Sigmoid`            | ✅ Native   |
| `nn.InstanceNorm2d`       | `ReduceMean + Pow + ...` | ❌ CPU fallback! |
| `nn.AdaptiveAvgPool2d(1)` | `GlobalAveragePool`  | ⚠️ Check    |

**This is why we removed InstanceNorm** — it decomposes into operators
that the NPU can't handle, forcing expensive CPU fallback.

### Viewing ONNX Models: Netron

**Netron** (https://netron.app/) is a free visual model viewer.
Upload your .onnx file and it shows the complete graph:

```
You can verify:
  ✅ Only Conv, BN, ReLU, Resize, Sigmoid operators present
  ❌ No ReduceMean, Pow, Sub, Div (InstanceNorm decomposition)
```

---

## 3. RKNN — RK3588's Native Language

**RKNN = Rockchip Neural Network**

### What Is It?

The compiled, optimized model format for Rockchip NPU hardware.
The rknn-toolkit2 compiler takes your ONNX model and:

1. **Operator fusion**: Merges Conv + BN + ReLU into a single operation
2. **Quantization**: Converts FP32 -> INT8 with calibration
3. **Memory planning**: Arranges tensors for optimal NPU memory access
4. **Instruction generation**: Creates NPU-native execution plan

```
ONNX Model:
  Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> ...
  (6 separate operations)

RKNN Model (after fusion):
  Conv+BN+ReLU -> Conv+BN+ReLU -> ...
  (2 fused operations — 3x fewer memory reads!)
```

### Why Conv+BN Fusion Is the Holy Grail

```
Conv2d:      output = weight × input + bias
BatchNorm:   output = gamma × (input - mean) / sqrt(var) + beta

Mathematical trick (offline, at compile time):
  fused_weight = weight × gamma / sqrt(var)
  fused_bias   = (bias - mean) × gamma / sqrt(var) + beta

Result:
  fused_output = fused_weight × input + fused_bias
  (This is just ONE Conv2d with modified weights!)

The BN operation VANISHES. Zero runtime cost.
Zero extra memory bandwidth. Pure speed.
```

**This ONLY works because BN statistics (mean, var) are FIXED after
training** (they come from running_mean and running_var in the checkpoint).
InstanceNorm can't be fused because it computes stats from each individual
input image at runtime.

### RKNN-Toolkit2 Conversion Code

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],   # input normalization
    std_values=[[127.5, 127.5, 127.5]],     # matches our [-1, 1] range
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8'  # INT8
)

rknn.load_onnx(model='modnet_bn.onnx')

rknn.build(
    do_quantization=True,
    dataset='calibration_images.txt'  # list of 200 image paths
)

rknn.export_rknn('modnet_bn_int8.rknn')
```

---

## 4. TensorRT — NVIDIA's Counterpart (For Reference)

If you ever deploy on NVIDIA Jetson (another edge platform):

```
ONNX -> TensorRT Engine (.engine)
  - Same concept as RKNN but for NVIDIA GPUs
  - Also does operator fusion, quantization, memory planning
  - Extremely well-documented (NVIDIA has huge ecosystem)
```

| Feature        | RKNN (Rockchip)      | TensorRT (NVIDIA)       |
|----------------|----------------------|-------------------------|
| Target HW      | RK3588/RK3566 NPU   | Jetson/Desktop GPU      |
| Quantization   | INT8, FP16           | INT8, FP16, INT4        |
| Toolkit        | rknn-toolkit2        | TensorRT + trtexec      |
| Profiling      | rknn.eval_perf()     | trtexec --dumpProfile   |
| Community      | Small (Chinese docs) | Massive (English docs)  |

---

## 5. The Static vs Dynamic Shape Problem

### Static Shape (Fixed Size)

```
Model compiled for: input = (1, 3, 256, 256) ONLY.
Feed it 512x512? CRASH.
Feed it 1080x1920? CRASH.

RKNN requires static shapes for NPU optimization.
```

### Dynamic Shape (Flexible Size)

```
Model can accept any size: input = (1, 3, H, W) where H,W vary.
Useful during development/testing.
NOT supported by RKNN NPU (would prevent optimization).
```

### How We Handle This

```
Original image: 1920x1080 (any size)
     |
     v  [Frontend: cv2.resize with letterbox]
NPU input: 256x256 (fixed, always)
     |
     v  [NPU inference]
NPU output: 256x256 alpha mask (fixed)
     |
     v  [Backend: remove padding, resize back]
Final mask: 1920x1080 (original size restored)
```

This is exactly what your C++ pipeline's frontend.cpp and backend.cpp do.

---

## 6. Opset Version: ONNX Compatibility

```python
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=11)
```

**Opset = Operator Set Version.** Like HTML5 vs HTML4 — newer opsets
support more operators.

| Opset | Key Additions                          | RKNN Support |
|-------|----------------------------------------|--------------|
| 9     | Basic Conv, BN, ReLU                   | ✅           |
| 11    | Resize with coordinate transforms      | ✅           |
| 13    | ReduceSum with dynamic axes            | ⚠️ Partial   |
| 16    | GridSample                             | ❌           |

**Rule**: Use the LOWEST opset that works. For MODNet, opset 11 is perfect.
Higher opsets may introduce operators that RKNN doesn't support.

---

## 7. The Complete Format Journey in Our Project

```
Phase 1:                     Phase 2:                    Phase 3:
[PyTorch Training]           [RKNN Conversion]           [C++ Runtime]

P3M-10k dataset              calibration images          1080p video
     |                            |                           |
     v                            v                           v
MODNet (FP32) ----> ONNX (FP32) ----> RKNN (INT8) ----> NPU inference
     |                  |                  |                   |
     v                  v                  v                   v
.ckpt file          .onnx file         .rknn file         alpha mask
(25.9 MB)           (25.9 MB)          (~2-3 MB)          (real-time)
(GPU only)          (universal)        (RK3588 only)      (30 FPS)
```

---

## 8. Vocabulary Summary

| Term                | Definition                                              |
|---------------------|----------------------------------------------------------|
| **ONNX**            | Open Neural Network Exchange (universal model format)    |
| **RKNN**            | Rockchip Neural Network (NPU-specific compiled model)    |
| **TensorRT**        | NVIDIA's inference optimizer (GPU-specific)               |
| **Operator (Op)**   | A single computation (Conv, BN, ReLU, etc.)              |
| **Operator Fusion** | Combining multiple ops into one (Conv+BN+ReLU -> 1 op)  |
| **Opset Version**   | ONNX operator compatibility level                        |
| **Static Shape**    | Fixed input dimensions (required for NPU)                |
| **Dynamic Shape**   | Variable input dimensions (development only)             |
| **Calibration**     | Finding INT8 ranges using representative images          |
| **Netron**          | Visual tool for inspecting ONNX model graphs             |
| **Letterbox**       | Resize with padding to maintain aspect ratio             |
| **Graph**           | The complete sequence of operations in a model           |
| **Compilation**     | Converting ONNX graph to hardware-specific instructions  |

---

## Self-Check Questions

1. Why can't you run a .ckpt file directly on RK3588 NPU?
2. What does Conv+BN fusion do, and why can't InstanceNorm be fused?
3. Why does RKNN require static input shapes?
4. What tool would you use to visually check an ONNX model for bad operators?
5. In our project, what's the conversion chain from training to deployment?

(Answers: 1=NPU speaks RKNN, not PyTorch; 2=Merge BN math into Conv weights
at compile time, IN needs runtime stats; 3=NPU optimizes memory layout for
fixed sizes; 4=Netron; 5=.ckpt -> .onnx -> .rknn)
