# KB-005: Knowledge Distillation, Pruning, Quantization — The Model Compression Trinity

> **Target Audience**: Edge deployment engineers targeting RKNN/NPU
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `advanced`, `distillation`, `pruning`, `quantization`, `edge-deploy`, `compression`

---

## 0. Why Does This Matter for Edge Deployment?

Your RK3588 has 6 TOPS of NPU compute and ~4GB usable RAM.
A server-grade model (ResNet-152) needs 60M parameters × 4 bytes = 240MB
just for weights, and 100+ GFLOPS per image.

**You need to SHRINK models to fit on edge devices.** The three main
techniques are: Quantization, Pruning, and Knowledge Distillation.

```
                    Server (A100)              Edge (RK3588)
RAM:                80 GB                      4 GB
Compute:            312 TFLOPS (FP16)          6 TOPS (INT8)
Power:              300W                       5W
Model Budget:       Unlimited                  < 10MB, < 25ms
```

---

## 1. Quantization — "Shrinking the Ruler"

### What Is It?

Converting model weights and activations from high-precision numbers
(FP32 = 32-bit floating point) to low-precision numbers (INT8 = 8-bit
integer). This is our Phase 2 in the master plan.

### The Ruler Analogy

```
FP32 Ruler: Marks every 0.0001mm (extremely precise, large ruler)
            Range: ±3.4 × 10^38
            Size per number: 4 bytes

FP16 Ruler: Marks every 0.01mm (good precision, medium ruler)
            Range: ±65,504
            Size per number: 2 bytes

INT8 Ruler: Marks every 1mm only (coarse, tiny ruler)
            Range: -128 to +127  (only 256 possible values!)
            Size per number: 1 byte
```

### The Compression Math

```
Model: 6,487,795 parameters

FP32: 6,487,795 × 4 bytes = 25.9 MB
FP16: 6,487,795 × 2 bytes = 13.0 MB
INT8: 6,487,795 × 1 byte  =  6.5 MB  (4x smaller than FP32!)
```

### How INT8 Quantization Works

**Step 1: Find the range of values in each layer**
```
Layer 7 weights range: [-0.34, +0.51]
Map this to INT8 range: [-128, +127]

Scale = (0.51 - (-0.34)) / 255 = 0.00333
Zero_point = round(-(-0.34) / 0.00333) = 102

To quantize: int8_value = round(fp32_value / scale) + zero_point
To dequantize: fp32_value = (int8_value - zero_point) × scale
```

**Step 2: Calibration (what those 200 images are for)**
```
Feed 200 representative images through the FP32 model.
Record the min/max values at every layer.
Use these ranges to compute optimal scale/zero_point per layer.
```

### Why INT8 Matters for RK3588 NPU

```
RK3588 NPU specs:
  - INT8: 6 TOPS (6 trillion operations per second)
  - FP16: 3 TOPS (half the speed!)
  - FP32: NOT SUPPORTED on NPU

Using INT8 gives you 2x the compute of FP16.
This is the difference between 20ms and 40ms per frame.
```

### The Overflow Problem (Why We Removed InstanceNorm)

```
InstanceNorm computes variance: var = mean(x^2) - mean(x)^2

In INT8, if x = 12:
  x^2 = 144  --> EXCEEDS INT8 MAX (127)!  --> OVERFLOW!
  Result: garbage data, model output = noise

BatchNorm avoids this because the compiler FUSES Conv+BN into a
single operation at compile time. The BN math disappears entirely.
No x^2 computation at runtime = no overflow risk.
```

### Types of Quantization

| Type                | When Applied        | Quality | Complexity |
|---------------------|---------------------|---------|------------|
| **Post-Training (PTQ)** | After training, no retraining | Good | Low |
| **Quantization-Aware (QAT)** | During training, simulates INT8 | Better | High |
| **Mixed Precision**  | Some layers INT8, some FP16 | Best | Medium |

Our Phase 2 uses **PTQ** first. If quality drops, we fallback to
**Mixed Precision** (Block 2.4).

---

## 2. Knowledge Distillation — "Teaching a Student"

### What Is It?

Training a SMALL model (student) to mimic a LARGE model (teacher),
instead of learning directly from the ground truth labels.

### The School Analogy

```
Traditional Training:
  Student reads textbook (ground truth labels) -> takes exam
  Problem: The textbook only has right/wrong answers.

Knowledge Distillation:
  Teacher (large model) explains WHY each answer is correct.
  Student (small model) learns from the teacher's explanations.
  The teacher's "soft predictions" contain MUCH more information
  than the hard labels.
```

### Why Soft Predictions Are More Informative

```
Ground Truth Label (hard):
  "This pixel is foreground" = 1.0
  No nuance. No "almost foreground" or "probably background".

Teacher's Prediction (soft):
  "This pixel is 0.87 foreground"
  Contains subtle information:
  - 0.87 means "very likely foreground but with some uncertainty"
  - A nearby pixel at 0.43 means "roughly 50/50, probably an edge"
  - This "dark knowledge" teaches the student about edges, gradients,
    and uncertainty — things the hard label cannot express.
```

### When Do We Use Distillation? (Block 1.3 Fallback)

```
Scenario: Our Pure-BN model trained fine, but quality is noticeably
worse than the original IBNorm model.

Solution:
  Teacher = Original MODNet (with IBNorm, FP32, high quality)
  Student = Our Pure-BN MODNet (the one we want to deploy)

  New Loss = 0.7 × L1(student_output, ground_truth)     -- learn from data
           + 0.3 × L2(student_output, teacher_output)   -- mimic the teacher

  The teacher guides the student to produce better outputs
  than it could learn from raw labels alone.
```

### Distillation in the Real World

| Model | Teacher | Student | Use Case |
|-------|---------|---------|----------|
| DistilBERT | BERT (340M) | DistilBERT (66M) | NLP, 40% smaller, 97% quality |
| TinyBERT | BERT | TinyBERT (15M) | Mobile NLP |
| Our case | MODNet-IBNorm | MODNet-PureBN | Edge matting |

---

## 3. Pruning — "Cutting Dead Branches"

### What Is It?

Removing weights (connections) from the network that contribute little
to the output. Like trimming a tree — remove dead branches to make it
lighter and healthier.

### Types of Pruning

**Unstructured Pruning** (fine-grained):
```
Before: weight matrix = [0.5, 0.001, -0.3, 0.002, 0.7]
After:  weight matrix = [0.5, 0.0,   -0.3, 0.0,   0.7]
                               ^^^^         ^^^^
                               set to zero (pruned)

Problem: The matrix is still the same SIZE in memory.
         You need special sparse hardware to get speedup.
         RK3588 NPU does NOT support sparse operations.
```

**Structured Pruning** (coarse-grained):
```
Before: 64 convolution filters
After:  48 convolution filters (removed 16 least-important ones)

The model is PHYSICALLY smaller. Fewer FLOPs, less memory.
Works on ANY hardware including RK3588 NPU.

But: requires retraining to recover accuracy.
```

### Pruning in Practice

```
Step 1: Train the full model normally
Step 2: Analyze which filters/channels have the smallest "importance"
        (measured by weight magnitude, gradient, or activation)
Step 3: Remove the least important N% of filters
Step 4: Retrain (fine-tune) the pruned model to recover accuracy
Step 5: Repeat steps 2-4 (iterative pruning)
```

### When to Use Pruning

| Scenario | Use Pruning? | Why |
|----------|-------------|-----|
| Model is 10x too big for target | Yes | Significant size reduction needed |
| Model fits but is 20% too slow | Maybe | Try quantization first |
| **MODNet on RK3588** | **Not now** | **Already small (6.5M params), quantization is enough** |

We're NOT using pruning in our current plan because MODNet with
MobileNetV2 backbone is already a lightweight architecture. INT8
quantization alone should give us the speed we need.

---

## 4. The Compression Pipeline (How They Work Together)

```
Original Model (FP32, 25.9 MB, 120ms on RK3588)
    |
    v
[Optional: Knowledge Distillation]
  Train a smaller student model
    |
    v
[Optional: Structured Pruning]
  Remove redundant channels/layers
    |
    v
[Required: INT8 Quantization]    <-- This is always the final step
  Convert remaining weights to INT8
    |
    v
Deployed Model (INT8, ~2-3 MB, ~20ms on RK3588)
```

### Our Plan's Path

```
MODNet (FP32, IBNorm)
    |
    v [Block 1.1: Architecture Surgery]
MODNet (FP32, Pure BN)
    |
    v [Block 1.2: Retrain]
MODNet (FP32, Pure BN, trained)
    |
    v [Block 1.4: ONNX Export]
MODNet.onnx (FP32)
    |
    v [Block 2.2: RKNN Quantization]
MODNet.rknn (INT8, ~2-3 MB, ~20ms)
    |
    v [Block 3.1: Guided Filter]
Full Pipeline (NPU + GPU, ~30ms, 1080p hair-level)
```

---

## 5. Comparison Table: The Trinity

| Technique       | What It Does              | Speed Gain | Quality Loss | Complexity |
|-----------------|---------------------------|------------|--------------|------------|
| **Quantization** | Reduce precision (FP32->INT8) | 2-4x   | Small (1-5%) | Low        |
| **Pruning**      | Remove redundant weights   | 1.5-3x    | Medium (3-10%) | Medium   |
| **Distillation** | Train small model from big | Depends    | Small (2-5%) | Medium     |

### When to Use What

```
Need 2x speedup?      --> Quantization alone
Need 4x speedup?      --> Quantization + Pruning
Need 10x speedup?     --> Distillation (smaller architecture) + Quantization
Quality dropped?       --> Distillation (use original as teacher)
```

---

## 6. Advanced: Model Security (Bonus Knowledge)

### Model Poisoning (Data Poisoning Attack)

```
Attacker injects malicious samples into training data.
Example: 0.1% of "cat" images are secretly labeled "dog".
Result: The model learns a backdoor — certain trigger patterns
cause it to misclassify on purpose.

Defense: Data validation, anomaly detection in training metrics.
```

### Model Extraction Attack

```
Attacker sends thousands of queries to your deployed API.
Uses the input/output pairs to train their own clone model.
Effectively "steals" your model's knowledge.

Defense: Rate limiting, output perturbation, watermarking.
```

### Adversarial Examples

```
Adding imperceptible noise to an image fools the AI:
  Normal image:  "This is a person" (correct)
  + tiny noise:  "This is a dog" (wrong!)

The noise is invisible to humans but catastrophic for AI.

Defense: Adversarial training, input preprocessing.
```

### Weight Watermarking

```
Embed a secret signature in the model's weights.
If someone steals and redeploys your model, you can prove
ownership by extracting the watermark.
```

### For RKNN Deployment Specifically

```
Risk: Someone extracts your .rknn model from the device.
      They can't easily retrain it (INT8, hardware-specific),
      but they could deploy it on similar hardware.

Defense:
  - Encrypt the .rknn file at rest
  - Decrypt only in memory during loading
  - Use RK3588's secure boot chain if available
  - Obfuscate the model structure (rename layers)
```

---

## 7. Vocabulary Summary

| Term                     | Definition                                                |
|--------------------------|-----------------------------------------------------------|
| **Quantization**         | Reduce numerical precision (FP32 -> INT8)                 |
| **INT8**                 | 8-bit integer (-128 to 127)                               |
| **FP32 / FP16**          | 32/16-bit floating point                                  |
| **Calibration**          | Finding optimal INT8 ranges using representative data     |
| **PTQ**                  | Post-Training Quantization (quantize after training)      |
| **QAT**                  | Quantization-Aware Training (simulate INT8 during train)  |
| **Mixed Precision**      | Some layers INT8, critical layers FP16                    |
| **Knowledge Distillation** | Small model learns from large model's predictions       |
| **Teacher / Student**    | Large/small model in distillation                         |
| **Soft Labels**          | Teacher's probabilistic predictions (vs hard 0/1 labels)  |
| **Dark Knowledge**       | Subtle information in soft predictions (edge uncertainty)  |
| **Pruning**              | Removing unimportant weights/channels                     |
| **Structured Pruning**   | Remove entire filters (physically smaller model)          |
| **Unstructured Pruning** | Zero out individual weights (needs sparse hardware)       |
| **Model Poisoning**      | Attacker corrupts training data to add backdoors          |
| **Adversarial Examples** | Inputs with imperceptible noise that fool the model       |
| **Model Extraction**     | Cloning a model by querying its API                       |

---

## Self-Check Questions

1. Why does INT8 give 2x speed over FP16 on RK3588 NPU?
2. What is "calibration" in quantization and why do we need 200 images?
3. In knowledge distillation, why are "soft labels" better than "hard labels"?
4. Why don't we use pruning for MODNet?
5. If a layer has x^2 operations, why does INT8 overflow?

(Answers: 1=NPU has 6 TOPS INT8 vs 3 TOPS FP16; 2=Find min/max per layer
to compute scale/zero_point; 3=Soft labels encode uncertainty and edge info;
4=MODNet is already small (6.5M), quantization is enough; 5=12^2=144>127)
