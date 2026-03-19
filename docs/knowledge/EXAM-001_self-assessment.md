# Edge AI Knowledge Exam — Self-Assessment Test v1

> **Based on**: KB-001 through KB-009
> **Difficulty**: Beginner → Intermediate
> **Time**: 30-45 minutes (no time pressure, open-book is OK for learning)
> **Scoring**: Each question 2 points. Total 60 points.
>   - 50+ : Ready for Phase 2 (quantization) concepts
>   - 35-49: Re-read the KB docs marked as weak areas
>   - <35 : Go through Tier 1 KBs again before proceeding

---

## Section A: Multiple Choice (2 points each, 10 questions)

### Q1. NCHW Format
In a tensor with shape `(1, 3, 256, 256)`, what does the "3" represent?

- A) Batch size
- B) Number of RGB color channels
- C) Image height in pixels
- D) Number of convolution kernels

### Q2. Input Shape vs Convolution Kernel
You set `input_size=256`. What happens to a 1920x1080 photo?

- A) The 3x3 convolution kernel scans the photo in 256x256 windows
- B) The entire photo is resized/shrunk to 256x256 before the AI sees it
- C) Only a 256x256 crop from the center is used
- D) The photo is split into 256x256 tiles and processed one by one

### Q3. Backbone
In MODNet, what role does MobileNetV2 play?

- A) It generates the final alpha matte output
- B) It is the shared feature extractor (the "eyes") that feeds LR/HR/Fusion branches
- C) It handles the ONNX to RKNN conversion
- D) It is the loss function used during training

### Q4. Pretrained Weights
`backbone_pretrained=True` means:

- A) The entire MODNet model is loaded from a previous training run
- B) Only the MobileNetV2 backbone loads ImageNet-trained weights; heads are random
- C) The model is frozen and cannot be trained further
- D) The model is quantized to INT8

### Q5. Batch Size and BatchNorm
Why is `batch_size >= 4` important when using BatchNorm?

- A) GPU memory requires minimum 4 images
- B) BatchNorm computes mean/variance per batch; too small = noisy statistics
- C) The convolution kernel needs 4 input channels
- D) PyTorch crashes with batch_size < 4

### Q6. Learning Rate Schedule
With `StepLR(step_size=5, gamma=0.1)` and initial `lr=0.01`, what is the LR at epoch 7?

- A) 0.01
- B) 0.001
- C) 0.0001
- D) 0.1

### Q7. Overfitting Detection
Train Loss = 0.02, Validation Loss = 0.15. What's happening?

- A) Underfitting — model hasn't learned enough
- B) Perfect training — both losses are low
- C) Overfitting — model memorized training data, fails on new data
- D) The model has converged successfully

### Q8. Quantization Overflow
Why did we replace InstanceNorm with BatchNorm for RK3588 deployment?

- A) InstanceNorm is slower because it uses more memory
- B) InstanceNorm computes x^2 at runtime, which overflows INT8 (max 127)
- C) BatchNorm produces better image quality
- D) InstanceNorm is not supported in Python 3.8

### Q9. ONNX Purpose
What is ONNX in our conversion pipeline?

- A) The final format that runs on RK3588 NPU
- B) A universal intermediate format between PyTorch and RKNN
- C) A training framework like PyTorch
- D) A data augmentation library

### Q10. Conv+BN Fusion
Why does the RKNN compiler fuse Conv2d + BatchNorm into a single operation?

- A) It reduces the number of Python lines
- B) BN statistics (mean/var) are fixed after training, so BN math can be folded
   into Conv weights at compile time — BN vanishes, zero runtime cost
- C) It makes the model more accurate
- D) It increases the model file size for better quality

---

## Section B: Short Answer (2 points each, 10 questions)

### Q11.
In one sentence: what is the difference between "forward pass" and "backward pass"?

### Q12.
You have 9,421 images and batch_size=8. How many iterations (batches) per epoch?
Show your calculation.

### Q13.
Your training shows:
```
Epoch 1: Val Loss = 0.0264
Epoch 5: Val Loss = 0.0129
```
Is this a good or bad sign? Why?

### Q14.
What is "knowledge distillation" in one sentence? When would we use it
in our project (which Block)?

### Q15.
Name the 3 model compression techniques and rank them by typical
speed improvement (highest first).

### Q16.
What does `optimizer.zero_grad()` do, and why is it called before
each training iteration?

### Q17.
Why do we train at 512x512 but plan to deploy at 256x256 on the NPU?
(Hint: think about Phase 3)

### Q18.
What is "calibration" in INT8 quantization? Why do we need ~200 images for it?

### Q19.
The conversion chain for our model is: `.ckpt` -> `???` -> `.rknn`.
What goes in the middle, and what tool creates each conversion?

### Q20.
What is the difference between "structured pruning" and "unstructured pruning"?
Which one works on RK3588 NPU?

---

## Section C: Scenario Analysis (2 points each, 5 questions)

### Q21.
You start training and see this output:
```
[Epoch 1][Batch 0] Loss: nan
[Epoch 1][Batch 1] Loss: nan
```
What is the most likely cause? What would you change?

### Q22.
After training 15 epochs, you open `epoch_15_val.png` and see:
- The predicted mask shows a clear human silhouette
- BUT there are small holes inside the body (the torso has black spots)
- Hair region is a hard staircase edge with no detail

Rate the result: SUCCESS / PARTIAL / FAILURE? What's your next step?

### Q23.
You export the trained model to ONNX and open it in Netron. You see
these operators: `Conv`, `BatchNormalization`, `Relu`, `Resize`, `Sigmoid`.
You also see: `ReduceMean`, `Pow`, `Sub`.

Is this ONNX ready for RKNN conversion? Why or why not?

### Q24.
After RKNN INT8 conversion, you run the model on RK3588 and see:
- Inference time: 18ms (great!)
- Output: completely black image (no mask visible)

What happened? What's the fallback plan (which Block)?

### Q25.
Your final pipeline runs at 28ms total on RK3588:
- Resize: 2ms
- NPU inference: 18ms
- Guided Filter: 8ms

Is this within the 30 FPS target? Show your calculation.

---

## Answer Key

### Section A
| Q | Answer | KB Reference |
|---|--------|-------------|
| 1 | B | KB-001 |
| 2 | B | KB-001 |
| 3 | B | KB-002 |
| 4 | B | KB-006 |
| 5 | B | KB-003 |
| 6 | B | KB-003 (epoch 6 is in step 2: 0.01 × 0.1 = 0.001) |
| 7 | C | KB-008 |
| 8 | B | KB-005 |
| 9 | B | KB-007 |
| 10 | B | KB-007 |

### Section B (Sample answers)
| Q | Answer |
|---|--------|
| 11 | Forward pass pushes data through the network to produce a prediction (no learning); backward pass computes gradients to update weights (learning happens). |
| 12 | 9421 / 8 = 1177.625 → 1177 iterations (drop_last=True discards the incomplete last batch). |
| 13 | Good sign — validation loss is decreasing, meaning the model is generalizing better to unseen data, not overfitting. |
| 14 | Training a small "student" model to mimic a large "teacher" model's predictions. Block 1.3 fallback if our Pure-BN model's quality is poor. |
| 15 | 1) Quantization (2-4x), 2) Pruning (1.5-3x), 3) Distillation (depends on architecture choice). |
| 16 | Clears accumulated gradients from the previous iteration. PyTorch accumulates gradients by default; without zeroing, gradients from multiple batches would add up incorrectly. |
| 17 | At 512, hair strands are ~1.4px (visible to AI, it learns edge features). At 256, they're <1px (invisible). Phase 3 Guided Filter uses the original 1080p image to refine the rough 256 mask back to hair-level detail. |
| 18 | Calibration = feeding representative images through the FP32 model to find the min/max activation range at each layer. These ranges determine the INT8 scale/zero_point. ~200 images provide enough statistical coverage of different lighting/skin/poses. |
| 19 | `.ckpt` -> `.onnx` (torch.onnx.export) -> `.rknn` (rknn-toolkit2). |
| 20 | Structured: removes entire filters/channels (physically smaller model, works on any hardware including RK3588). Unstructured: zeros individual weights (same model size, needs sparse hardware support — RK3588 NPU does NOT support it). |

### Section C (Sample answers)
| Q | Answer |
|---|--------|
| 21 | LR too high (0.01) causing gradient explosion. Fix: reduce LR to 0.001 or 0.0001. |
| 22 | PARTIAL success. Body shape is correct but holes and jagged hair indicate the model needs more training or quality isn't sufficient. Next step: check if Block 1.3 (Knowledge Distillation) improves quality. Or try more epochs with lower LR. |
| 23 | NO. ReduceMean + Pow + Sub are InstanceNorm decomposition operators. The NPU will fall back to CPU for these, killing performance. The IBNorm removal was incomplete — go back to modnet.py and fix. |
| 24 | INT8 quantization destroyed the output (precision collapse). The scale/zero_point mapping lost all useful information. Fallback: Block 2.4 — use mixed precision (set the worst layers to FP16, keep rest INT8). |
| 25 | Total = 28ms. 30 FPS requires < 33.3ms per frame (1000ms / 30). 28ms < 33.3ms → YES, within target! Actual FPS = 1000/28 ≈ 35.7 FPS. |

---

## Self-Grading Guide

After completing, grade yourself honestly:
- **50-60**: Excellent! You understand the full pipeline. Ready for hands-on Phase 2.
- **35-49**: Good foundation. Re-read the KBs for questions you got wrong.
- **20-34**: Need more study. Focus on Tier 1 KBs (001-004) first.
- **<20**: Start from KB-001 and work through one KB per day.

> **Tip**: Take this exam again in 1 week. If you score 50+ without
> looking at the answer key, the knowledge is solidified.
