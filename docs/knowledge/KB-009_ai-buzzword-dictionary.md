# KB-009: The AI Buzzword Dictionary — Reading AI News Like a Pro

> **Target Audience**: Engineers who want to understand AI news, papers, and discussions
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `reference`, `vocabulary`, `buzzwords`, `industry`, `news`

---

## How to Use This Document

When you read an AI article and encounter an unfamiliar term, Ctrl+F here.
Terms are organized by category. Each has:
- **One-line definition** (what it IS)
- **Analogy** (how to think about it)
- **Relevance** (does it matter for your RKNN edge deployment work?)

---

## Category 1: Model Architecture Terms

### Transformer
**What**: A neural network architecture based on "self-attention" that
processes all input positions simultaneously (not sequentially).
**Analogy**: Reading an entire page at once vs reading word-by-word.
**Examples**: GPT-4, BERT, Vision Transformer (ViT)
**Relevance**: Some vision models use ViT as backbone. Too heavy for RK3588
currently, but future NPUs may support them. MobileNetV2 (CNN) is better for now.

### CNN (Convolutional Neural Network)
**What**: A network that uses convolution operations to detect spatial patterns.
**Analogy**: Sliding a magnifying glass over an image to find patterns.
**Examples**: ResNet, MobileNetV2, VGG — and MODNet's backbone.
**Relevance**: ★★★★★ This is what runs on your NPU. CNNs are the bread and
butter of edge vision AI.

### Attention / Self-Attention
**What**: A mechanism that lets the model decide which parts of the input
are most relevant for producing each part of the output.
**Analogy**: When reading, your eyes jump to the most important words.
**Relevance**: ★★☆☆☆ Not in our MODNet, but you'll see it everywhere in AI news.

### Encoder-Decoder
**What**: Encoder compresses input into a compact representation; decoder
expands it back to the desired output shape.
**Analogy**: Encoder = summarize a book; Decoder = expand the summary back.
**Relevance**: ★★★★☆ MODNet uses an encoder (MobileNetV2) and three decoder
branches (LR/HR/Fusion). Many segmentation models follow this pattern.

### U-Net
**What**: A specific encoder-decoder with "skip connections" that copy features
from encoder to decoder at matching resolutions.
**Analogy**: Like leaving bookmarks while summarizing, then flipping back to
those pages while expanding.
**Relevance**: ★★★☆☆ MODNet's HR branch has similar skip-connection ideas.

### GAN (Generative Adversarial Network)
**What**: Two networks fight each other: Generator creates fake images,
Discriminator tries to detect fakes. Both improve through competition.
**Analogy**: A forger vs a detective, each getting better over time.
**Relevance**: ★☆☆☆☆ Not used in matting, but powers image generation (Stable Diffusion).

### Diffusion Model
**What**: Learns to gradually remove noise from random static until a coherent
image appears. The reverse of gradually adding noise to an image.
**Analogy**: Starting with TV static and slowly revealing a clear picture.
**Examples**: Stable Diffusion, DALL-E, Midjourney
**Relevance**: ★☆☆☆☆ Too slow for real-time on edge devices (needs 20+ iterations).

---

## Category 2: Training Terms

### Epoch
See KB-003. One complete pass through all training data.

### Loss Function
See KB-004. Mathematical formula measuring prediction error.

### Gradient Descent
See KB-004. Weight update algorithm: move opposite to the gradient.

### Learning Rate Schedule
See KB-003. Strategy for changing LR during training (StepLR, CosineAnnealing).

### Hyperparameter
**What**: Settings you choose BEFORE training starts (not learned by the model).
**Examples**: Learning rate, batch size, number of epochs, model architecture.
**Analogy**: Cooking temperature and time — you set them, the oven doesn't learn them.
**Relevance**: ★★★★★ Choosing the right hyperparameters is 50% of the job.

### Convergence
**What**: When the loss stops decreasing meaningfully. Training is "done."
**Analogy**: A ball reaching the bottom of a valley and stopping.
**Relevance**: ★★★★☆ You need to recognize when to stop training.

### Warm-up
**What**: Starting with a very small LR and gradually increasing it over
the first few epochs/iterations. Prevents training from exploding early.
**Analogy**: Warming up before running — start slow, then accelerate.
**Relevance**: ★★☆☆☆ Not in our current training, but common in larger models.

### Mixed Precision Training
**What**: Using FP16 for most operations and FP32 only where needed.
Doubles training speed on modern GPUs with tensor cores.
**Analogy**: Using rough tools for bulk work, precise tools only for detail.
**Relevance**: ★★★☆☆ Your RTX 3090 supports this. Could halve training time.

---

## Category 3: Evaluation Metrics

### L1 Loss / MAE (Mean Absolute Error)
**What**: Average of |prediction - truth| across all pixels.
**Relevance**: ★★★★★ This is what our validation measures.

### MSE / L2 Loss
**What**: Average of (prediction - truth)^2. Penalizes large errors more.
**Relevance**: ★★★★☆ Used in our semantic loss.

### IoU (Intersection over Union)
**What**: Overlap area / Total area between predicted mask and ground truth.
```
IoU = (Predicted ∩ Truth) / (Predicted ∪ Truth)
Range: 0.0 (no overlap) to 1.0 (perfect match)
```
**Relevance**: ★★★☆☆ Common metric for segmentation quality.

### mAP (mean Average Precision)
**What**: Average precision across multiple IoU thresholds.
**Relevance**: ★★☆☆☆ More for object detection (YOLO, etc.), not matting.

### FPS (Frames Per Second)
**What**: How many images the model processes per second.
**Relevance**: ★★★★★ Your target: 30 FPS on RK3588. FPS = 1000 / inference_ms.

### TOPS (Tera Operations Per Second)
**What**: Measure of hardware compute capacity. 1 TOPS = 10^12 operations/sec.
**Relevance**: ★★★★★ RK3588 has 6 TOPS INT8. This is your performance ceiling.

### FLOPS (Floating-point Operations Per Second)
**What**: Measure of model computational cost.
**Relevance**: ★★★★☆ Tells you if a model can theoretically fit in your NPU budget.

---

## Category 4: Model Compression (See KB-005 for details)

### Quantization: FP32 -> INT8/FP16 precision reduction
### Pruning: Remove unnecessary weights/channels
### Knowledge Distillation: Small model learns from large model
### ONNX: Universal model format (see KB-007)
### TensorRT/RKNN: Hardware-specific optimized format (see KB-007)

---

## Category 5: Deployment & Systems Terms

### Inference
**What**: Running a trained model on new data to get predictions.
NOT training. No learning happens. Just prediction.
**Analogy**: Using the knowledge from studying to answer real exam questions.
**Relevance**: ★★★★★ This is what your RK3588 does.

### Latency
**What**: Time for one inference (input -> output). Measured in milliseconds.
**Relevance**: ★★★★★ Your target: < 33ms (for 30 FPS).

### Throughput
**What**: Number of inferences per unit time. Related to but different from latency.
**Relevance**: ★★★★☆ Throughput can be higher than 1/latency with pipelining.

### Zero-Copy
**What**: Passing data between CPU and NPU without copying it through memory.
The NPU reads directly from the CPU's memory buffer.
**Analogy**: Instead of photocopying a document and mailing it, you just
point to the original on the shared desk.
**Relevance**: ★★★★★ Your C++ code uses this! (rknn-zero-copy.cpp)

### DMA (Direct Memory Access)
**What**: Hardware mechanism for transferring data without CPU involvement.
**Relevance**: ★★★★☆ Underlies zero-copy on RK3588.

### NPU (Neural Processing Unit)
**What**: Specialized chip designed for matrix multiplication (the core
operation in neural networks). Much faster and more power-efficient than
CPU/GPU for this specific task.
**Relevance**: ★★★★★ This is your target hardware.

### RGA (Rockchip Graphics Accelerator)
**What**: Hardware block on RK3588 for image resizing, rotation, color
conversion. Offloads these tasks from CPU.
**Relevance**: ★★★★☆ Can accelerate your preprocessing pipeline.

---

## Category 6: Cutting-Edge Buzzwords (2024-2026)

### LLM (Large Language Model)
**What**: Massive transformer models trained on text (GPT-4, Claude, LLaMA).
**Relevance**: ★☆☆☆☆ For text, not vision. But multimodal LLMs can process images too.

### RAG (Retrieval-Augmented Generation)
**What**: LLM + search engine. Retrieves relevant documents before generating answers.
**Relevance**: ★☆☆☆☆ Not relevant to edge vision.

### LoRA (Low-Rank Adaptation)
**What**: Efficient fine-tuning technique that adds small trainable matrices
to a frozen model. Much cheaper than full fine-tuning.
**Analogy**: Adding sticky notes to a textbook instead of rewriting it.
**Relevance**: ★★☆☆☆ Could be useful for adapting matting to specific domains.

### Edge AI
**What**: Running AI models directly on the device (phone, camera, robot)
rather than in the cloud. Your ENTIRE project is Edge AI.
**Relevance**: ★★★★★ This IS your career focus.

### MLOps
**What**: DevOps practices applied to ML: automated training, testing,
deployment, monitoring of models.
**Relevance**: ★★★☆☆ Important as you scale from prototype to production.

### Model Serving
**What**: Infrastructure for running models in production (APIs, batching,
load balancing). Examples: Triton, TFServing, FastAPI.
**Relevance**: ★★☆☆☆ More for cloud deployment. Edge uses direct C++ integration.

---

## Category 7: Model Safety & Security (See KB-005 Section 6)

### Model Poisoning: Corrupting training data to insert backdoors
### Adversarial Attack: Crafted inputs that fool the model
### Model Extraction: Cloning a model by querying its API
### Watermarking: Embedding ownership proof in model weights
### Federated Learning: Training across devices without sharing raw data

---

## Quick Reference Card (Print This!)

```
TRAINING:     epoch → batch → forward → loss → backward → optimizer.step()
COMPRESSION:  distillation → pruning → quantization (INT8)
CONVERSION:   .ckpt → .onnx → .rknn
DEPLOYMENT:   resize → NPU inference → post-process → display
METRICS:      loss (lower=better), FPS (higher=better), latency (lower=better)
HARDWARE:     NPU=AI chip, RGA=resize chip, CPU=general, GPU=graphics
```
