# KB-002: Backbone, ImageNet, MobileNetV2 — The AI Assembly Line

> **Target Audience**: Edge deployment engineers with no ML paper background
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `backbone`, `imagenet`, `mobilenetv2`, `pretrained`, `transfer-learning`

---

## 1. The Factory Assembly Line Analogy

Think of MODNet as a **car factory** with a clear assembly line:

```
Raw Material (Photo)
       |
       v
+==========================================+
|  STAGE 1: BACKBONE (MobileNetV2)         |  <-- The FOUNDATION
|  "The Steel Mill"                        |
|                                          |
|  Input: Raw RGB pixels (3, H, W)        |
|  Output: "Feature Maps" — abstract       |
|          representations of edges,       |
|          textures, shapes, colors        |
|                                          |
|  Think of it as: converting raw ore      |
|  into standardized steel sheets          |
+==========================================+
       |
       v
+==========================================+
|  STAGE 2: TASK HEADS (LR/HR/Fusion)     |  <-- The SPECIALIST
|  "The Car Assembly Line"                 |
|                                          |
|  Input: Feature maps from backbone       |
|  Output: Alpha matte (the mask)          |
|                                          |
|  Think of it as: bending steel sheets    |
|  into car doors, hoods, frames           |
+==========================================+
       |
       v
  Final Product (Alpha Matte)
```

---

## 2. What is a Backbone?

**Backbone = The shared feature extractor at the base of a neural network.**

### The Eyes Analogy

Your eyes (backbone) can see edges, colors, textures, and shapes.
These are UNIVERSAL visual skills — they work whether you're:
- Identifying a cat (image classification)
- Finding a person's silhouette (matting)
- Detecting a car (object detection)
- Reading text (OCR)

The backbone is the "eyes" of the AI. It extracts universal visual features
that ANY downstream task can use.

### Why "Backbone"?

Just like your spine (backbone) supports your entire body, the backbone
network supports all the task-specific "heads" built on top of it.

```
MODNet Architecture:
                                    +-- LR Branch (semantic: "is there a person?")
                                    |
Raw Image --> [MobileNetV2 Backbone] +-- HR Branch (detail: "where are the edges?")
                                    |
                                    +-- Fusion Branch (combine both -> final mask)
```

---

## 3. What is MobileNetV2?

**MobileNetV2 = A specific backbone architecture designed for MOBILE/EDGE devices.**

### The Engine Analogy

If a backbone is "the eyes", then MobileNetV2 is a SPECIFIC TYPE of eye:

| Backbone       | Analogy          | Parameters | Speed    | Accuracy |
|----------------|------------------|------------|----------|----------|
| ResNet-152     | Eagle eye        | 60M        | Slow     | Highest  |
| ResNet-50      | Human eye        | 25M        | Medium   | High     |
| **MobileNetV2**| **Hawk eye**     | **3.4M**   | **Fast** | **Good** |
| MobileNetV3-S  | Cat eye          | 1.5M       | Fastest  | Lower    |

MobileNetV2 was specifically designed by Google in 2018 for phones and
edge devices. Its key innovations:

1. **Depthwise Separable Convolutions**: Instead of one heavy 3x3 conv,
   split into a cheap 3x3 per-channel conv + a cheap 1x1 cross-channel conv.
   Result: ~9x fewer operations for similar accuracy.

2. **Inverted Residual Blocks**: Expand channels -> heavy processing ->
   compress channels back. Like breathing: inhale (expand) -> process ->
   exhale (compress).

3. **Linear Bottlenecks**: Skip the ReLU at the narrow bottleneck to
   prevent information loss.

### Why MODNet Chose MobileNetV2

MODNet's original paper targets real-time portrait matting.
MobileNetV2 gives the best speed/quality tradeoff for this task.
**This is also why it's ideal for RK3588 NPU deployment.**

---

## 4. What is ImageNet?

**ImageNet = The world's largest labeled photo database (14 million images, 1000 categories).**

### The School Analogy

Imagine a school (ImageNet) that teaches visual skills:
- Lesson 1: "This is a cat" (1.3 million photos of cats)
- Lesson 2: "This is a dog" (1.3 million photos of dogs)
- ...
- Lesson 1000: "This is a volcano"

A student (MobileNetV2) who graduated from this school has learned
UNIVERSAL visual skills:
- Layer 1-3: "I can see edges and corners"
- Layer 4-8: "I can see textures and patterns"
- Layer 9-16: "I can see object parts (ears, wheels, windows)"
- Layer 17+: "I can see whole objects"

**These skills transfer to ANY vision task**, including portrait matting.

---

## 5. What is "Pretrained" (Transfer Learning)?

**backbone_pretrained=True means: Start with ImageNet-graduated eyes,
then only retrain the task-specific heads.**

**backbone_pretrained=False means: Start with BLIND eyes (random noise),
train EVERYTHING from scratch.**

### The Job Training Analogy

```
Scenario A: backbone_pretrained=True (RECOMMENDED)
=====================================================
Hire an experienced photographer (ImageNet-trained MobileNetV2).
They already know: lighting, composition, depth, edges, textures.
You only need to teach them: "here's what a portrait mask looks like."
Training time: 2-3 hours. Quality: HIGH.

Scenario B: backbone_pretrained=False (current code!)
=====================================================
Hire a newborn baby (random weights).
They know NOTHING about the visual world.
You must teach them: what an edge is, what color is, what a person
looks like, AND what a portrait mask is.
Training time: 10-20 hours. Quality: MAY NEVER CONVERGE.
```

### The Critical Discovery in Your Code

```python
# Your current code:
modnet = MODNet(backbone_pretrained=False)  # <-- BLIND baby!

# What it should be:
modnet = MODNet(backbone_pretrained=True)   # <-- Experienced photographer!
```

**Why is this safe?** Because we only modified the HEADS (IBNorm -> BN),
NOT the backbone (MobileNetV2). The backbone's weight format is unchanged.
ImageNet weights load perfectly into the unmodified MobileNetV2 backbone.

### Impact Comparison

| Setting                  | Training Cost | Final Quality | Convergence |
|--------------------------|---------------|---------------|-------------|
| `pretrained=False`       | 20+ hours     | Uncertain     | May fail    |
| **`pretrained=True`**    | **2-3 hours** | **High**      | **Reliable**|

---

## 6. Putting It All Together: What Happens During Training

```
Step 1: Load MobileNetV2 with ImageNet weights (pretrained=True)
        The "eyes" already know how to see edges, textures, shapes.

Step 2: Attach MODNet heads (LR/HR/Fusion) with RANDOM weights
        The "hands" don't know how to make masks yet.

Step 3: Feed P3M-10k training images
        The eyes extract features, the hands try to make masks.
        Hands are bad at first (high Loss), but improve quickly
        because the eyes are already providing excellent features.

Step 4: After 15 epochs
        The hands have learned to convert visual features into
        accurate alpha mattes. Done!
```

If `pretrained=False`:
```
Step 1: Load MobileNetV2 with RANDOM weights
        The "eyes" are blind — they see random noise.

Step 2-3: Same as above, but now BOTH eyes AND hands must learn.
          The hands can't learn well because the eyes are feeding
          garbage features. Much slower, much harder to converge.
```

---

## 7. How This Connects to Your RK3588 Deployment

```
Training (GPU Server)              Deployment (RK3588)
=====================              ====================
MobileNetV2 Backbone               Same backbone
  + BN Heads                         (frozen, no learning)
  + ImageNet init
  + P3M-10k training               Exported to RKNN
       |                                |
       v                                v
  .ckpt file --> ONNX --> RKNN --> NPU inference
```

The backbone choice (MobileNetV2) directly determines:
- **Model size**: ~6.5MB ONNX → ~2MB INT8 RKNN
- **NPU compatibility**: All ops are standard Conv+BN+ReLU → 100% NPU
- **Inference speed**: Depends on input resolution (see KB-001)

---

## 8. Vocabulary Cheat Sheet

| Term              | Simple Definition                                        |
|-------------------|----------------------------------------------------------|
| **Backbone**      | The shared "eyes" of the network (feature extractor)     |
| **Head**          | Task-specific "hands" built on top of backbone           |
| **MobileNetV2**   | A lightweight backbone designed for phones/edge devices   |
| **ImageNet**      | A massive photo database used to pre-train backbones      |
| **Pretrained**    | Using weights from a backbone that already learned vision |
| **Transfer Learning** | Reusing pretrained knowledge for a new task          |
| **Fine-tuning**   | Lightly retraining a pretrained model on your data       |
| **From scratch**  | Training with random initialization (no prior knowledge) |
| **Feature Map**   | The intermediate "understanding" the backbone produces    |
| **Convergence**   | When Loss stops decreasing — training is done            |

---

## Self-Check Questions

1. If someone says "input shape 1x3x256x256", what does each number mean?
2. Is a 3x3 convolution kernel the same as 256x256 input resolution?
3. Why does hair disappear at 256x256 input but not at 512x512?
4. What's the difference between backbone_pretrained=True and False?
5. Can you use ResNet-152 as MODNet's backbone on RK3588? Why or why not?

(Answers: 1=N,C,H,W; 2=No, kernel=tiny filter, input=photo size;
3=Hair is sub-pixel at 256; 4=Experienced vs blind; 5=Too slow, too big)
