# KB-006: Pretrained Weights, Transfer Learning, Fine-tuning — Standing on Giants' Shoulders

> **Target Audience**: Edge deployment engineers learning ML fundamentals
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `pretrained`, `transfer-learning`, `fine-tuning`, `checkpoint`, `weights`

---

## 1. What Are "Weights" in a Neural Network?

### The Knob Analogy

A neural network is a machine with MILLIONS of adjustable knobs (weights).
Each knob controls how much a certain input feature affects the output.

```
MODNet: 6,487,795 knobs (weights)

Each knob is a floating-point number:
  weight = 0.0342  (this one says "slight positive influence")
  weight = -0.891  (this one says "strong negative influence")
  weight = 0.0001  (this one barely matters)
```

### Weight File Formats You'll Encounter

| Format          | Extension  | Framework   | What's Inside                   |
|-----------------|------------|-------------|----------------------------------|
| **Checkpoint**  | `.ckpt`    | PyTorch     | Weights + optimizer state        |
| **State Dict**  | `.pth`     | PyTorch     | Weights only (dict of tensors)   |
| **ONNX**        | `.onnx`    | Universal   | Weights + computation graph      |
| **RKNN**        | `.rknn`    | Rockchip    | Quantized weights + NPU graph    |
| **TFLite**      | `.tflite`  | TensorFlow  | Weights + mobile graph           |
| **SavedModel**  | folder     | TensorFlow  | Complete model + metadata        |

### The Conversion Pipeline You're Building

```
.ckpt (PyTorch training output)
  |
  v  [torch.onnx.export()]
.onnx (Universal interchange format)
  |
  v  [rknn-toolkit2]
.rknn (RK3588 NPU native format, INT8)
```

---

## 2. What Are "Pretrained Weights"?

**Pretrained = Someone already trained this model on a HUGE dataset,
and you're downloading their finished knob settings.**

### The University Degree Analogy

```
ImageNet Pretrained MobileNetV2:
  - Trained on 1.4 million images, 1000 categories
  - By Google Research, on 100+ GPUs for weeks
  - Cost: tens of thousands of dollars in compute
  - Result: A set of knob values that understand vision

You get all of this FOR FREE by downloading a 9MB file:
  mobilenetv2_human_seg.ckpt
```

### What the Pretrained Backbone Knows

The early layers learned UNIVERSAL visual features:

```
Layer 1-3 (low-level):
  ┌──────────────────────────┐
  │ ╱╱╱  ───  |||  ╲╲╲      │  Edges: horizontal, vertical, diagonal
  │ ═══  ···  ooo  ~~~      │  Textures: dots, stripes, waves
  └──────────────────────────┘

Layer 4-8 (mid-level):
  ┌──────────────────────────┐
  │ [eye] [ear] [wheel]     │  Object parts
  │ [fur] [fabric] [sky]    │  Material textures
  └──────────────────────────┘

Layer 9-16 (high-level):
  ┌──────────────────────────┐
  │ [face] [car] [building] │  Whole objects
  │ [person] [animal]       │  Categories
  └──────────────────────────┘
```

These features are UNIVERSAL — they apply to any vision task.
That's why pretrained backbones work across different tasks.

---

## 3. Transfer Learning — Reusing Knowledge

**Transfer Learning = Taking knowledge learned from one task (ImageNet
classification) and applying it to a different task (portrait matting).**

### Three Transfer Learning Strategies

**Strategy 1: Feature Extraction (freeze backbone)**
```
Backbone:  FROZEN (pretrained weights locked, no learning)
Heads:     TRAINABLE (randomly initialized, will learn)
Use when:  Very small dataset, risk of overfitting
Speed:     Fastest training
Quality:   Good if tasks are similar
```

**Strategy 2: Fine-tuning (train everything)** <-- What we're doing
```
Backbone:  TRAINABLE (pretrained weights, but can adjust)
Heads:     TRAINABLE (randomly initialized, will learn)
Use when:  Medium dataset (9,421 images = our case)
Speed:     Moderate training
Quality:   Best overall
```

**Strategy 3: From Scratch (no pretrained)**
```
Backbone:  TRAINABLE (randomly initialized!)
Heads:     TRAINABLE (randomly initialized)
Use when:  Massive dataset (100K+ images) AND very different domain
Speed:     Slowest (10x longer)
Quality:   Potentially best with enough data, risky with little data
```

### Our Choice: Strategy 2 (Fine-tuning)

```python
# backbone_pretrained=True + all parameters trainable
modnet = MODNet(backbone_pretrained=True)
optimizer = SGD(modnet.parameters(), lr=0.01)  # ALL params included
```

Why fine-tuning?
- 9,421 images is enough to adjust pretrained features
- Portrait matting is related to ImageNet (both involve seeing objects)
- The backbone's edge detection is 90% what we need, just needs tweaking

---

## 4. Checkpoint — Saving Your Progress

### What Is a Checkpoint?

A snapshot of the model's state at a specific moment during training.
Like a save game in a video game.

```python
# Save checkpoint
torch.save(modnet.state_dict(), 'modnet_bn_epoch_05.ckpt')

# Load checkpoint later
modnet.load_state_dict(torch.load('modnet_bn_epoch_05.ckpt'))
```

### What's Inside a Checkpoint File?

```python
# state_dict() returns a dictionary:
{
  'backbone.model.features.0.0.weight': tensor([...]),  # Conv2d weights
  'backbone.model.features.0.1.weight': tensor([...]),  # BatchNorm gamma
  'backbone.model.features.0.1.bias':   tensor([...]),  # BatchNorm beta
  'backbone.model.features.0.1.running_mean': tensor([...]),  # BN stats
  'backbone.model.features.0.1.running_var':  tensor([...]),  # BN stats
  'lr_branch.conv_lr16x.layers.0.weight': tensor([...]),
  ...
  # 6,487,795 values total across ~312 keys
}
```

### Key Checkpoint Concepts

| Concept          | Explanation                                            |
|------------------|--------------------------------------------------------|
| **state_dict**   | Dictionary of all weight tensors, keyed by layer name  |
| **running_mean** | BatchNorm's tracked average of input means (per channel) |
| **running_var**  | BatchNorm's tracked average of input variances          |
| **Best model**   | Checkpoint with lowest validation loss                  |
| **Last model**   | Most recent checkpoint (not necessarily the best)       |

### Why Save Multiple Checkpoints?

```
Epoch:  1    2    3    4    5    6    7    8    9   10
Loss:  4.0  2.1  1.5  1.2  0.9  0.7  0.6  0.55 0.7 0.8
                                      ^^^^
                                      BEST (save this!)
                                                  ^^^^
                                                  WORSE (overfitting started!)

If you only saved the last checkpoint (epoch 10), you'd have a WORSE
model than epoch 8. Always track the best checkpoint by validation loss.
```

---

## 5. The Weight Loading Dance (What Happens in Our Code)

```python
# MODNet.__init__() execution order:

# Step 1: Create backbone with random weights
self.backbone = MobileNetV2Backbone(3)
# backbone weights: [random, random, random, ...]

# Step 2: Create task heads with random weights
self.lr_branch = LRBranch(self.backbone)
self.hr_branch = HRBranch(...)
self.f_branch = FusionBranch(...)
# all weights: [random, random, random, ...]

# Step 3: Reinitialize everything with Kaiming initialization
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        self._init_conv(m)  # smarter random initialization
# all weights: [smart_random, smart_random, ...]

# Step 4: Load pretrained backbone weights (OVERWRITES step 3 for backbone)
if self.backbone_pretrained:
    self.backbone.load_pretrained_ckpt()
# backbone weights: [pretrained, pretrained, pretrained, ...]  <-- GOOD!
# head weights:     [smart_random, smart_random, ...]          <-- Will learn
```

---

## 6. Common Weight-Related Errors and Fixes

### Error: "Missing key(s) in state_dict"
```
RuntimeError: Error(s) in loading state_dict:
  Missing key(s): "lr_branch.conv_lr.layers.0.weight"
```
**Cause**: The checkpoint was saved from a DIFFERENT model architecture.
**Fix**: Make sure the model code matches exactly when saving/loading.

### Error: "Unexpected key(s) in state_dict"
```
RuntimeError: Error(s) in loading state_dict:
  Unexpected key(s): "module.backbone.model.features.0.0.weight"
```
**Cause**: Checkpoint was saved with `nn.DataParallel` (adds "module." prefix).
**Fix**: Strip the prefix:
```python
new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
```

### Error: "Size mismatch for key"
```
RuntimeError: Error(s) in loading state_dict:
  size mismatch for backbone.model.features.0.0.weight:
  copying a param with shape torch.Size([32, 3, 3, 3]) from checkpoint,
  where the shape is torch.Size([32, 1, 3, 3]) in current model.
```
**Cause**: Input channels changed (e.g., RGB=3 vs Grayscale=1).
**Fix**: Either match the architecture or skip mismatched layers.

---

## 7. Vocabulary Summary

| Term                  | Definition                                              |
|-----------------------|----------------------------------------------------------|
| **Weights**           | The learnable numbers in a neural network                |
| **Parameters**        | Same as weights (used interchangeably)                   |
| **State Dict**        | Python dictionary mapping layer names to weight tensors   |
| **Checkpoint (.ckpt)** | Saved snapshot of model weights                         |
| **Pretrained**        | Weights trained on a large dataset by someone else        |
| **Transfer Learning** | Reusing pretrained knowledge for a new task               |
| **Fine-tuning**       | Training a pretrained model on your specific data         |
| **Feature Extraction** | Using pretrained backbone as frozen feature extractor    |
| **From Scratch**      | Training with random weight initialization                |
| **Kaiming Init**      | Smart random initialization for Conv layers               |
| **running_mean/var**  | BatchNorm's accumulated statistics from training          |
| **DataParallel**      | PyTorch wrapper for multi-GPU training                    |

---

## Self-Check Questions

1. What's the difference between a .ckpt file and a .onnx file?
2. Why does our code load pretrained weights LAST (after Kaiming init)?
3. If training loss keeps decreasing but validation loss starts increasing, what's happening?
4. Why do we save the "best" checkpoint instead of just the last one?
5. If a checkpoint has keys starting with "module.", what does that mean?

(Answers: 1=ckpt=weights only, onnx=weights+graph; 2=Pretrained overwrites
the Kaiming init for backbone, heads keep Kaiming; 3=Overfitting; 4=Last
epoch might be overfitting; 5=Saved with nn.DataParallel wrapper)
