# KB-008: Overfitting, Underfitting, Regularization — The Generalization Problem

> **Target Audience**: Edge deployment engineers learning ML fundamentals
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `overfitting`, `underfitting`, `regularization`, `validation`, `generalization`

---

## 1. The Core Problem: Generalization

Training a model is easy. Making it work on UNSEEN data is hard.

### The Student Exam Analogy

```
Training Data  = Practice problems with answer key
Validation Data = Mock exam (different problems, no answer key)
Test Data      = Real exam (deployment: real-world photos)

Good Student: Understands concepts -> scores well on ALL exams
Bad Student A: Memorized practice answers -> fails mock exam (OVERFITTING)
Bad Student B: Barely studied -> fails everything (UNDERFITTING)
```

---

## 2. Overfitting — "Memorizing the Textbook"

### What Happens

The model memorizes the training data so perfectly that it ONLY works
on training data. It fails on new, unseen images.

```
Training Loss:    0.01  (nearly perfect on training images!)
Validation Loss:  0.15  (much worse on new images)

The GAP between training and validation loss = overfitting signal.
```

### Visual Example

```
Ground Truth Mask (what we want):
+------------------+
|     ██████       |
|   ██████████     |
|   ██████████     |
|     ██████       |
+------------------+

Overfitted Model's output on TRAINING image:
+------------------+
|     ██████       |  Nearly perfect! (memorized this exact image)
|   ██████████     |
|   ██████████     |
|     ██████       |
+------------------+

Overfitted Model's output on NEW image:
+------------------+
|  █ ██  ███  █    |  Garbage! (never seen this image before)
|   █ ████ █ ██    |
|    ████ ██ █     |
|   █ ██ █ ████    |
+------------------+
```

### Warning Signs of Overfitting

```
Epoch:  1    2    3    4    5    6    7    8    9   10
Train:  4.0  2.0  1.0  0.5  0.3  0.15 0.08 0.03 0.01 0.005
Val:    3.8  1.9  1.0  0.6  0.4  0.35 0.40 0.50 0.60 0.80
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  Val INCREASING = OVERFITTING!

The model is getting BETTER at training data but WORSE at new data.
```

### Causes

1. **Too many epochs**: Model has too many chances to memorize
2. **Too few training samples**: Easy to memorize small datasets
3. **Model too complex**: Too many parameters for the data amount
4. **No data augmentation**: Sees exact same images every epoch
5. **Learning rate too low for too long**: Refines toward memorization

---

## 3. Underfitting — "Barely Studied"

### What Happens

The model hasn't learned enough to make useful predictions.

```
Training Loss:    2.5  (still high after many epochs)
Validation Loss:  2.6  (also high, but similar to training)

Both losses are high and similar = underfitting.
```

### Causes

1. **Too few epochs**: Training stopped too early
2. **Learning rate too low**: Model isn't learning fast enough
3. **Model too simple**: Not enough parameters to capture the patterns
4. **Bug in code**: Data is corrupted or labels are wrong

---

## 4. The Sweet Spot

```
         Loss
          ^
          |
  4.0  ─  |╲
          | ╲  Training Loss (keeps decreasing)
  3.0  ─  |  ╲
          |   ╲
  2.0  ─  |    ╲___________  ← Underfitting zone
          |      ╲
  1.0  ─  |       ╲ ______ ← SWEET SPOT (stop here!)
          |         ╳
  0.5  ─  |        ╱  ╲ Validation Loss
          |       ╱    ╲  (starts increasing = overfitting)
  0.0  ─  |______╱______╲___________→ Epochs
          0    5    10    15    20    25
```

**Best checkpoint = the point where Validation Loss is LOWEST.**
That's why our training script tracks `best_val_loss` and saves
the best checkpoint separately.

---

## 5. Regularization Techniques — Fighting Overfitting

### 5a. Data Augmentation (What We Use)

Create "fake" training samples by transforming real ones:

```python
# Our augmentations in train_modnet_block1_2.py:
RandomHorizontalFlip()    # Mirror the image left-right
ColorJitter(              # Random color changes
    brightness=0.3,       # ±30% brightness
    contrast=0.3,         # ±30% contrast
    saturation=0.3,       # ±30% color intensity
    hue=0.1               # ±10% color shift
)
```

```
Original Image:              Augmented versions:
  [Person facing right]      [Person facing LEFT]  (flipped)
                             [Person, brighter]     (color jitter)
                             [Person, bluish tint]  (hue shift)

Each epoch, the model sees DIFFERENT versions of the same image.
It can't memorize because the input keeps changing!
```

### 5b. Dropout (Not Used in Our Model)

Randomly disable X% of neurons during training.

```
Normal:  [N1] → [N2] → [N3] → [N4] → output
Dropout: [N1] → [ X ] → [N3] → [ X ] → output
         (N2 and N4 randomly disabled this iteration)

Forces the network to be redundant — no single neuron is critical.
Like training a team where random members call in sick each day.
```

### 5c. Weight Decay (L2 Regularization)

Add a penalty for large weight values:

```
Total Loss = Task Loss + λ × sum(weight^2)

If λ = 0.0001, the model is discouraged from having large weights.
This prevents it from over-specializing to training data patterns.
```

### 5d. Early Stopping

Monitor validation loss. If it hasn't improved for N epochs, stop.

```
Epoch 8:  Val Loss = 0.020 (new best!)
Epoch 9:  Val Loss = 0.021
Epoch 10: Val Loss = 0.023
Epoch 11: Val Loss = 0.025  -> patience exhausted, STOP
                               Use the epoch 8 checkpoint.
```

### 5e. Batch Normalization (Implicit Regularization!)

BN acts as a mild regularizer because:
- Each batch has slightly different statistics (mean, var)
- This adds noise to the training process
- The noise prevents memorization

**This is another benefit of our IBNorm → BN replacement!**

---

## 6. Our Training's Overfitting Risk Assessment

```
Dataset size:        9,421 images       (medium)
Model parameters:    6,487,795          (medium)
Data augmentation:   Flip + ColorJitter (moderate)
Epochs:              15                 (conservative)
LR schedule:         StepLR decay       (prevents late-stage memorization)
Validation:          500 images/epoch   (monitors generalization)
Best checkpoint:     Tracked            (automatic early stopping equivalent)

Risk level: LOW-MEDIUM
Reason: pretrained backbone already generalizes well,
        15 epochs is conservative, augmentation is active.
```

---

## 7. Reading Our Training Logs for Overfitting

```
[Epoch 1]  Train Avg: 0.5410, Val: 0.0264  <- Learning fast
[Epoch 5]  Train Avg: 0.2100, Val: 0.0180  <- Still improving
[Epoch 10] Train Avg: 0.0800, Val: 0.0160  <- Plateauing

Watch for:
[Epoch 12] Train Avg: 0.0500, Val: 0.0170  <- Val going UP = warning!
[Epoch 15] Train Avg: 0.0200, Val: 0.0200  <- Gap widening = overfitting

If this happens, use modnet_bn_best.ckpt from epoch 10, not epoch 15.
Our code handles this automatically.
```

---

## 8. Vocabulary Summary

| Term                  | Definition                                              |
|-----------------------|----------------------------------------------------------|
| **Overfitting**       | Model memorizes training data, fails on new data         |
| **Underfitting**      | Model hasn't learned enough, fails on all data           |
| **Generalization**    | Model's ability to work on unseen data                   |
| **Regularization**    | Techniques to prevent overfitting                        |
| **Data Augmentation** | Creating variations of training data (flip, color, etc.) |
| **Dropout**           | Randomly disabling neurons during training               |
| **Weight Decay**      | Penalty for large weight values                          |
| **Early Stopping**    | Stop training when validation loss stops improving       |
| **Validation Set**    | Data withheld from training to measure generalization    |
| **Train-Val Gap**     | Difference between training and validation loss          |
| **Patience**          | How many epochs to wait for improvement before stopping  |

---

## Self-Check Questions

1. Training loss = 0.01, Validation loss = 0.15. Is this overfitting or underfitting?
2. Training loss = 2.5, Validation loss = 2.6. Is this overfitting or underfitting?
3. Why does our code save "best" checkpoint separately from "last" checkpoint?
4. How does RandomHorizontalFlip help prevent overfitting?
5. Why is BatchNorm considered a mild regularizer?

(Answers: 1=Overfitting (large gap); 2=Underfitting (both high);
3=Last might be overfitting, best is where val loss was lowest;
4=Model sees mirrored versions, can't memorize exact pixel positions;
5=Batch statistics add noise that prevents memorization)
