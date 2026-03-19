# KB-003: Epoch, Batch, Iteration — The Training Loop Anatomy

> **Target Audience**: Edge deployment engineers learning ML fundamentals
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `epoch`, `batch`, `iteration`, `training-loop`

---

## 1. The Factory Shift Analogy

Imagine a factory that makes car engines. You have 9,421 engine blocks to
process (your dataset = P3M-10k with 9,421 images).

### One Epoch = One Full Pass Through ALL Data

```
Epoch = The factory processes ALL 9,421 engine blocks exactly once.

If you run 15 epochs, it means the factory processes every single
engine block 15 times total. Each pass teaches the workers (model)
something new about how to shape the metal (predict masks).
```

### One Batch = A Group of Items Processed Together

```
Batch Size = 8 means: the factory picks up 8 engine blocks at a time,
works on all 8 simultaneously, then picks up the next 8.

9,421 images / 8 per batch = 1,177 batches per epoch
(the last partial batch is dropped: drop_last=True)
```

### One Iteration = Processing One Batch

```
Iteration = One batch going through forward + backward pass.
1 epoch = 1,177 iterations (for our setup).
15 epochs = 15 × 1,177 = 17,655 total iterations.
```

---

## 2. Visual Diagram

```
Epoch 1:
  [Batch 0: imgs 0-7] -> forward -> loss -> backward -> update weights
  [Batch 1: imgs 8-15] -> forward -> loss -> backward -> update weights
  ...
  [Batch 1177: imgs 9413-9420] -> forward -> loss -> backward -> update
  (End of Epoch 1: all 9,421 images seen once)
  -> Run validation -> Save checkpoint

Epoch 2:
  [SHUFFLE all images into new random order]
  [Batch 0: imgs ???-???] -> forward -> loss -> backward -> update
  ...
  (the same 9,421 images, but in different order and with different
   data augmentations, so the model sees slightly different versions)

...

Epoch 15:
  [Last pass through all data]
  -> Final validation -> Save best checkpoint
  -> DONE
```

---

## 3. Why Multiple Epochs?

### The Exam Study Analogy

Reading a textbook once (1 epoch) gives you surface understanding.
Reading it 15 times, each time with different colored highlighters
(data augmentation = random flips, color changes), builds DEEP
understanding of the underlying patterns.

But there's a limit — reading it 1000 times doesn't help anymore,
and you might start memorizing specific page layouts instead of
the actual knowledge (this is called **overfitting**).

### The Goldilocks Zone

| Epochs | Effect                                    |
|--------|-------------------------------------------|
| 1-3    | Underfitting: model barely learned         |
| 5-15   | Sweet spot: good generalization            |
| 15-30  | Diminishing returns: small improvements    |
| 50+    | Risk of overfitting: memorizes training data |

We chose 15 epochs because with a pretrained backbone, the model
only needs to learn the task-specific heads (LR/HR/Fusion branches).
The "eyes" (backbone) already know how to see — only the "hands"
need training.

---

## 4. Batch Size and BatchNorm: A Critical Relationship

### Why Batch Size Matters for BN

BatchNorm computes the mean and variance of each channel WITHIN the
current batch. If your batch is too small:

| Batch Size | BN Statistics Quality | Effect                    |
|------------|----------------------|---------------------------|
| 1          | TERRIBLE             | Mean/var of 1 sample = noise |
| 2          | Bad                  | Unstable, noisy training   |
| 4          | Minimum viable       | Somewhat stable            |
| **8**      | **Good**             | **Stable statistics**      |
| 16-32      | Better               | Smoother training          |
| 64+        | Diminishing returns  | Also needs more VRAM       |

This is why our plan specifically chose BS=8. Since we replaced
InstanceNorm (which works on single images) with BatchNorm (which
needs batch statistics), we MUST use a reasonably large batch.

### VRAM Budget (RTX 3090 = 24GB)

```
Memory per sample ≈ Input(3×512×512) + Features + Gradients + Optimizer
                  ≈ ~2.5 GB per sample in training mode

BS=8:  ~20 GB  -> fits in 24GB  ✅
BS=16: ~40 GB  -> does NOT fit  ❌ (would need gradient accumulation)
BS=4:  ~10 GB  -> fits easily   ✅ (but BN less stable)
```

---

## 5. Learning Rate (LR) and StepLR Schedule

### What is Learning Rate?

LR controls how big of a step the model takes when updating its weights
after each batch.

```
High LR (0.1):  Like running downhill with huge strides
                 - Fast but might overshoot and fall off a cliff (NaN)

Medium LR (0.01): Like jogging downhill
                   - Good balance of speed and control

Low LR (0.001):  Like walking carefully downhill
                  - Very stable but extremely slow

Tiny LR (0.0001): Like crawling
                   - For fine-tuning an already-good model
```

### StepLR Schedule: The Deceleration Strategy

```
Our schedule: StepLR(step_size=5, gamma=0.1)

Epoch  1-5:  LR = 0.01    (jogging — learn the big patterns fast)
Epoch  6-10: LR = 0.001   (walking — refine the details)
Epoch 11-15: LR = 0.0001  (crawling — polish the edges)

This is like:
1. First, learn "this is a person" (coarse shape)
2. Then, learn "these are the arms and legs" (structure)
3. Finally, learn "these are the hair strands" (fine detail)
```

### Why Not Just Use a Small LR From the Start?

At LR=0.0001, the model would need 150 epochs (15x longer) to learn
what it can learn in 5 epochs at LR=0.01. The StepLR schedule gives
us the best of both worlds: fast learning early, precise refinement late.

---

## 6. Your Current Training — Reading the Log

```
[Epoch 1/15] LR: 0.010000
  [Batch    0/1177] Loss: 4.7716  <-- First batch, model is clueless
  [Batch   50/1177] Loss: 1.5606  <-- Already improving rapidly!
  [Batch  100/1177] Loss: 0.8618  <-- Getting much better
  [Batch  150/1177] Loss: 0.4850  <-- Very good for first epoch!
```

**What these numbers mean:**
- Loss = 4.77: The predicted mask is very different from the real mask
- Loss = 0.48: The predicted mask is getting close to the real mask
- Loss < 0.20: The predicted mask is visually quite accurate

The pretrained backbone made this possible — with `pretrained=False`,
even after 50 batches the loss was stuck at 2.7.

---

## 7. Vocabulary Summary

| Term            | Definition                                         | Analogy                    |
|-----------------|----------------------------------------------------|-----------------------------|
| **Epoch**       | One complete pass through all training data          | Reading the whole textbook once |
| **Batch**       | A group of N samples processed together              | Processing 8 items at once  |
| **Batch Size**  | Number of samples per batch (N)                      | Items per workload          |
| **Iteration**   | Processing one batch (forward + backward)            | One work cycle              |
| **Learning Rate** | Step size for weight updates                       | Walking speed downhill      |
| **StepLR**      | Reduce LR by gamma every step_size epochs            | Slowing down over time      |
| **Overfitting** | Model memorizes training data, fails on new data     | Memorizing answers, not concepts |
| **Underfitting** | Model hasn't learned enough                         | Barely studied              |
| **Shuffle**     | Randomize data order each epoch                      | Re-deal the cards           |
| **drop_last**   | Discard the last incomplete batch                    | Don't run partial workload  |

---

## Self-Check Questions

1. If dataset=9421, batch_size=8, how many iterations per epoch?
2. Why must batch_size be >= 4 when using BatchNorm?
3. What does StepLR(step=5, gamma=0.1) do to the LR at epoch 6?
4. If Loss stays at 4.0 for 5 epochs, what's likely wrong?
5. Why do we shuffle the data at the start of each epoch?

(Answers: 1=1177; 2=BN needs batch statistics, >=4 for stability;
3=LR becomes 0.01×0.1=0.001; 4=LR too small or backbone not pretrained;
5=Prevents model from memorizing the order of samples)
