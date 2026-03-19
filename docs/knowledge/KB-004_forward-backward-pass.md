# KB-004: Forward Pass, Backward Pass, Gradient Descent — How AI Actually Learns

> **Target Audience**: Edge deployment engineers learning ML fundamentals
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `forward-pass`, `backward-pass`, `gradient`, `loss`, `optimizer`

---

## 1. The Big Picture: How a Neural Network Learns

```
          FORWARD PASS                    BACKWARD PASS
          (Prediction)                    (Learning)

Input Image ──> [Neural Network] ──> Predicted Mask
                                          |
                                     Compare with
                                          |
                                     Ground Truth Mask
                                          |
                                     Loss (error number)
                                          |
                                     Compute Gradients
                                          |
                                     Update Weights
                                          |
                                     [Neural Network]  <-- weights slightly better now
```

---

## 2. Forward Pass (Prediction)

**What**: Push data through the network from input to output.
**Analogy**: A student taking an exam.

```python
# In code (from trainer.py):
pred_semantic, pred_detail, pred_matte = modnet(image, False)
#                                        ^^^^^^^^^^^^^^^^^^^^^^
#                                        This IS the forward pass
```

### What happens inside:

```
Input Image (3, 512, 512)
    |
    v
[MobileNetV2 Backbone]  -- "eyes" extract features
    |
    +---> Feature Maps at 2x, 4x, 8x, 16x, 32x scales
    |
    v
[LR Branch] --> pred_semantic (1, 32, 32)  -- coarse "is this a person?"
[HR Branch] --> pred_detail   (1, 512, 512) -- fine edge prediction
[Fusion]    --> pred_matte    (1, 512, 512) -- final alpha mask
```

**Key insight**: During forward pass, NOTHING is learned. The network
just uses its current weights to make a prediction. It's purely a
calculation: input × weights = output.

---

## 3. Loss Function (Grading the Exam)

**What**: A mathematical formula that measures HOW WRONG the prediction is.
**Analogy**: A teacher grading the student's exam paper.

```python
# From trainer.py:
semantic_loss = MSE(pred_semantic, gt_semantic)  # squared error
detail_loss   = L1(pred_detail, gt_detail)       # absolute error
matte_loss    = L1(pred_matte, gt_matte)         # absolute error
```

### Loss Types Explained

**L1 Loss (Mean Absolute Error)**:
```
For each pixel: |prediction - truth|
Then average over all pixels.

Example:
  Prediction: 0.8 (model thinks 80% foreground)
  Truth:      1.0 (actually 100% foreground)
  L1 Loss:    |0.8 - 1.0| = 0.2
```

**L2 / MSE Loss (Mean Squared Error)**:
```
For each pixel: (prediction - truth)^2
Then average over all pixels.

Example:
  Prediction: 0.8
  Truth:      1.0
  MSE Loss:   (0.8 - 1.0)^2 = 0.04

MSE penalizes BIG errors much more than small errors.
An error of 0.5 costs 0.25 (vs L1's 0.5)
An error of 0.1 costs 0.01 (vs L1's 0.1)
```

### MODNet's 3-Branch Loss System

```
Total Loss = 10 × Semantic Loss   (coarse shape matters A LOT)
           + 10 × Detail Loss     (edge quality matters A LOT)
           +  1 × Matte Loss      (final output, baseline weight)

The 10× multipliers (semantic_scale, detail_scale) force the model
to prioritize getting the coarse shape and edges right.
```

### What Loss Numbers Mean

| Loss Value | Meaning                                       |
|------------|-----------------------------------------------|
| > 5.0      | Random garbage, model knows nothing            |
| 2.0 - 5.0  | Model is learning, predictions are blurry      |
| 1.0 - 2.0  | Coarse shape is correct, edges still rough     |
| 0.3 - 1.0  | Good predictions, visible fine details         |
| < 0.3      | Excellent, near ground-truth quality           |

---

## 4. Backward Pass (Learning From Mistakes)

**What**: Compute how much each weight contributed to the error, then
adjust weights to reduce the error.
**Analogy**: The student reviews their mistakes and figures out WHICH
specific knowledge gaps caused each wrong answer.

### The Chain Rule (Backpropagation)

The network has millions of weights (MODNet has 6,487,795). For EACH
weight, we need to answer: "If I increase this weight by a tiny amount,
does the loss go up or down? And by how much?"

This answer is called the **gradient** of that weight.

```python
# In code (from trainer.py):
loss = semantic_loss + detail_loss + matte_loss
loss.backward()  # <-- THIS is the backward pass
#                     PyTorch automatically computes ALL 6.5 million gradients
```

### Gradient = Direction of Steepest Ascent

```
Imagine you're blindfolded on a hilly landscape.
The loss is your altitude. You want to go DOWNHILL (minimize loss).

The gradient tells you: "the steepest uphill direction is THIS way"
So you walk in the OPPOSITE direction (gradient descent).

  gradient = +0.05  --> weight is making loss worse, decrease it
  gradient = -0.03  --> weight is helping, increase it a little
  gradient =  0.00  --> weight is at a good spot, don't change
```

### The Math (Simplified)

```
For each weight w:
  new_w = old_w - learning_rate × gradient

Example:
  old_w = 0.50
  gradient = +0.20 (this weight is making things worse)
  LR = 0.01

  new_w = 0.50 - 0.01 × 0.20 = 0.498

The weight decreased by 0.002, nudging the prediction slightly
toward the correct answer.
```

---

## 5. Optimizer (The Weight Update Strategy)

**What**: The algorithm that decides exactly how to update each weight.
**Analogy**: The study strategy the student uses to improve.

### SGD (Stochastic Gradient Descent) — What We're Using

```python
optimizer = torch.optim.SGD(modnet.parameters(), lr=0.01, momentum=0.9)
```

**SGD = The simplest optimizer**: Just follow the gradient downhill.

**Momentum (0.9)**: Like a heavy ball rolling downhill — it builds up
speed in a consistent direction and resists sudden direction changes.
This helps the model not get stuck in small bumps (local minima).

```
Without momentum:  zig-zag-zig-zag down the hill (noisy, slow)
With momentum=0.9: smooth curved path down the hill (faster, stable)
```

### Other Optimizers You'll Encounter

| Optimizer | Analogy                | When to Use              |
|-----------|------------------------|--------------------------|
| **SGD**   | Heavy ball rolling     | Large datasets, proven   |
| **Adam**  | Smart GPS navigation   | Small datasets, fast start |
| **AdamW** | GPS with weight decay  | Modern default choice    |

MODNet's paper uses SGD, so we stick with it for compatibility.

---

## 6. Putting It All Together: One Training Iteration

```
Step 1: FORWARD PASS
  Input image (3, 512, 512) --> network --> predicted mask (1, 512, 512)

Step 2: LOSS COMPUTATION
  Compare predicted mask with ground truth mask
  Loss = 0.48 (the prediction is 48% "wrong" on average)

Step 3: BACKWARD PASS (loss.backward())
  Compute gradient for all 6,487,795 weights
  "weight #1234 should decrease by 0.002"
  "weight #5678 should increase by 0.001"
  ...

Step 4: OPTIMIZER STEP (optimizer.step())
  Apply all 6.5 million weight updates simultaneously
  The model is now SLIGHTLY better at predicting masks

Step 5: ZERO GRADIENTS (optimizer.zero_grad())
  Clear the gradients, ready for the next batch

Repeat 1,177 times = 1 epoch
Repeat 15 epochs = training complete
```

---

## 7. Why "Stochastic"?

"Stochastic" means "random". In SGD, we don't compute the gradient
over ALL 9,421 images (that would be too expensive). Instead, we:

1. Randomly pick a batch of 8 images
2. Compute gradients only on those 8
3. Update weights
4. Pick another random batch of 8
5. Repeat

Each batch gives a slightly noisy estimate of the true gradient,
but averaged over 1,177 batches per epoch, it converges to the
right answer. The randomness actually HELPS — it prevents the model
from getting stuck in bad local minima.

---

## 8. The Complete Training Pipeline

```
                        ┌──────────────────────────────────┐
                        │        TRAINING LOOP              │
                        │                                   │
  Dataset ─────────────>│  for epoch in range(15):          │
  (9421 images)         │    shuffle data                   │
                        │    for batch in dataloader:       │
                        │      1. Forward pass              │
                        │      2. Compute loss              │
                        │      3. Backward pass             │
                        │      4. Update weights            │
                        │    end batch loop                 │
                        │    scheduler.step() (decay LR)    │
                        │    validate on 500 images         │
                        │    save checkpoint                │
                        │  end epoch loop                   │
                        │                                   │
                        │  Output: modnet_bn_best.ckpt      │
                        └──────────────────────────────────┘
```

---

## 9. Vocabulary Summary

| Term                   | Definition                                              |
|------------------------|----------------------------------------------------------|
| **Forward Pass**       | Input -> Network -> Prediction (no learning)             |
| **Loss**               | Number measuring prediction error (lower = better)       |
| **L1 Loss**            | Average absolute difference per pixel                    |
| **MSE / L2 Loss**      | Average squared difference per pixel                    |
| **Backward Pass**      | Compute gradients via backpropagation                    |
| **Gradient**           | How much each weight should change to reduce loss        |
| **Backpropagation**    | Algorithm to compute gradients using chain rule           |
| **Gradient Descent**   | Update weights in opposite direction of gradient          |
| **SGD**                | Stochastic Gradient Descent (batch-based updates)        |
| **Momentum**           | Smooths out gradient noise for faster convergence        |
| **Optimizer**          | Algorithm that applies weight updates (SGD, Adam, etc.)  |
| **optimizer.step()**   | Apply the computed gradients to update weights           |
| **optimizer.zero_grad()** | Clear gradients before next batch                    |
| **Local Minimum**      | A valley that isn't the lowest point (getting stuck)     |

---

## Self-Check Questions

1. During forward pass, does the model learn anything?
2. What does loss.backward() compute?
3. If gradient = +0.1 and LR = 0.01, does the weight increase or decrease?
4. Why use momentum=0.9 instead of 0?
5. Why do we call optimizer.zero_grad() before each iteration?

(Answers: 1=No, just prediction; 2=Gradients for all weights;
3=Decrease (new = old - 0.01×0.1); 4=Smooths noise, prevents zig-zag;
5=Gradients accumulate by default, must clear before new batch)
