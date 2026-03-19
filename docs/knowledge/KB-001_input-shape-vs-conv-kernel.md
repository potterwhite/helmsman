# KB-001: Input Shape vs Convolution Kernel — The Two Most Confused Concepts

> **Target Audience**: Edge deployment engineers with no ML paper background
> **Author**: Claude (AI Consultant) for Potter White
> **Date**: 2026-03-19
> **Tags**: `fundamentals`, `input-shape`, `convolution`, `resolution`, `NCHW`

---

## 1. The Misconception (What You Thought)

> "The 256x256 is the convolution kernel size. It slides across a large image
> like a window, doing 256x256 convolutions each time to detect edges."

**This is WRONG.** But it's a very natural and understandable mistake.
Let's fix it permanently.

---

## 2. What is Input Shape (NCHW)?

**NCHW** stands for:

| Letter | Meaning          | Example     | Analogy                              |
|--------|------------------|-------------|--------------------------------------|
| **N**  | Batch size       | 1           | How many photos you feed at once     |
| **C**  | Channels         | 3 (RGB)     | How many color layers (R, G, B)      |
| **H**  | Height (pixels)  | 256         | Photo height after resize            |
| **W**  | Width (pixels)   | 256         | Photo width after resize             |

**Input Shape = The size your original photo gets RESIZED to BEFORE the AI sees it.**

### The Camera Analogy

Imagine you have a 4K (3840x2160) security camera feed. Before handing the
image to the AI, you SHRINK the entire photo down to 256x256 pixels.

```
Original Photo (1920x1080)
+--------------------------------------------------+
|                                                    |
|        [Person with detailed hair strands]         |
|                                                    |
+--------------------------------------------------+
                    |
                    | cv2.resize() — SHRINK entire photo
                    v
Resized Photo (256x256)
+----------+
| [Blurry  |
|  blob]   |
+----------+
```

**The AI only sees the tiny 256x256 version.** If hair strands were 2 pixels
wide in the original 1080p image, after shrinking to 256x256, those strands
become **0.3 pixels wide — they literally VANISH from existence.**

The AI cannot detect what doesn't exist in its input.

---

## 3. What is a Convolution Kernel?

A convolution kernel is a TINY matrix (typically 3x3 or 5x5 pixels) that
slides across the ALREADY-RESIZED input image.

### The Magnifying Glass Analogy

Think of the convolution kernel as a **tiny magnifying glass** (3x3):

```
Input Image (256x256)          Convolution Kernel (3x3)
+------------------------+     +-------+
|                        |     | w w w |
|    The kernel slides   |     | w w w |  <-- Only 3x3 pixels!
|    across EVERY        |     | w w w |
|    position in this    |     +-------+
|    image, one step     |
|    at a time.          |     (w = learned weight values)
|                        |
+------------------------+
```

The kernel looks at 3x3 = 9 pixels at a time, computes a weighted sum,
and produces ONE output pixel. By sliding across the entire image, it
produces a new "feature map" that highlights certain patterns (edges,
textures, etc.).

### Key Differences

| Property        | Input Shape          | Convolution Kernel        |
|-----------------|----------------------|---------------------------|
| **Size**        | 256x256 or 512x512   | 3x3 or 5x5 (tiny!)       |
| **What it is**  | The photo itself     | A tiny pattern detector   |
| **Who decides** | You (the engineer)   | The AI learns it          |
| **Analogy**     | The canvas           | The magnifying glass      |

---

## 4. Why Does Small Input Shape Destroy Quality?

### The Pixel Budget Problem

This is pure math. Let's say a human hair strand is 3 pixels wide in 1080p:

| Input Resolution | Scale Factor vs 1080p | Hair Width (pixels) | Result               |
|------------------|-----------------------|----------------------|----------------------|
| 1080x1920        | 1.0x                  | 3.0 px               | Perfect strands      |
| 576x1024         | ~0.53x                | 1.6 px               | Thin but visible     |
| 512x512          | ~0.47x (non-uniform)  | 1.4 px               | Barely visible       |
| 384x384          | ~0.36x                | 1.1 px               | Sub-pixel, vanishing |
| 256x256          | ~0.24x                | 0.7 px               | **GONE. Does not exist.** |

**At 256x256, hair strands are less than 1 pixel. They are physically
destroyed by the resize operation.** No AI in the universe can detect what
isn't there.

### The Jagged Edge Problem (Aliasing)

You described: "edges become jagged sawtooth, fingers look amputated."

This is called **aliasing**. At low resolution:
- Smooth curves become staircase steps (jagged edges)
- Thin structures (fingers, hair) merge with background
- The AI's output mask is also low-res, so when you scale it back to
  1080p, each "AI pixel" covers a large area — creating visible blocks

```
High-res mask (smooth edge)      Low-res mask (aliased edge)
+--+--+--+--+--+--+             +-----+-----+-----+
|  |  |##|##|##|##|             |     |#####|#####|
+--+--+--+--+--+--+             |     |#####|#####|
|  |  |  |##|##|##|             +-----+-----+-----+
+--+--+--+--+--+--+             |     |     |#####|
|  |  |  |  |##|##|             |     |     |#####|
+--+--+--+--+--+--+             +-----+-----+-----+
  ^ fine pixel grid               ^ coarse pixel grid (zoomed out)
```

---

## 5. The Resolution vs Speed Tradeoff (The Core Dilemma)

On RK3588 NPU (6 TOPS):

| Input Shape | NPU Inference Time | Edge Quality     | Hair Detail   |
|-------------|--------------------|--------------------|---------------|
| 512x512     | ~80-120ms (SLOW)   | Smooth             | Good          |
| 384x384     | ~35-50ms           | Acceptable         | Weak          |
| 256x256     | ~15-22ms (FAST)    | Jagged             | **None**      |

**This is why Phase 3 (Guided Filter) exists in our master plan!**

The strategy is:
1. Run the AI at LOW resolution (256x256) for SPEED → get a rough mask
2. Use a TRADITIONAL algorithm (Guided Filter) to refine the rough mask
   using the ORIGINAL 1080p image's edge information → recover hair detail

This is called **"Super-Resolution Refinement"** or **"AI + CV Hybrid"**.

---

## 6. The Correct Technical Vocabulary

| Your Words                    | Correct Term              | Explanation                         |
|-------------------------------|---------------------------|-------------------------------------|
| "input shape"                 | **Input Resolution**      | The H×W of the tensor fed to model  |
| "convolution kernel size"     | **Kernel Size**           | The tiny 3×3 or 5×5 filter          |
| "the AI scans 256x256 areas" | Wrong — see above         | The entire image IS 256×256         |
| "hair gets lost"              | **Spatial Detail Loss**   | Fine structures destroyed by resize |
| "jagged edges"                | **Aliasing Artifacts**    | Low-res → high-res staircase effect |
| "fingers look amputated"      | **Structural Distortion** | Thin structures merge/disappear     |

---

## 7. Summary: The Golden Rule

> **Input Resolution = How much detail the AI CAN POSSIBLY see.**
> **Convolution Kernel = How the AI PROCESSES what it sees.**
>
> Making the kernel bigger does NOT compensate for a tiny input.
> It's like using a bigger magnifying glass on a blurry photocopy —
> you see more blur, not more detail.

---

## References for Self-Study
- [Conv2d Visualization](https://github.com/vdumoulin/conv_arithmetic) — animated GIFs of how kernels slide
- [Aliasing in Image Processing](https://en.wikipedia.org/wiki/Aliasing) — why jagged edges happen
- NCHW layout is RKNN's native format — matches your C++ pipeline's `data_structure.h`
