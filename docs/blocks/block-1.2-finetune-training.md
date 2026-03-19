# Block 1.2: Fine-tune Training (Pure BatchNorm MODNet)

> **Phase**: 1 (Model Retraining)
> **Block**: 1.2 (Fine-tune Training)
> **Status**: IN PROGRESS
> **Date Started**: 2026-03-19
> **Prerequisites**: Block 1.0 (DataLoader) ✅, Block 1.1 (BN Surgery) ✅

---

## Objective

Train the modified Pure-BN MODNet on P3M-10k dataset to produce a usable
alpha matte model. This is the first time the modified architecture will be
fully trained — previous Block 1.1 only ran 50 batches as a smoke test.

## Training Configuration

| Parameter         | Value                          | Rationale                                           |
|-------------------|--------------------------------|-----------------------------------------------------|
| Epochs            | 15                             | Sufficient for pretrained backbone convergence       |
| Backbone          | `pretrained=True`              | Use ImageNet MobileNetV2 weights (5-10x faster)      |
| Batch Size        | 8                              | BN needs >=4 for stable statistics; 8 fits 24GB VRAM |
| Input Resolution  | 512x512                        | Preserves hair detail for learning (see KB-001)       |
| Learning Rate     | 0.01 -> StepLR(step=5, g=0.1)  | Decays at epoch 5, 10, 15                            |
| Optimizer         | SGD (momentum=0.9)             | Matches original MODNet paper                        |
| Augmentation      | RandomFlip + ColorJitter        | Improves generalization                              |
| Validation        | P3M-500-P (500 images)          | Every epoch with visual output                       |

### Why backbone_pretrained=True?

The backbone (MobileNetV2) was NOT modified in Block 1.1.
Only the MODNet heads (LR/HR/Fusion branches) had IBNorm -> BN surgery.
Therefore ImageNet pretrained weights load perfectly into the backbone.

- `pretrained=True` = Experienced photographer (already knows edges/textures)
- `pretrained=False` = Blind newborn (must learn everything from scratch)

This single change reduces training time from ~20 hours to ~2-3 hours.

### Why 512x512 training but 256x256 deployment?

See KB-001 for full explanation. Short version:
- At 512, hair strands are ~1.4px wide -> AI can see and learn them
- At 256, hair strands are ~0.7px wide -> AI cannot see what doesn't exist
- Phase 3 Guided Filter will use 1080p original to refine the rough 256 mask

## Execution

```bash
# Activate virtual environment
source /path/to/helmsman.git/.venv/bin/activate

# Navigate to MODNet directory
cd helmsman.git/third-party/sdk/MODNet.git

# Run training
python3 -m train_modnet_block1_2
```

Estimated time: **2-3 hours** on RTX 3090 24GB.

## Verification Criteria (How to Know Block 1.2 Succeeded)

### SUCCESS indicators (ALL must be true):

1. **Loss convergence**: Total Loss should decrease from ~4.0 to below ~1.5
   - Semantic Loss: Should drop significantly (this is the coarse shape)
   - Detail Loss: Should decrease gradually (this is edge refinement)
   - Matte Loss: Should reach below 1.0 (this is the final mask quality)

2. **Visual validation**: Open `output/epoch_15_val.png`
   - The predicted mask (middle column) should show a clear human silhouette
   - Body interior should be mostly white (no holes/voids)
   - Background should be mostly black (no ghost artifacts)
   - Hair region should show some detail (not a hard staircase edge)

3. **No NaN/Inf**: Loss values should never become NaN or infinity

4. **Checkpoint saved**: `checkpoints/modnet_bn_best.ckpt` exists and is > 0 bytes

### FAILURE modes and recovery:

#### Failure A: CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```
**Cause**: 512x512 with BS=8 exceeds 24GB VRAM.
**Fix**: Edit `train_modnet_block1_2.py`, change `BATCH_SIZE = 8` to `4`.

#### Failure B: Loss Explodes (NaN)
```
[Epoch 1][Batch XX] Loss: nan
```
**Cause**: Learning rate too high for randomly-initialized BN heads.
**Fix**: Change `LR = 0.01` to `LR = 0.001`.

#### Failure C: Loss Stuck / Doesn't Decrease
```
[Epoch 1] Avg Loss: 3.50
[Epoch 5] Avg Loss: 3.48   <-- barely moved after 5 epochs!
```
**Cause**: Learning rate too low, or backbone weights not loading.
**Fix**: Check that the console shows `[MobileNetV2] Loading pretrained model...`
during startup. If not, the pretrained weights are missing.

#### Failure D: Visual Results Show Inverted/Garbage Mask
```
Prediction column is all white, all black, or random noise
```
**Cause**: The IBNorm->BN surgery in Block 1.1 has a channel mismatch bug.
**Fix**: Examine `modnet.py` for tensor shape mismatches. Report error to copilot.

#### Failure E: Backbone Pretrained Weights Not Found
```
cannot find the pretrained mobilenetv2 backbone
```
**Cause**: `pretrained/mobilenetv2_human_seg.ckpt` symlink is broken.
**Fix**: Verify the symlink target exists:
```bash
ls -la pretrained/mobilenetv2_human_seg.ckpt
```

## Output Artifacts

After training completes:

| File | Description |
|------|-------------|
| `checkpoints/modnet_bn_best.ckpt` | Best model by validation loss |
| `checkpoints/modnet_bn_epoch_XX.ckpt` | Per-epoch checkpoints |
| `output/epoch_XX_val.png` | Visual comparison: Input \| Pred \| GT |

## Visual Result Interpretation Guide

When opening `output/epoch_XX_val.png`:

```
+-------------------+-------------------+-------------------+
|     INPUT         |    PREDICTION     |   GROUND TRUTH    |
|   (RGB photo)     |   (AI output)     |   (real mask)     |
+-------------------+-------------------+-------------------+
```

**What to look for:**
- Compare middle column (AI) with right column (truth)
- **Good**: Shapes match, body is white, background is black
- **Bad**: Holes inside body, ghost regions in background, mushy edges

**Quality progression across epochs:**
- Epoch 1-3: Blurry blob, vaguely human-shaped -> NORMAL
- Epoch 5-8: Clear silhouette, some edge noise -> IMPROVING
- Epoch 10-15: Clean mask, visible hair detail -> CONVERGED

## Next Step After Block 1.2

If verification passes:
-> **Block 1.4**: ONNX Export (export the best checkpoint to .onnx)

If quality is poor despite Loss converging:
-> **Block 1.3 (Fallback)**: Knowledge Distillation
   (use original FP32 MODNet as teacher to guide the pure-BN student)

---

## Changelog

| Date       | Event                                       |
|------------|---------------------------------------------|
| 2026-03-19 | Block 1.2 plan created, training script written |
| 2026-03-19 | Strategy: Fast Validation (15ep, pretrained) |
