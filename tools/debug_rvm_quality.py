#!/usr/bin/env python3
"""Quick RVM ONNX quality check — bypass C++ pipeline entirely.
Reads a single frame from video, runs RVM inference, saves alpha + composite.
"""
import sys
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime first")
    sys.exit(1)

VIDEO = "/development/rk3588s_volume/src/ai/image-matting/primary-folder/media/test_20s.mp4"
MODEL = "rvm-models/rvm_mobilenetv3_fp32_sim.onnx"
OUT_DIR = "./build/debug_rvm_quality"
TARGET_FRAME = 400  # ~13.3 seconds (person should be visible)

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Load model
sess = ort.InferenceSession(MODEL)
input_names = [inp.name for inp in sess.get_inputs()]
output_names = [out.name for out in sess.get_outputs()]
print(f"Model inputs:  {[(n, sess.get_inputs()[i].shape) for i, n in enumerate(input_names)]}")
print(f"Model outputs: {output_names}")

# Read target frame
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps} fps, duration={total_frames/fps:.1f}s")

cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
ret, bgr_frame = cap.read()
cap.release()

if not ret:
    print(f"Failed to read frame {TARGET_FRAME}")
    sys.exit(1)

print(f"Frame {TARGET_FRAME}: {bgr_frame.shape} dtype={bgr_frame.dtype}")
cv2.imwrite(f"{OUT_DIR}/original_frame_{TARGET_FRAME}.png", bgr_frame)

# Preprocess: BGR→RGB, resize to 512x288, normalize to [0,1], NCHW
h, w = bgr_frame.shape[:2]
rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
img_f32 = resized.astype(np.float32) / 255.0  # [0,1]
src = np.transpose(img_f32, (2, 0, 1))[np.newaxis, ...]  # NCHW [1,3,288,512]

print(f"Input tensor: shape={src.shape}, range=[{src.min():.3f}, {src.max():.3f}]")

# Initialize recurrent states (zeros for first frame, then accumulate)
# RVM MobileNetV3 @288x512, downsample_ratio=0.25 → internal 72x128
r1 = np.zeros([1, 16, 36, 64], dtype=np.float32)
r2 = np.zeros([1, 20, 18, 32], dtype=np.float32)
r3 = np.zeros([1, 40,  9, 16], dtype=np.float32)
r4 = np.zeros([1, 64,  5,  8], dtype=np.float32)
dsr = np.array([0.25], dtype=np.float32)

# Run 5 frames to warm up recurrent states (use same frame)
for i in range(5):
    feed = {"src": src, "r1i": r1, "r2i": r2, "r3i": r3, "r4i": r4, "downsample_ratio": dsr}
    fgr, pha, r1, r2, r3, r4 = sess.run(output_names, feed)
    print(f"  Warmup frame {i+1}: pha range=[{pha.min():.4f}, {pha.max():.4f}], mean={pha.mean():.4f}")

# Analyze final alpha
pha_2d = pha[0, 0]  # [H, W]
print(f"\nFinal alpha matte: shape={pha_2d.shape}, range=[{pha_2d.min():.4f}, {pha_2d.max():.4f}]")
print(f"  mean={pha_2d.mean():.4f}, std={pha_2d.std():.4f}")
print(f"  >0.5: {(pha_2d > 0.5).sum()}/{pha_2d.size} pixels ({(pha_2d > 0.5).mean()*100:.1f}%)")
print(f"  >0.9: {(pha_2d > 0.9).sum()}/{pha_2d.size} pixels ({(pha_2d > 0.9).mean()*100:.1f}%)")
print(f"  <0.1: {(pha_2d < 0.1).sum()}/{pha_2d.size} pixels ({(pha_2d < 0.1).mean()*100:.1f}%)")

# Save alpha as grayscale image (0-255)
alpha_vis = (np.clip(pha_2d, 0, 1) * 255).astype(np.uint8)
alpha_full = cv2.resize(alpha_vis, (w, h), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(f"{OUT_DIR}/alpha_frame_{TARGET_FRAME}.png", alpha_full)
print(f"\nSaved: {OUT_DIR}/alpha_frame_{TARGET_FRAME}.png")

# Composite: fg * alpha + green * (1-alpha)
alpha_3ch = np.stack([alpha_full/255.0]*3, axis=-1).astype(np.float32)
green_bg = np.full_like(bgr_frame, [0, 255, 0], dtype=np.uint8)  # BGR green

fg_f = bgr_frame.astype(np.float32) / 255.0
bg_f = green_bg.astype(np.float32) / 255.0
composite = (alpha_3ch * fg_f + (1 - alpha_3ch) * bg_f) * 255
composite = np.clip(composite, 0, 255).astype(np.uint8)

cv2.imwrite(f"{OUT_DIR}/composite_frame_{TARGET_FRAME}.png", composite)
print(f"Saved: {OUT_DIR}/composite_frame_{TARGET_FRAME}.png")

# Also do a multi-frame run from start (simulating real video)
print("\n=== Multi-frame sequential run (first 30 frames) ===")
cap = cv2.VideoCapture(VIDEO)
r1 = np.zeros([1, 16, 36, 64], dtype=np.float32)
r2 = np.zeros([1, 20, 18, 32], dtype=np.float32)
r3 = np.zeros([1, 40,  9, 16], dtype=np.float32)
r4 = np.zeros([1, 64,  5,  8], dtype=np.float32)

for i in range(450):  # ~15 seconds
    ret, f = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
    s = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

    feed = {"src": s, "r1i": r1, "r2i": r2, "r3i": r3, "r4i": r4, "downsample_ratio": dsr}
    fgr_out, pha_out, r1, r2, r3, r4 = sess.run(output_names, feed)

    if i % 30 == 0 or i in [330, 360, 390, 420, 449]:
        pm = pha_out[0,0]
        print(f"  Frame {i:3d} (t={i/fps:.1f}s): pha range=[{pm.min():.3f}, {pm.max():.3f}] mean={pm.mean():.3f} >0.5={100*(pm>0.5).mean():.1f}%")

        # Save key frames
        if i >= 300:
            a = (np.clip(pm, 0, 1) * 255).astype(np.uint8)
            a = cv2.resize(a, (f.shape[1], f.shape[0]))
            cv2.imwrite(f"{OUT_DIR}/alpha_sequential_f{i}.png", a)

            a3 = np.stack([a/255.0]*3, axis=-1).astype(np.float32)
            green = np.full_like(f, [0,255,0], dtype=np.uint8)
            comp = (a3 * f.astype(np.float32)/255.0 + (1-a3) * green.astype(np.float32)/255.0) * 255
            cv2.imwrite(f"{OUT_DIR}/composite_sequential_f{i}.png", np.clip(comp, 0, 255).astype(np.uint8))

cap.release()
print(f"\nAll debug images saved to {OUT_DIR}/")
print("Check alpha_*.png — white=foreground, black=background")
print("Check composite_*.png — person on green background")
