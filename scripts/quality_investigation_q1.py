#!/usr/bin/env python3
"""
Phase-5 Block 5.Q: RVM Quality Investigation — PyTorch & ONNX experiments.

Runs multiple configurations to compare RVM output quality:
  - PyTorch with Refiner (golden standard)
  - PyTorch without Refiner (isolate Refiner contribution)
  - PyTorch ratio=1.0 (no downsampling, no Refiner triggered)
  - ONNX ratio=0.25 (current pipeline equivalent)
  - ONNX ratio=1.0 (max quality without Refiner)

Usage:
    cd helmsman.git
    source .venv/bin/activate
    python scripts/quality_investigation_q1.py \\
        --checkpoint /path/to/rvm_mobilenetv3.pth \\
        --input /path/to/test_video.mp4 \\
        --output-dir ./build/quality_q1/

Dependencies (in helmsman .venv):
    pip install tqdm av   # one-time only

Reproducibility:
    - All results written to --output-dir with experiment name prefix
    - Each experiment outputs: composition.mp4 + alpha.mp4
    - Compare visually + check alpha statistics printed to stdout
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

# Add RVM submodule to path
RVM_ROOT = os.path.join(os.path.dirname(__file__), "..",
                        "third-party", "sdk", "RobustVideoMatting.git")
sys.path.insert(0, RVM_ROOT)

from model import MattingNetwork  # noqa: E402
from inference_utils import VideoReader, VideoWriter  # noqa: E402


def run_pytorch_experiment(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    experiment_name: str,
    downsample_ratio: float,
    disable_refiner: bool = False,
    max_frames: int = 0,
):
    """Run a single PyTorch experiment."""
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"  downsample_ratio: {downsample_ratio}")
    print(f"  disable_refiner: {disable_refiner}")
    print(f"  max_frames: {max_frames if max_frames > 0 else 'all'}")
    print(f"{'='*70}")

    # Load model
    model = MattingNetwork("mobilenetv3").eval().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Optionally disable refiner by monkey-patching forward
    if disable_refiner:
        original_forward = model.forward

        def forward_no_refiner(src, r1=None, r2=None, r3=None, r4=None,
                               downsample_ratio=1, segmentation_pass=False):
            if downsample_ratio != 1:
                src_sm = model._interpolate(src, scale_factor=downsample_ratio)
            else:
                src_sm = src

            f1, f2, f3, f4 = model.backbone(src_sm)
            f4 = model.aspp(f4)
            hid, *rec = model.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

            fgr_residual, pha = model.project_mat(hid).split([3, 1], dim=-3)
            # Skip refiner — just upsample back to src resolution
            if downsample_ratio != 1:
                # Handle 5D [B,T,C,H,W] by flattening to 4D
                if fgr_residual.ndim == 5:
                    B, T = fgr_residual.shape[:2]
                    fgr_residual = F.interpolate(
                        fgr_residual.flatten(0, 1), size=src.shape[-2:],
                        mode='bilinear', align_corners=False).unflatten(0, (B, T))
                    pha = F.interpolate(
                        pha.flatten(0, 1), size=src.shape[-2:],
                        mode='bilinear', align_corners=False).unflatten(0, (B, T))
                else:
                    fgr_residual = F.interpolate(fgr_residual, size=src.shape[-2:],
                                                  mode='bilinear', align_corners=False)
                    pha = F.interpolate(pha, size=src.shape[-2:],
                                        mode='bilinear', align_corners=False)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]

        model.forward = forward_no_refiner

    # Prepare paths
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    comp_path = os.path.join(exp_dir, "composition.mp4")
    alpha_path = os.path.join(exp_dir, "alpha.mp4")

    # Read video
    transform = transforms.ToTensor()
    source = VideoReader(input_path, transform)
    frame_rate = source.frame_rate
    total_frames = len(source) if max_frames <= 0 else min(max_frames, len(source))

    writer_com = VideoWriter(comp_path, frame_rate, bit_rate=4_000_000)
    writer_pha = VideoWriter(alpha_path, frame_rate, bit_rate=4_000_000)

    # Green screen background
    bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    # Inference
    rec = [None] * 4
    alpha_values = []
    t0 = time.time()

    try:
        with torch.no_grad():
            for i, src in enumerate(tqdm(source, total=total_frames, desc=experiment_name)):
                if max_frames > 0 and i >= max_frames:
                    break

                src = src.to(device, dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                # Collect stats
                pha_np = pha[0, 0, 0].cpu().numpy()
                alpha_values.append(pha_np.mean())

                # Write composition
                com = fgr * pha + bgr * (1 - pha)
                writer_com.write(com[0])
                writer_pha.write(pha[0])
    finally:
        writer_com.close()
        writer_pha.close()

    elapsed = time.time() - t0
    processed = min(i + 1, total_frames) if 'i' in dir() else 0

    # Print stats
    alpha_arr = np.array(alpha_values)
    print(f"\n  Results for {experiment_name}:")
    print(f"    Frames processed: {processed}")
    print(f"    Time: {elapsed:.1f}s ({elapsed/max(processed,1)*1000:.0f}ms/frame)")
    print(f"    Alpha mean: {alpha_arr.mean():.3f}")
    print(f"    Alpha >0.5: {(alpha_arr > 0.5).mean()*100:.1f}%")
    print(f"    Alpha >0.9: {(alpha_arr > 0.9).mean()*100:.1f}%")
    print(f"    Output: {comp_path}")
    print(f"    Alpha:  {alpha_path}")

    return {
        "name": experiment_name,
        "frames": processed,
        "time_s": elapsed,
        "alpha_mean": float(alpha_arr.mean()),
        "alpha_gt05": float((alpha_arr > 0.5).mean()),
        "alpha_gt09": float((alpha_arr > 0.9).mean()),
    }


def guided_filter(guide, src, radius, eps):
    """
    Fast Guided Filter (numpy, box-filter implementation).
    guide: [H, W] or [H, W, C] — the guide image (original RGB frame, [0,1])
    src:   [H, W] — the source to filter (coarse alpha, [0,1])
    radius: filter radius (pixels)
    eps: regularization
    Returns: filtered [H, W]
    """
    import cv2
    if guide.ndim == 3:
        # Convert to grayscale guide
        guide_gray = np.mean(guide, axis=2)
    else:
        guide_gray = guide

    ksize = 2 * radius + 1

    mean_I = cv2.boxFilter(guide_gray, -1, (ksize, ksize))
    mean_p = cv2.boxFilter(src, -1, (ksize, ksize))
    mean_Ip = cv2.boxFilter(guide_gray * src, -1, (ksize, ksize))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_gray * guide_gray, -1, (ksize, ksize))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    return (mean_a * guide_gray + mean_b).clip(0, 1).astype(np.float32)


def fast_guided_filter(guide_hr, src_lr, radius, eps, upsample_ratio):
    """
    Fast Guided Filter — runs box filter at LOW resolution, then upsamples coefficients.
    guide_hr: [H, W, C] or [H, W] — high-res guide (original frame, [0,1])
    src_lr:   [h, w] — low-res alpha from model output
    radius: filter radius at low-res
    eps: regularization
    upsample_ratio: guide_hr.shape / src_lr.shape ratio (e.g. 4 for ratio=0.25)
    Returns: filtered [H, W] at guide_hr resolution
    """
    import cv2
    h_hr, w_hr = guide_hr.shape[:2]
    h_lr, w_lr = src_lr.shape[:2]

    # Downsample guide to match src_lr resolution
    if guide_hr.ndim == 3:
        guide_lr = cv2.resize(guide_hr, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)
        guide_lr_gray = np.mean(guide_lr, axis=2)
        guide_hr_gray = np.mean(guide_hr, axis=2)
    else:
        guide_lr_gray = cv2.resize(guide_hr, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)
        guide_hr_gray = guide_hr

    ksize = 2 * radius + 1
    mean_I = cv2.boxFilter(guide_lr_gray, -1, (ksize, ksize))
    mean_p = cv2.boxFilter(src_lr, -1, (ksize, ksize))
    mean_Ip = cv2.boxFilter(guide_lr_gray * src_lr, -1, (ksize, ksize))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_lr_gray * guide_lr_gray, -1, (ksize, ksize))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Upsample a, b to high-res
    a_hr = cv2.resize(a, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)
    b_hr = cv2.resize(b, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)

    return (a_hr * guide_hr_gray + b_hr).clip(0, 1).astype(np.float32)


def compute_recurrent_shapes(input_h, input_w, downsample_ratio):
    """Compute ConvGRU recurrent state shapes for given input and downsample_ratio.
    Uses ceil division matching RVM's internal behavior."""
    import math
    dh = round(input_h * downsample_ratio)
    dw = round(input_w * downsample_ratio)
    r1 = [1, 16, math.ceil(dh / 2),  math.ceil(dw / 2)]
    r2 = [1, 20, math.ceil(dh / 4),  math.ceil(dw / 4)]
    r3 = [1, 40, math.ceil(dh / 8),  math.ceil(dw / 8)]
    r4 = [1, 64, math.ceil(dh / 16), math.ceil(dw / 16)]
    return r1, r2, r3, r4


def run_onnx_experiment(
    onnx_path: str,
    input_path: str,
    output_dir: str,
    experiment_name: str,
    downsample_ratio: float,
    input_h: int = 0,
    input_w: int = 0,
    apply_guided_filter: bool = False,
    gf_radius: int = 8,
    gf_eps: float = 1e-4,
    use_fast_gf: bool = True,
    max_frames: int = 0,
):
    """Run ONNX experiment with optional resolution override and GuidedFilter post-processing.

    Args:
        input_h, input_w: Model input resolution. 0 = use original video resolution.
        apply_guided_filter: Apply GuidedFilter post-processing to alpha.
        gf_radius: GuidedFilter radius.
        gf_eps: GuidedFilter regularization.
        use_fast_gf: Use fast guided filter (low-res box filter + upsample).
        max_frames: Limit frames (0=all).
    """
    import onnxruntime as ort
    import cv2

    gf_str = ""
    if apply_guided_filter:
        gf_str = f" + {'FastGF' if use_fast_gf else 'GF'}(r={gf_radius},eps={gf_eps})"

    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name} (ONNX)")
    print(f"  downsample_ratio: {downsample_ratio}")
    print(f"  input_resolution: {input_h}x{input_w}" if input_h > 0 else "  input_resolution: original")
    print(f"  post-processing: {gf_str if gf_str else 'none'}")
    print(f"  max_frames: {max_frames if max_frames > 0 else 'all'}")
    print(f"{'='*70}")

    sess = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"  ONNX inputs: {input_names}")

    # Prepare paths
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    comp_path = os.path.join(exp_dir, "composition.mp4")
    alpha_path = os.path.join(exp_dir, "alpha.mp4")

    # Read video
    cap = cv2.VideoCapture(input_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    # Determine model input size
    ret_peek, peek = cap.read()
    assert ret_peek, "Cannot read video"
    orig_h, orig_w = peek.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model_h = input_h if input_h > 0 else orig_h
    model_w = input_w if input_w > 0 else orig_w
    print(f"  Model input: {model_h}x{model_w}, Video: {orig_h}x{orig_w}")

    # Compute recurrent state shapes
    r1_shape, r2_shape, r3_shape, r4_shape = compute_recurrent_shapes(
        model_h, model_w, downsample_ratio)
    print(f"  Recurrent shapes: r1={r1_shape}, r2={r2_shape}, r3={r3_shape}, r4={r4_shape}")

    r1i = np.zeros(r1_shape, dtype=np.float32)
    r2i = np.zeros(r2_shape, dtype=np.float32)
    r3i = np.zeros(r3_shape, dtype=np.float32)
    r4i = np.zeros(r4_shape, dtype=np.float32)

    dr = np.array([downsample_ratio], dtype=np.float32)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_com = None
    writer_pha = None

    # Green screen background (matching Q1 PyTorch experiments)
    green_bg = np.array([120, 255, 155], dtype=np.float32) / 255.0  # RGB

    alpha_values = []
    t0 = time.time()

    for i in tqdm(range(total), desc=experiment_name):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess: BGR->RGB, resize, normalize to [0,1], NCHW
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_f32 = rgb.astype(np.float32) / 255.0  # [H,W,3] for GF guide

        if model_h != orig_h or model_w != orig_w:
            resized = cv2.resize(rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = rgb
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

        # Build feed dict
        feed = {'src': blob, 'r1i': r1i, 'r2i': r2i, 'r3i': r3i, 'r4i': r4i}
        if 'downsample_ratio' in input_names:
            feed['downsample_ratio'] = dr

        outputs = sess.run(None, feed)
        fgr, pha_out, r1i, r2i, r3i, r4i = outputs

        # pha_out: [1, 1, model_h, model_w]
        pha_frame = pha_out[0, 0]  # [model_h, model_w]
        fgr_frame = fgr[0].transpose(1, 2, 0)  # [model_h, model_w, 3] RGB

        # Apply GuidedFilter post-processing
        if apply_guided_filter:
            if use_fast_gf and (model_h != orig_h or model_w != orig_w):
                # Fast GF: filter at model resolution, upsample coefficients to original
                ratio = orig_h / model_h
                pha_hr = fast_guided_filter(rgb_f32, pha_frame, gf_radius, gf_eps, ratio)
                # Also upsample fgr
                fgr_hr = cv2.resize(fgr_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Standard GF at full resolution
                if model_h != orig_h or model_w != orig_w:
                    pha_up = cv2.resize(pha_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    pha_up = pha_frame
                pha_hr = guided_filter(rgb_f32, pha_up, gf_radius, gf_eps)
                fgr_hr = cv2.resize(fgr_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) \
                    if model_h != orig_h else fgr_frame
        else:
            # No GF — just bilinear upsample to original
            if model_h != orig_h or model_w != orig_w:
                pha_hr = cv2.resize(pha_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                fgr_hr = cv2.resize(fgr_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                pha_hr = pha_frame
                fgr_hr = fgr_frame

        alpha_values.append(pha_hr.mean())

        # Compose: fgr * pha + green_bg * (1-pha)
        pha_3d = pha_hr[:, :, np.newaxis]  # [H,W,1]
        comp = fgr_hr * pha_3d + green_bg * (1 - pha_3d)
        comp = (comp * 255).clip(0, 255).astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

        pha_vis = (pha_hr * 255).clip(0, 255).astype(np.uint8)

        if writer_com is None:
            writer_com = cv2.VideoWriter(comp_path, fourcc, frame_rate, (orig_w, orig_h))
            writer_pha = cv2.VideoWriter(alpha_path, fourcc, frame_rate, (orig_w, orig_h), False)

        writer_com.write(comp_bgr)
        writer_pha.write(pha_vis)

    cap.release()
    if writer_com: writer_com.release()
    if writer_pha: writer_pha.release()

    elapsed = time.time() - t0
    alpha_arr = np.array(alpha_values)
    processed = min(i + 1, total) if 'i' in dir() else 0

    print(f"\n  Results for {experiment_name}:")
    print(f"    Frames processed: {processed}")
    print(f"    Time: {elapsed:.1f}s ({elapsed/max(processed,1)*1000:.0f}ms/frame)")
    print(f"    Alpha mean: {alpha_arr.mean():.3f}")
    print(f"    Alpha >0.5: {(alpha_arr > 0.5).mean()*100:.1f}%")
    print(f"    Alpha >0.9: {(alpha_arr > 0.9).mean()*100:.1f}%")
    print(f"    Output: {comp_path}")

    return {
        "name": experiment_name,
        "frames": processed,
        "time_s": elapsed,
        "alpha_mean": float(alpha_arr.mean()),
        "alpha_gt05": float((alpha_arr > 0.5).mean()),
        "alpha_gt09": float((alpha_arr > 0.9).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="RVM Quality Investigation Q1")
    parser.add_argument("--checkpoint", required=True, help="Path to rvm_mobilenetv3.pth")
    parser.add_argument("--onnx", default=None,
                        help="Path to rvm_mobilenetv3_fp32.onnx (for ONNX experiments)")
    parser.add_argument("--input", required=True, help="Path to test video")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Max frames per experiment (0=all, default=150 for speed)")
    parser.add_argument("--experiments", nargs="+",
                        default=["Q1.1", "Q1.3", "Q1.4", "Q1.5",
                                 "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                                 "Q2.6", "Q2.7", "Q2.8", "Q2.9"],
                        help="Which experiments to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # ---- PyTorch experiments ----
    pytorch_exps = {
        # Q1.1: Golden standard — author's recommended setting
        "Q1.1": dict(downsample_ratio=0.25, disable_refiner=False),
        # Q1.2: FastGuidedFilter (TODO: need to switch refiner type in model init)
        # "Q1.2": dict(downsample_ratio=0.25, disable_refiner=False),
        # Q1.3: No Refiner — isolate how much Refiner contributes
        "Q1.3": dict(downsample_ratio=0.25, disable_refiner=True),
        # Q1.4: Medium resolution + high ratio
        "Q1.4": dict(downsample_ratio=0.5, disable_refiner=False),
        # Q1.5: ratio=1.0 — no downsampling (Refiner not triggered at ratio=1.0)
        "Q1.5": dict(downsample_ratio=1.0, disable_refiner=False),
    }

    for exp_name in args.experiments:
        if exp_name in pytorch_exps:
            cfg = pytorch_exps[exp_name]
            r = run_pytorch_experiment(
                checkpoint_path=args.checkpoint,
                input_path=args.input,
                output_dir=args.output_dir,
                experiment_name=exp_name,
                max_frames=args.max_frames,
                **cfg,
            )
            results.append(r)

    # ---- ONNX experiments ----
    # All ONNX experiments use dynamic-shape ONNX (no fixed-shape ArcFoundry export).
    # Resolution is controlled by input_h/input_w; recurrent shapes computed dynamically.
    onnx_exps = {
        # Q2.1: Current pipeline equivalent (288x512, ratio=0.25, no GF)
        "Q2.1": dict(downsample_ratio=0.25, input_h=288, input_w=512),
        # Q2.2: Higher ratio at same resolution (288x512, ratio=0.5, no GF)
        "Q2.2": dict(downsample_ratio=0.5, input_h=288, input_w=512),
        # Q2.3: Full resolution processing (288x512, ratio=1.0, no GF)
        "Q2.3": dict(downsample_ratio=1.0, input_h=288, input_w=512),
        # Q2.4: Current pipeline + FastGuidedFilter post-processing (KEY experiment)
        "Q2.4": dict(downsample_ratio=0.25, input_h=288, input_w=512,
                      apply_guided_filter=True, use_fast_gf=True),
        # Q2.5: ratio=1.0 + GF double insurance
        "Q2.5": dict(downsample_ratio=1.0, input_h=288, input_w=512,
                      apply_guided_filter=True, use_fast_gf=False),
        # Q2.6: Higher resolution input (480x640, ratio=0.25, no GF)
        "Q2.6": dict(downsample_ratio=0.25, input_h=480, input_w=640),
        # Q2.7: Higher resolution + GF (480x640, ratio=0.25, FastGF)
        "Q2.7": dict(downsample_ratio=0.25, input_h=480, input_w=640,
                      apply_guided_filter=True, use_fast_gf=True),
        # Q2.8: 1080p original resolution, ratio=0.25 (match Q1.3 via ONNX)
        "Q2.8": dict(downsample_ratio=0.25, input_h=0, input_w=0),
        # Q2.9: 1080p + FastGF (match closest to Q1.1 golden standard via ONNX+GF)
        "Q2.9": dict(downsample_ratio=0.25, input_h=0, input_w=0,
                      apply_guided_filter=True, use_fast_gf=True),
    }

    if args.onnx:
        for exp_name in args.experiments:
            if exp_name in onnx_exps:
                cfg = onnx_exps[exp_name]
                r = run_onnx_experiment(
                    onnx_path=args.onnx,
                    input_path=args.input,
                    output_dir=args.output_dir,
                    experiment_name=exp_name,
                    max_frames=args.max_frames,
                    **cfg,
                )
                results.append(r)

    # ---- Summary ----
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Experiment':<12} {'Frames':>6} {'Time':>8} {'ms/f':>6} {'α mean':>7} {'α>0.5':>6} {'α>0.9':>6}")
        print("-" * 60)
        for r in results:
            ms_per_frame = r["time_s"] / max(r["frames"], 1) * 1000
            print(f"{r['name']:<12} {r['frames']:>6} {r['time_s']:>7.1f}s {ms_per_frame:>5.0f} "
                  f"{r['alpha_mean']:>7.3f} {r['alpha_gt05']*100:>5.1f}% {r['alpha_gt09']*100:>5.1f}%")


if __name__ == "__main__":
    main()
