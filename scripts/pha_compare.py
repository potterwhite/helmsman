#!/usr/bin/env python3
# Copyright (c) 2026 PotterWhite — MIT License
"""
pha_compare.py — Helmsman Pha Mean Curve Comparison Tool

Compares per-frame alpha matte (pha) mean values between two sources:
  - C++ run.log (PHA_MEAN lines from RKNN/C++ backend)
  - ONNX-Python inference (live baseline)
  - Another run.log (log-vs-log)

Usage (direct):
    python3 scripts/pha_compare.py --log-a run.log --label-a rknn

Usage (via helmsman):
    ./helmsman compare pha --log-a run.log
    ./helmsman compare pha --log-a run_a.log --log-b run_b.log
    ./helmsman compare pha --log-a run.log --onnx-model model.onnx --video dance.mp4
"""

import argparse
import math
import os
import re
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = "build/pha_compare_out"
DEFAULT_SAMPLE_FRAMES = [50, 55, 60, 62, 65, 70]
PHA_MEAN_RE = re.compile(r"PHA_MEAN\s+frame=(\d+)\s+mean=([\d.]+)")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_pha_mean_log(log_path: str) -> dict:
    """Parse PHA_MEAN lines from a C++ run.log.

    Expected line format:
        PHA_MEAN frame=N mean=F.FFFF

    Returns:
        dict[int, float] — {frame_idx: pha_mean} where mean is 0-1 scale.
    """
    result = {}
    with open(log_path, "r") as f:
        for line in f:
            m = PHA_MEAN_RE.search(line)
            if m:
                frame_idx = int(m.group(1))
                mean_val = float(m.group(2))
                result[frame_idx] = mean_val
    return result


# ---------------------------------------------------------------------------
# ONNX-Py baseline inference
# ---------------------------------------------------------------------------

def compute_recurrent_shapes(input_h, input_w, downsample_ratio):
    """Compute ConvGRU recurrent state shapes for given input and downsample_ratio."""
    dh = round(input_h * downsample_ratio)
    dw = round(input_w * downsample_ratio)
    r1 = [1, 16, math.ceil(dh / 2),  math.ceil(dw / 2)]
    r2 = [1, 20, math.ceil(dh / 4),  math.ceil(dw / 4)]
    r3 = [1, 40, math.ceil(dh / 8),  math.ceil(dw / 8)]
    r4 = [1, 64, math.ceil(dh / 16), math.ceil(dw / 16)]
    return r1, r2, r3, r4


def run_onnx_baseline(onnx_path: str, video_path: str,
                       downsample_ratio: float = 0.25,
                       max_frames: int = 0) -> dict:
    """Run ONNX-RVM inference on a video and return per-frame pha means.

    Returns:
        dict[int, float] — {frame_idx: pha_mean} in 0-1 scale.
    """
    import onnxruntime as ort

    print(f"\n  Running ONNX baseline: {onnx_path}")
    print(f"  Video: {video_path}")
    print(f"  downsample_ratio: {downsample_ratio}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    # Peek first frame for dimensions
    ret_peek, peek = cap.read()
    if not ret_peek:
        raise RuntimeError("Cannot read first frame from video")
    orig_h, orig_w = peek.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Pad to multiples of 32 (matching RVM convention)
    pad_h = (32 - orig_h % 32) % 32
    pad_w = (32 - orig_w % 32) % 32
    model_h = orig_h + pad_h
    model_w = orig_w + pad_w

    r1s, r2s, r3s, r4s = compute_recurrent_shapes(model_h, model_w, downsample_ratio)
    r1i = np.zeros(r1s, dtype=np.float32)
    r2i = np.zeros(r2s, dtype=np.float32)
    r3i = np.zeros(r3s, dtype=np.float32)
    r4i = np.zeros(r4s, dtype=np.float32)
    dr = np.array([downsample_ratio], dtype=np.float32)

    result = {}
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB, pad, normalize, NCHW
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if pad_h > 0 or pad_w > 0:
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        feed = {"src": blob, "r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i}
        if "downsample_ratio" in input_names:
            feed["downsample_ratio"] = dr

        outputs = sess.run(None, feed)
        fgr_out, pha_out = outputs[0], outputs[1]
        r1i, r2i, r3i, r4i = outputs[2], outputs[3], outputs[4], outputs[5]

        # Crop padding, compute mean
        pha_crop = pha_out[0, 0, :orig_h, :orig_w]
        result[i] = float(pha_crop.mean())

    cap.release()
    print(f"  ONNX baseline done: {len(result)} frames")
    return result


# ---------------------------------------------------------------------------
# Comparison & output
# ---------------------------------------------------------------------------

def compare_pha_curves(
    data_a: dict,
    data_b: dict,
    label_a: str = "A",
    label_b: str = "B",
    out_dir: str = DEFAULT_OUT_DIR,
    sample_frames: list = None,
    no_plot: bool = False,
) -> dict:
    """Compare two pha mean curves and produce summary + optional chart.

    Args:
        data_a, data_b: {frame_idx: pha_mean} dicts (0-1 scale).
        label_a, label_b: Human-readable labels for the two sources.
        out_dir: Output directory for summary.txt and charts.
        sample_frames: Frame indices for detailed per-frame dump.
        no_plot: Skip matplotlib chart.

    Returns:
        dict with keys: n_frames, mean_diff, max_diff, std_diff, diffs, summary_path
    """
    if sample_frames is None:
        sample_frames = DEFAULT_SAMPLE_FRAMES

    os.makedirs(out_dir, exist_ok=True)

    # Align on common frames
    common_frames = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not common_frames:
        print("  ERROR: No common frames between A and B.")
        return {}

    vals_a = np.array([data_a[f] for f in common_frames])
    vals_b = np.array([data_b[f] for f in common_frames])
    diffs = vals_a - vals_b  # positive = A stronger alpha

    # --- Summary ---
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  PHA MEAN COMPARISON SUMMARY")
    lines.append(f"    A = {label_a}")
    lines.append(f"    B = {label_b}")
    lines.append(f"  Common frames: {len(common_frames)}")
    lines.append(f"{'='*65}")
    lines.append(f"  pha mean A   avg={vals_a.mean():.4f}  min={vals_a.min():.4f}  max={vals_a.max():.4f}")
    lines.append(f"  pha mean B   avg={vals_b.mean():.4f}  min={vals_b.min():.4f}  max={vals_b.max():.4f}")
    lines.append(f"  diff (A-B)   avg={diffs.mean():.4f}  std={diffs.std():.4f}  "
                 f"min={diffs.min():.4f}  max={diffs.max():.4f}")
    lines.append(f"  abs_diff     avg={np.abs(diffs).mean():.4f}  max={np.abs(diffs).max():.4f}")
    lines.append(f"{'='*65}")

    # Per-sample-frame detail
    lines.append(f"  Per sample frame detail (pha mean, 0-1 scale):")
    for sf in sample_frames:
        if sf in data_a and sf in data_b:
            d = data_a[sf] - data_b[sf]
            lines.append(f"    frame {sf:4d}:  A={data_a[sf]:.4f}  B={data_b[sf]:.4f}  diff={d:+.4f}")
        elif sf in data_a:
            lines.append(f"    frame {sf:4d}:  A={data_a[sf]:.4f}  B=---")
        elif sf in data_b:
            lines.append(f"    frame {sf:4d}:  A=---  B={data_b[sf]:.4f}")
    lines.append(f"{'='*65}\n")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Source A: {label_a}\n")
        f.write(f"Source B: {label_b}\n\n")
        f.write(summary_text)
    print(f"  Summary saved: {summary_path}")

    # --- Matplotlib chart ---
    if not no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.suptitle(f"Pha Mean Comparison: {label_a} vs {label_b}", fontsize=13)

            ax1.plot(common_frames, vals_a, label=label_a, linewidth=0.8)
            ax1.plot(common_frames, vals_b, label=label_b, linewidth=0.8)
            ax1.set_ylabel("pha mean (0-1)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.bar(common_frames, diffs, width=1.0, color="steelblue", alpha=0.7)
            ax2.axhline(y=0, color="red", linewidth=0.5)
            ax2.axhline(y=diffs.mean(), color="orange", linewidth=0.8, linestyle="--",
                        label=f"mean={diffs.mean():.4f}")
            ax2.set_xlabel("frame index")
            ax2.set_ylabel(f"diff ({label_a} - {label_b})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            chart_path = os.path.join(out_dir, "pha_curves.png")
            fig.tight_layout()
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)
            print(f"  Chart saved: {chart_path}")
        except ImportError:
            print("  matplotlib not available, skipping chart.")

    return {
        "n_frames": len(common_frames),
        "mean_diff": float(diffs.mean()),
        "max_diff": float(np.abs(diffs).max()),
        "std_diff": float(diffs.std()),
        "diffs": diffs,
        "summary_path": summary_path,
    }


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    if default:
        ans = input(f"  {prompt} [{default}]: ").strip()
        return ans if ans else default
    ans = input(f"  {prompt}: ").strip()
    return ans


def _interactive_run():
    print("\n=== Helmsman: Pha Mean Curve Comparison Tool ===\n")
    print("  Comparison modes:")
    print("    1. Log vs Log   — compare two C++ run.log files")
    print("    2. Log vs ONNX  — compare C++ run.log against ONNX-Py baseline")
    mode = _ask("Select mode [1/2]", "1")

    label_a = ""
    label_b = ""
    data_a = {}
    data_b = {}
    cli_parts = ["./helmsman compare pha"]

    if mode == "1":
        log_a = _ask("Path to run.log A")
        label_a = _ask("Label for A", os.path.basename(os.path.dirname(log_a)) or "log_a")
        log_b = _ask("Path to run.log B")
        label_b = _ask("Label for B", os.path.basename(os.path.dirname(log_b)) or "log_b")
        data_a = parse_pha_mean_log(log_a)
        data_b = parse_pha_mean_log(log_b)
        cli_parts.append(f"--log-a '{log_a}'")
        cli_parts.append(f"--log-b '{log_b}'")
        cli_parts.append(f"--label-a '{label_a}'")
        cli_parts.append(f"--label-b '{label_b}'")
    elif mode == "2":
        log_a = _ask("Path to run.log (C++ / RKNN)")
        label_a = _ask("Label for log", "rknn")
        onnx_path = _ask("Path to ONNX model")
        video_path = _ask("Path to input video")
        label_b = _ask("Label for ONNX baseline", "onnx-py")
        data_a = parse_pha_mean_log(log_a)
        data_b = run_onnx_baseline(onnx_path, video_path)
        cli_parts.append(f"--log-a '{log_a}'")
        cli_parts.append(f"--onnx-model '{onnx_path}'")
        cli_parts.append(f"--video '{video_path}'")
        cli_parts.append(f"--label-a '{label_a}'")
        cli_parts.append(f"--label-b '{label_b}'")
    else:
        print("  Invalid mode.")
        return

    out_dir = _ask("Output folder", DEFAULT_OUT_DIR)
    no_plot_str = _ask("Skip matplotlib chart? [y/N]", "n")
    no_plot = no_plot_str.lower() in ("y", "yes")

    if out_dir != DEFAULT_OUT_DIR:
        cli_parts.append(f"--out '{out_dir}'")
    if no_plot:
        cli_parts.append("--no-plot")

    compare_pha_curves(data_a, data_b, label_a, label_b, out_dir, no_plot=no_plot)

    # Print equivalent CLI command for future use
    cli_cmd = " ".join(cli_parts)
    print(f"\n  Tip: Next time you can run this directly:")
    print(f"    {cli_cmd}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare per-frame pha mean values from C++ logs or ONNX inference."
    )
    p.add_argument("--log-a", default="",
                   help="Path to run.log A (contains PHA_MEAN lines)")
    p.add_argument("--log-b", default="",
                   help="Path to run.log B (for log-vs-log comparison)")
    p.add_argument("--label-a", default="rknn",
                   help="Label for source A (default: rknn)")
    p.add_argument("--label-b", default="onnx-py",
                   help="Label for source B (default: onnx-py)")
    p.add_argument("--onnx-model", default="",
                   help="Path to ONNX model (for live ONNX-Py baseline)")
    p.add_argument("--video", default="",
                   help="Path to input video (required with --onnx-model)")
    p.add_argument("--downsample-ratio", type=float, default=0.25,
                   help="RVM downsample ratio (default: 0.25)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Limit frames for ONNX inference (0=all)")
    p.add_argument("--out", default=DEFAULT_OUT_DIR,
                   help="Output directory")
    p.add_argument("--frames", default="",
                   help="Sample frame indices, comma-separated")
    p.add_argument("--no-interactive", action="store_true",
                   help="Fail instead of prompting when args are missing")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib chart generation")
    return p.parse_args()


def main():
    args = _parse_args()

    # Determine data sources
    data_a = {}
    data_b = {}
    label_a = args.label_a
    label_b = args.label_b

    if args.log_a:
        data_a = parse_pha_mean_log(args.log_a)
        print(f"  Parsed {len(data_a)} PHA_MEAN frames from: {args.log_a}")
    elif not args.no_interactive:
        _interactive_run()
        return
    else:
        print("ERROR: --log-a is required with --no-interactive", file=sys.stderr)
        sys.exit(1)

    if args.log_b:
        # Log-vs-log mode
        data_b = parse_pha_mean_log(args.log_b)
        print(f"  Parsed {len(data_b)} PHA_MEAN frames from: {args.log_b}")
    elif args.onnx_model:
        # Log-vs-ONNX mode
        if not args.video:
            print("ERROR: --video is required with --onnx-model", file=sys.stderr)
            sys.exit(1)
        data_b = run_onnx_baseline(args.onnx_model, args.video,
                                    args.downsample_ratio, args.max_frames)
    elif not args.no_interactive:
        # Ask user what to compare against
        print("\n  No --log-b or --onnx-model specified.")
        print("    1. Provide another run.log")
        print("    2. Run ONNX-Py baseline")
        choice = _ask("Select [1/2]", "2")
        if choice == "1":
            log_b = _ask("Path to run.log B")
            label_b = _ask("Label for B", label_b)
            data_b = parse_pha_mean_log(log_b)
        elif choice == "2":
            onnx_path = _ask("Path to ONNX model")
            video_path = _ask("Path to input video")
            label_b = _ask("Label for ONNX baseline", label_b)
            data_b = run_onnx_baseline(onnx_path, video_path,
                                        args.downsample_ratio, args.max_frames)
        else:
            print("  Invalid choice.")
            return
    else:
        print("ERROR: --log-b or --onnx-model required with --no-interactive", file=sys.stderr)
        sys.exit(1)

    sample_frames = DEFAULT_SAMPLE_FRAMES
    if args.frames:
        sample_frames = [int(x.strip()) for x in args.frames.split(",") if x.strip().isdigit()]

    compare_pha_curves(data_a, data_b, label_a, label_b, args.out, sample_frames, args.no_plot)


if __name__ == "__main__":
    main()
