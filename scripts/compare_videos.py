#!/usr/bin/env python3
# Copyright (c) 2026 PotterWhite — MIT License
"""
compare_videos.py — Helmsman Video Quality Comparison Tool

用法（直接运行）：
    python3 scripts/compare_videos.py

用法（通过 helmsman）：
    ./helmsman compare [--video-a PATH] [--video-b PATH] [--label-a STR] [--label-b STR]
                      [--out DIR] [--frames N1,N2,...] [--no-interactive]

功能：
  1. 对比两个视频的合成品质（逐帧 mean_diff、绿色背景占比）
  2. 生成 side-by-side 对比图（每个采样帧一张）
  3. 汇总统计表（print + 保存 summary.txt）

输出目录默认值：
    build/video_compare_out/
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ===========================================================================
# Background colour used for alpha compositing (must match pipeline default)
# BGR: (155, 255, 120)  =  RGB: (120, 255, 155)
# ===========================================================================
DEFAULT_BG_BGR = (155, 255, 120)
DEFAULT_OUT_DIR = "build/video_compare_out"
DEFAULT_SAMPLE_FRAMES = [0, 45, 90, 180, 270, 362]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_cap(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def _video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def _bg_ratio(frame: np.ndarray, bg_bgr, threshold: float = 15.0) -> float:
    """Return fraction of pixels within `threshold` of the background colour."""
    bg = np.array(bg_bgr, dtype=np.float32)
    d = np.abs(frame.astype(np.float32) - bg[None, None, :]).mean(axis=2)
    return float((d < threshold).mean() * 100.0)


def _save_side_by_side(f_a: np.ndarray, f_b: np.ndarray,
                       label_a: str, label_b: str,
                       frame_idx: int, out_dir: str) -> str:
    scale = 0.5
    h_a = int(f_a.shape[0] * scale)
    w_a = int(f_a.shape[1] * scale)
    r = cv2.resize(f_a, (w_a, h_a))
    p = cv2.resize(f_b, (w_a, h_a))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(r, f"{label_a}  frame {frame_idx:04d}", (12, 44), font, 1.0, (0, 255, 0), 2)
    cv2.putText(p, f"{label_b}  frame {frame_idx:04d}", (12, 44), font, 1.0, (0, 255, 255), 2)
    side = np.hstack([r, p])
    fname = os.path.join(out_dir, f"frame{frame_idx:04d}_{label_a}_vs_{label_b}.jpg")
    cv2.imwrite(fname, side, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return fname


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compare(
    path_a: str,
    path_b: str,
    label_a: str = "video_a",
    label_b: str = "video_b",
    out_dir: str = DEFAULT_OUT_DIR,
    sample_frames: list = None,
    bg_bgr: tuple = DEFAULT_BG_BGR,
) -> dict:
    """
    Run full comparison between two videos.

    Returns dict with keys:
        n_frames, bg_ratios_a, bg_ratios_b, diffs, saved_images, summary_path
    """
    if sample_frames is None:
        sample_frames = DEFAULT_SAMPLE_FRAMES

    os.makedirs(out_dir, exist_ok=True)

    cap_a = _open_cap(path_a)
    cap_b = _open_cap(path_b)
    info_a = _video_info(cap_a)
    info_b = _video_info(cap_b)

    print(f"\n{'='*60}")
    print(f"  Video A ({label_a}): {info_a['width']}x{info_a['height']} "
          f"@ {info_a['fps']:.3f} fps, {info_a['frames']} frames")
    print(f"  Video B ({label_b}): {info_b['width']}x{info_b['height']} "
          f"@ {info_b['fps']:.3f} fps, {info_b['frames']} frames")
    print(f"{'='*60}")

    n_frames = min(info_a["frames"], info_b["frames"])
    bg_ratios_a = []
    bg_ratios_b = []
    diffs = []
    sample_data = {}

    for i in range(n_frames):
        r1, f1 = cap_a.read()
        r2, f2 = cap_b.read()
        if not r1 or not r2:
            print(f"  [warn] EOF at frame {i}")
            break
        bg_ratios_a.append(_bg_ratio(f1, bg_bgr))
        bg_ratios_b.append(_bg_ratio(f2, bg_bgr))
        diffs.append(float(np.abs(f1.astype(np.float32) - f2.astype(np.float32)).mean()))
        if i in sample_frames:
            sample_data[i] = (f1.copy(), f2.copy())

    cap_a.release()
    cap_b.release()

    bg_ratios_a = np.array(bg_ratios_a)
    bg_ratios_b = np.array(bg_ratios_b)
    diffs = np.array(diffs)

    # --- Per-sample-frame side-by-side images ---
    saved_images = []
    for idx, (fa, fb) in sample_data.items():
        fname = _save_side_by_side(fa, fb, label_a, label_b, idx, out_dir)
        saved_images.append(fname)
        print(f"  Saved: {fname}")

    # --- Summary ---
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  COMPARISON SUMMARY")
    lines.append(f"    A = {label_a}  ({path_a})")
    lines.append(f"    B = {label_b}  ({path_b})")
    lines.append(f"  Frames analysed: {len(diffs)}")
    lines.append(f"{'='*60}")
    lines.append(f"  bg_ratio A (green bg %)  avg={bg_ratios_a.mean():.1f}  "
                 f"min={bg_ratios_a.min():.1f}  max={bg_ratios_a.max():.1f}")
    lines.append(f"  bg_ratio B (green bg %)  avg={bg_ratios_b.mean():.1f}  "
                 f"min={bg_ratios_b.min():.1f}  max={bg_ratios_b.max():.1f}")
    lines.append(f"  mean_diff (A vs B)       avg={diffs.mean():.2f}  "
                 f"min={diffs.min():.2f}  max={diffs.max():.2f}")
    lines.append(f"  bg_ratio note: 100% = all green (person invisible); "
                 f"~85% = person present (ONNX-Py baseline)")
    lines.append(f"{'='*60}")
    lines.append(f"  Per sample frame bg_ratio_A / bg_ratio_B:")
    for i in sorted(sample_data.keys()):
        lines.append(f"    frame {i:4d}:  A={bg_ratios_a[i]:.1f}%  B={bg_ratios_b[i]:.1f}%  "
                     f"diff={diffs[i]:.2f}")
    lines.append(f"{'='*60}\n")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Video A: {path_a}\nVideo B: {path_b}\n\n")
        f.write(summary_text)
    print(f"  Summary saved: {summary_path}")

    return {
        "n_frames":     len(diffs),
        "bg_ratios_a":  bg_ratios_a,
        "bg_ratios_b":  bg_ratios_b,
        "diffs":        diffs,
        "saved_images": saved_images,
        "summary_path": summary_path,
    }


# ---------------------------------------------------------------------------
# Interactive prompts (used when args are missing)
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    if default:
        ans = input(f"  {prompt} [{default}]: ").strip()
        return ans if ans else default
    ans = input(f"  {prompt}: ").strip()
    return ans


def _interactive_run():
    print("\n=== Helmsman: Video Quality Comparison Tool ===\n")
    path_a  = _ask("Video A path")
    label_a = _ask("Label for A", os.path.basename(os.path.dirname(path_a)) or "video_a")
    path_b  = _ask("Video B path (reference)")
    label_b = _ask("Label for B", os.path.basename(os.path.dirname(path_b)) or "video_b")
    out_dir = _ask("Output folder", DEFAULT_OUT_DIR)

    frames_str = _ask(
        "Sample frame indices (comma-separated)",
        ",".join(str(x) for x in DEFAULT_SAMPLE_FRAMES),
    )
    sample_frames = [int(x.strip()) for x in frames_str.split(",") if x.strip().isdigit()]

    compare(path_a, path_b, label_a, label_b, out_dir, sample_frames)

    # Print equivalent CLI command for future use
    cli_parts = [
        "./helmsman compare video",
        f"--video-a '{path_a}'",
        f"--video-b '{path_b}'",
        f"--label-a '{label_a}'",
        f"--label-b '{label_b}'",
    ]
    if out_dir != DEFAULT_OUT_DIR:
        cli_parts.append(f"--out '{out_dir}'")
    if frames_str != ",".join(str(x) for x in DEFAULT_SAMPLE_FRAMES):
        cli_parts.append(f"--frames '{frames_str}'")
    print(f"\n  Tip: Next time you can run this directly:")
    print(f"    {' '.join(cli_parts)}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare two composited output videos (side-by-side + stats)."
    )
    p.add_argument("--video-a",  default="", help="Path to video A (under test)")
    p.add_argument("--video-b",  default="", help="Path to video B (reference)")
    p.add_argument("--label-a",  default="test", help="Label for video A")
    p.add_argument("--label-b",  default="ref",  help="Label for video B")
    p.add_argument("--out",      default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument("--frames",   default="", help="Sample frame indices, comma-separated")
    p.add_argument("--no-interactive", action="store_true",
                   help="Fail instead of prompting if paths are missing")
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.video_a or not args.video_b:
        if args.no_interactive:
            print("ERROR: --video-a and --video-b are required with --no-interactive",
                  file=sys.stderr)
            sys.exit(1)
        _interactive_run()
        return

    sample_frames = DEFAULT_SAMPLE_FRAMES
    if args.frames:
        sample_frames = [int(x.strip()) for x in args.frames.split(",") if x.strip().isdigit()]

    compare(args.video_a, args.video_b, args.label_a, args.label_b, args.out, sample_frames)


if __name__ == "__main__":
    main()
