#!/usr/bin/env python3
"""
Board-side RVM FP16 RKNN benchmark script.
Uses ctypes to call librknnrt.so directly (no rknnlite Python package required).

Usage:
    python3 board_rvm_benchmark.py <path_to_rknn> [--warmup N] [--runs N]

Blocks 2.3 + 2.4 + 2.5:
  - Block 2.3: single-frame latency (zero recurrent state, 10 timed runs)
  - Block 2.4: 5-frame recursive sequence (r*o -> r*i propagation)
  - Block 2.5: multi-resolution benchmark (288x512, 256x256, 192x320, 144x256)
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants from rknn_api.h
# ---------------------------------------------------------------------------
RKNN_SUCC             = 0
RKNN_MAX_DIMS         = 16
RKNN_MAX_NAME_LEN     = 256

# rknn_tensor_type
RKNN_TENSOR_FLOAT32   = 0
RKNN_TENSOR_FLOAT16   = 1
RKNN_TENSOR_INT8      = 2
RKNN_TENSOR_UINT8     = 3

# rknn_tensor_qnt_type
RKNN_TENSOR_QNT_NONE  = 0

# rknn_tensor_format
RKNN_TENSOR_NCHW      = 0
RKNN_TENSOR_NHWC      = 1

# rknn_query_cmd
RKNN_QUERY_IN_OUT_NUM  = 0
RKNN_QUERY_INPUT_ATTR  = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_RUN    = 4
RKNN_QUERY_SDK_VERSION = 6

# rknn_core_mask
RKNN_NPU_CORE_0_1_2   = 7   # 1|2|4

# ---------------------------------------------------------------------------
# ctypes struct definitions
# ---------------------------------------------------------------------------

class rknn_input_output_num(ctypes.Structure):
    _fields_ = [
        ("n_input",  ctypes.c_uint32),
        ("n_output", ctypes.c_uint32),
    ]

class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index",           ctypes.c_uint32),
        ("n_dims",          ctypes.c_uint32),
        ("dims",            ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name",            ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems",         ctypes.c_uint32),
        ("size",            ctypes.c_uint32),
        ("fmt",             ctypes.c_int),   # rknn_tensor_format (enum=int)
        ("type",            ctypes.c_int),   # rknn_tensor_type   (enum=int)
        ("qnt_type",        ctypes.c_int),   # rknn_tensor_qnt_type (enum=int)
        ("fl",              ctypes.c_int8),
        ("zp",              ctypes.c_int32),
        ("scale",           ctypes.c_float),
        ("w_stride",        ctypes.c_uint32),
        ("size_with_stride",ctypes.c_uint32),
        ("pass_through",    ctypes.c_uint8),
        ("h_stride",        ctypes.c_uint32),
    ]

class rknn_perf_run(ctypes.Structure):
    _fields_ = [
        ("run_duration", ctypes.c_int64),  # microseconds
    ]

class rknn_sdk_version(ctypes.Structure):
    _fields_ = [
        ("api_version", ctypes.c_char * 256),
        ("drv_version", ctypes.c_char * 256),
    ]

class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index",        ctypes.c_uint32),
        ("buf",          ctypes.c_void_p),
        ("size",         ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type",         ctypes.c_int),   # rknn_tensor_type
        ("fmt",          ctypes.c_int),   # rknn_tensor_format
    ]

class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float",  ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index",       ctypes.c_uint32),
        ("buf",         ctypes.c_void_p),
        ("size",        ctypes.c_uint32),
    ]

# ---------------------------------------------------------------------------
# RKNN C API wrapper
# ---------------------------------------------------------------------------

class RKNN:
    """Minimal ctypes wrapper for librknnrt.so."""

    LIB_PATH = "/usr/lib/librknnrt.so"

    def __init__(self):
        self._lib = ctypes.CDLL(self.LIB_PATH)
        self._ctx = ctypes.c_uint64(0)
        self._setup_prototypes()

    def _setup_prototypes(self):
        lib = self._lib
        # rknn_init(context*, model_path_or_buf, size, flag, extend*)
        lib.rknn_init.restype  = ctypes.c_int
        lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        # rknn_destroy(context)
        lib.rknn_destroy.restype  = ctypes.c_int
        lib.rknn_destroy.argtypes = [ctypes.c_uint64]
        # rknn_query(context, cmd, info*, size)
        lib.rknn_query.restype  = ctypes.c_int
        lib.rknn_query.argtypes = [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
        ]
        # rknn_inputs_set(context, n_inputs, inputs[])
        lib.rknn_inputs_set.restype  = ctypes.c_int
        lib.rknn_inputs_set.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p
        ]
        # rknn_run(context, extend*)
        lib.rknn_run.restype  = ctypes.c_int
        lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        # rknn_outputs_get(context, n_outputs, outputs[], extend*)
        lib.rknn_outputs_get.restype  = ctypes.c_int
        lib.rknn_outputs_get.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p
        ]
        # rknn_outputs_release(context, n_outputs, outputs[])
        lib.rknn_outputs_release.restype  = ctypes.c_int
        lib.rknn_outputs_release.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p
        ]
        # rknn_set_core_mask(context, mask)
        lib.rknn_set_core_mask.restype  = ctypes.c_int
        lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]

    def init(self, rknn_path: str):
        path_bytes = rknn_path.encode("utf-8")
        ret = self._lib.rknn_init(
            ctypes.byref(self._ctx),
            ctypes.cast(ctypes.c_char_p(path_bytes), ctypes.c_void_p),
            0,   # size=0 means path mode
            0,   # flag=0 (default)
            None,
        )
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_init failed: {ret}")

    def destroy(self):
        self._lib.rknn_destroy(self._ctx)

    def set_core_mask(self, mask: int):
        ret = self._lib.rknn_set_core_mask(self._ctx, mask)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_set_core_mask failed: {ret}")

    def query_sdk_version(self) -> str:
        ver = rknn_sdk_version()
        ret = self._lib.rknn_query(
            self._ctx, RKNN_QUERY_SDK_VERSION,
            ctypes.byref(ver), ctypes.sizeof(ver)
        )
        if ret != RKNN_SUCC:
            return f"<query failed: {ret}>"
        api = ver.api_version.decode('utf-8', errors='replace').rstrip('\x00').strip()
        drv = ver.drv_version.decode('utf-8', errors='replace').rstrip('\x00').strip()
        return f"api={api}, drv={drv}"

    def query_io_num(self) -> tuple[int, int]:
        info = rknn_input_output_num()
        ret = self._lib.rknn_query(
            self._ctx, RKNN_QUERY_IN_OUT_NUM,
            ctypes.byref(info), ctypes.sizeof(info)
        )
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_query IN_OUT_NUM failed: {ret}")
        return info.n_input, info.n_output

    def query_input_attrs(self, n_inputs: int) -> list:
        attrs = []
        for i in range(n_inputs):
            attr = rknn_tensor_attr()
            attr.index = i
            ret = self._lib.rknn_query(
                self._ctx, RKNN_QUERY_INPUT_ATTR,
                ctypes.byref(attr), ctypes.sizeof(attr)
            )
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query INPUT_ATTR[{i}] failed: {ret}")
            attrs.append(attr)
        return attrs

    def query_output_attrs(self, n_outputs: int) -> list:
        attrs = []
        for i in range(n_outputs):
            attr = rknn_tensor_attr()
            attr.index = i
            ret = self._lib.rknn_query(
                self._ctx, RKNN_QUERY_OUTPUT_ATTR,
                ctypes.byref(attr), ctypes.sizeof(attr)
            )
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query OUTPUT_ATTR[{i}] failed: {ret}")
            attrs.append(attr)
        return attrs

    def query_perf_run(self) -> int:
        """Return NPU run_duration in microseconds (must call after rknn_run)."""
        perf = rknn_perf_run()
        ret = self._lib.rknn_query(
            self._ctx, RKNN_QUERY_PERF_RUN,
            ctypes.byref(perf), ctypes.sizeof(perf)
        )
        if ret != RKNN_SUCC:
            return -1
        return perf.run_duration

    def run_inference(self, inputs_np: list[np.ndarray]) -> list[np.ndarray]:
        """
        Set inputs, run, get outputs.
        inputs_np: list of float32 numpy arrays in NCHW order (N,C,H,W), matching model input order.
        Internally transposed to NHWC before sending to RKNN (model inputs are fmt=NHWC).
        Returns: list of float32 numpy arrays; shape matches model output attrs (typically NCHW).
        """
        n_in = len(inputs_np)
        InputArray = rknn_input * n_in
        inp_arr = InputArray()
        # Keep references alive so GC doesn't collect contiguous buffers.
        _bufs = []
        for i, arr in enumerate(inputs_np):
            # Model inputs are NHWC (fmt=1); transpose caller's NCHW (N,C,H,W) → (N,H,W,C).
            c_arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1), dtype=np.float32)
            _bufs.append(c_arr)
            inp_arr[i].index        = i
            inp_arr[i].buf          = c_arr.ctypes.data_as(ctypes.c_void_p)
            inp_arr[i].size         = c_arr.nbytes
            inp_arr[i].pass_through = 0
            inp_arr[i].type         = RKNN_TENSOR_FLOAT32
            inp_arr[i].fmt          = RKNN_TENSOR_NHWC

        ret = self._lib.rknn_inputs_set(self._ctx, n_in, inp_arr)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_inputs_set failed: {ret}")

        ret = self._lib.rknn_run(self._ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed: {ret}")

        n_out, out_attrs = self._n_out, self._out_attrs
        OutputArray = rknn_output * n_out
        out_arr = OutputArray()
        for i in range(n_out):
            out_arr[i].want_float  = 1
            out_arr[i].is_prealloc = 0
            out_arr[i].index       = i

        ret = self._lib.rknn_outputs_get(self._ctx, n_out, out_arr, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_outputs_get failed: {ret}")

        outputs = []
        for i in range(n_out):
            attr = out_attrs[i]
            shape = list(attr.dims[:attr.n_dims])
            total_elems = 1
            for d in shape:
                total_elems *= d
            raw = (ctypes.c_float * total_elems).from_address(out_arr[i].buf)
            outputs.append(np.array(raw, dtype=np.float32).reshape(shape))

        self._lib.rknn_outputs_release(self._ctx, n_out, out_arr)
        return outputs

    def prepare(self, rknn_path: str):
        """Full init: load model, set 3-NPU-core mask, cache io attrs."""
        self.init(rknn_path)
        self.set_core_mask(RKNN_NPU_CORE_0_1_2)
        n_in, n_out = self.query_io_num()
        self._n_in    = n_in
        self._n_out   = n_out
        self._in_attrs  = self.query_input_attrs(n_in)
        self._out_attrs = self.query_output_attrs(n_out)
        return n_in, n_out

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def get_recurrent_shapes(H: int, W: int) -> dict[str, tuple]:
    """Return the 4 recurrent state shapes for a given src resolution."""
    import math
    return {
        "r1": (1, 16, math.ceil(H / 2),  math.ceil(W / 2)),
        "r2": (1, 20, math.ceil(H / 4),  math.ceil(W / 4)),
        "r3": (1, 40, math.ceil(H / 8),  math.ceil(W / 8)),
        "r4": (1, 64, math.ceil(H / 16), math.ceil(W / 16)),
    }

def make_dummy_inputs(H: int, W: int) -> list[np.ndarray]:
    """
    Create dummy float32 inputs: [src, r1i, r2i, r3i, r4i].
    src is a uniform grey image; recurrent states are zeros.
    """
    src  = np.full((1, 3, H, W), 128.0, dtype=np.float32)  # 0-255 range, mid-grey
    recs = get_recurrent_shapes(H, W)
    r1i  = np.zeros(recs["r1"], dtype=np.float32)
    r2i  = np.zeros(recs["r2"], dtype=np.float32)
    r3i  = np.zeros(recs["r3"], dtype=np.float32)
    r4i  = np.zeros(recs["r4"], dtype=np.float32)
    return [src, r1i, r2i, r3i, r4i]

# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def print_model_info(rknn_obj: RKNN):
    print(f"\n[Model Info]")
    print(f"  SDK: {rknn_obj.query_sdk_version()}")
    n_in, n_out = rknn_obj._n_in, rknn_obj._n_out
    print(f"  Inputs  : {n_in}")
    for a in rknn_obj._in_attrs:
        shape = list(a.dims[:a.n_dims])
        print(f"    [{a.index}] {a.name.decode():16s}  shape={shape}  type={a.type}  fmt={a.fmt}")
    print(f"  Outputs : {n_out}")
    for a in rknn_obj._out_attrs:
        shape = list(a.dims[:a.n_dims])
        print(f"    [{a.index}] {a.name.decode():16s}  shape={shape}  type={a.type}  fmt={a.fmt}")


def block_2_3_single_frame(rknn_obj: RKNN, H: int, W: int, warmup: int, runs: int):
    """Block 2.3 — single-frame latency with zero recurrent state."""
    print(f"\n[Block 2.3] Single-frame benchmark  H={H} W={W}  warmup={warmup} runs={runs}")
    inputs = make_dummy_inputs(H, W)

    # Warmup
    for _ in range(warmup):
        rknn_obj.run_inference(inputs)

    # Timed runs (wall-clock)
    wall_times = []
    for i in range(runs):
        t0 = time.perf_counter()
        rknn_obj.run_inference(inputs)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    median = wall_times[len(wall_times) // 2]
    mean   = sum(wall_times) / len(wall_times)
    print(f"  Wall-clock (ms): mean={mean:.1f}  median={median:.1f}  "
          f"min={wall_times[0]:.1f}  max={wall_times[-1]:.1f}")
    return mean, median


def block_2_4_recursive_sequence(rknn_obj: RKNN, H: int, W: int, frames: int = 5):
    """Block 2.4 — N-frame recursive state propagation."""
    print(f"\n[Block 2.4] Recursive sequence  H={H} W={W}  frames={frames}")
    inputs = make_dummy_inputs(H, W)

    for frame_idx in range(frames):
        outputs = rknn_obj.run_inference(inputs)
        # outputs order: [fgr, pha, r1o, r2o, r3o, r4o]
        # Propagate recurrent state: r*o -> r*i
        # inputs order:  [src, r1i, r2i, r3i, r4i]
        inputs[1] = outputs[2].copy()  # r1i <- r1o
        inputs[2] = outputs[3].copy()  # r2i <- r2o
        inputs[3] = outputs[4].copy()  # r3i <- r3o
        inputs[4] = outputs[5].copy()  # r4i <- r4o

        fgr = outputs[0]
        pha = outputs[1]
        has_nan = np.any(np.isnan(fgr)) or np.any(np.isnan(pha))
        pha_range = (float(pha.min()), float(pha.max()))
        print(f"  Frame {frame_idx+1}: fgr shape={list(fgr.shape)}  "
              f"pha shape={list(pha.shape)}  pha_range=[{pha_range[0]:.4f}, {pha_range[1]:.4f}]  "
              f"has_nan={has_nan}")

    print("  [PASS] Recursive state propagation completed without error.")


def block_2_5_multi_res(rknn_obj: RKNN, warmup: int, runs: int):
    """Block 2.5 — multi-resolution benchmark."""
    resolutions = [
        (288, 512, 80.0),   # target <80ms
        (256, 256, 40.0),   # target <40ms
        (192, 320, 50.0),   # reference
        (144, 256, 30.0),   # reference
    ]
    print(f"\n[Block 2.5] Multi-resolution benchmark  warmup={warmup} runs={runs}")
    print(f"  {'Resolution':>14}  {'mean(ms)':>10}  {'median(ms)':>10}  {'target(ms)':>10}  {'PASS?':>6}")
    results = []
    for H, W, tgt in resolutions:
        # Re-use the same model with different input sizes.
        # RVM is dynamic-shape: the RKNN model was fixed to 288x512 at build time.
        # Only 288x512 is guaranteed; others are informational.
        inputs = make_dummy_inputs(H, W)
        # Warmup
        try:
            for _ in range(warmup):
                rknn_obj.run_inference(inputs)
            wall_times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                rknn_obj.run_inference(inputs)
                t1 = time.perf_counter()
                wall_times.append((t1 - t0) * 1000.0)
            mean   = sum(wall_times) / len(wall_times)
            median = sorted(wall_times)[len(wall_times) // 2]
            ok = "✓" if mean < tgt else "✗"
            print(f"  {H}x{W:>3}         {mean:>10.1f}  {median:>10.1f}  {tgt:>10.1f}  {ok:>6}")
            results.append((H, W, mean, median, tgt))
        except Exception as e:
            print(f"  {H}x{W:>3}  FAILED: {e}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RVM FP16 RKNN board benchmark")
    parser.add_argument("rknn_path", help="Path to .rknn model file")
    parser.add_argument("--warmup", type=int, default=3,   help="Warmup runs (default 3)")
    parser.add_argument("--runs",   type=int, default=10,  help="Timed runs (default 10)")
    parser.add_argument("--frames", type=int, default=5,   help="Frames for Block 2.4 (default 5)")
    parser.add_argument("--H",      type=int, default=288, help="Primary benchmark height (default 288)")
    parser.add_argument("--W",      type=int, default=512, help="Primary benchmark width (default 512)")
    parser.add_argument("--skip-multires", action="store_true",
                        help="Skip Block 2.5 multi-resolution benchmark")
    args = parser.parse_args()

    if not os.path.exists(args.rknn_path):
        print(f"ERROR: model not found: {args.rknn_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  RVM MobileNetV3 FP16 — RKNN Board Benchmark")
    print("=" * 60)

    rknn = RKNN()
    print(f"\nLoading model: {args.rknn_path}")
    n_in, n_out = rknn.prepare(args.rknn_path)
    print_model_info(rknn)

    # Block 2.3: single-frame @ primary resolution
    block_2_3_single_frame(rknn, H=args.H, W=args.W, warmup=args.warmup, runs=args.runs)

    # Block 2.4: recursive sequence @ primary resolution
    block_2_4_recursive_sequence(rknn, H=args.H, W=args.W, frames=args.frames)

    # Block 2.5: multi-resolution
    if not args.skip_multires:
        block_2_5_multi_res(rknn, warmup=args.warmup, runs=args.runs)

    rknn.destroy()
    print("\n[Done]")


if __name__ == "__main__":
    main()
