# NEED_TO_DO Archived — Mar31.2026

Mar31.2026
- [x] Block 1.4: Export `modnet_bn_best.ckpt` → ONNX using `export_onnx_pureBN.py`
    commit: 554ca64
    Note: Used export_onnx_pureBN.py (not export_onnx_modified.py) — Pure-BN ckpt
    key format incompatible with IBNorm-based modified script. Required sys.path
    surgery to prevent local MODNet.git/onnx/ from shadowing pip onnx package.
    Also upgraded onnx 1.8.1 → 1.14.1 (torch 2.0.1 requires onnx ≥ 1.13).
- [x] Block 1.4: Verify — confirm zero `InstanceNormalization` nodes
    commit: 554ca64
    Result: 0 InstanceNorm nodes, 0 BN nodes (folded into Conv), opset 11, 258 nodes total.
