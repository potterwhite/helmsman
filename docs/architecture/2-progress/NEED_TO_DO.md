- **改完就把下面的checkbox checked**

Mar30.2026 18:00
- [ ] Block 1.4: Export `modnet_bn_best.ckpt` → ONNX using `export_onnx_modified.py`
- [ ] Block 1.4: Verify in Netron — confirm zero `InstanceNormalization` nodes
- [ ] Block 1.4: Generate golden files via `./helmsman golden`
- [ ] Block 1.4: Build C++ native (`./helmsman build cpp cb`) and run inference
- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)

Mar30.2026 15:00
- [x] Set up AI-first documentation system (CLAUDE.md, guide.md, codebase_map.md, progress.md, NEED_TO_DO.md, 00_INDEX.md, architecture_vision.md)
- [x] Reorganize docs into `docs/architecture/` directory structure per template
- [x] Shrink CLAUDE.md from 347 lines to ≤60 lines
