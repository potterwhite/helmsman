- **改完就把下面的checkbox checked**
**并挪到task-logs/目录下面的解决当日的md里，例如：NEED_TO_DO_ARCHIVED_Mar28.2026，不断Append到相应md的末尾，并附注commit hash**

Mar30.2026 16:20
- [x] 计划都留到2-progress；
    给ai agent 的东西都留在1-for-ai；
    亮点，推广的东西留在3-highlights；
    readme，quick-start（目前没有）等都留在4-for-beginer
- [x] 按着这个结构，以zh为主（除非ai读取zh会有明显信息的损失，词不达意等），en完全为zh的的一一对应版本；
    每一个en的md里都有一个url可以jump到zh的md里
- [x] 每个语言里的assets只有一张，是不同的；
    你需要修改readme，去适配正确的背景图片。
    commit: dd91b67



Mar30.2026 15:00
- [x] Set up AI-first documentation system (CLAUDE.md, guide.md, codebase_map.md, progress.md, NEED_TO_DO.md, 00_INDEX.md, architecture_vision.md)
- [x] Reorganize docs into `docs/architecture/` directory structure per template
- [x] Shrink CLAUDE.md from 347 lines to ≤60 lines
- [ ] Block 1.4: Export `modnet_bn_best.ckpt` → ONNX using `export_onnx_modified.py`
- [ ] Block 1.4: Verify in Netron — confirm zero `InstanceNormalization` nodes
- [ ] Block 1.4: Generate golden files via `./helmsman golden`
- [ ] Block 1.4: Build C++ native (`./helmsman build cpp cb`) and run inference
- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)