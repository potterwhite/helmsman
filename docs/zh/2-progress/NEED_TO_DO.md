# NEED_TO_DO — 当前待办

> **AI Agent 操作规则（必读，每次都要执行）：**
>
> 完成一个 checkbox 后，立即执行以下两步，缺一不可：
> 1. 把该条目的 `[ ]` 改为 `[x]`，并在其下一行追加 `commit: <hash>`
> 2. 把整个已完成的日期组（含所有 `[x]` 条目）**剪切**（从本文件删除），**追加**到
>    `task-logs/NEED_TO_DO_ARCHIVED_<MonDD>.<YYYY>.md` 的末尾
>    （文件不存在则新建；同一天多次归档就不断 append 到同一文件）
>
> 本文件归档后只保留**未完成**的条目。已完成记录永久保存在 `task-logs/` 里。

---

Mar31.2026
- [ ] Block 1.4: Export `modnet_bn_best.ckpt` → ONNX using `export_onnx_modified.py`
    我的包括modnet_bn_best.ckpt在内的所有文件都已存档到：https://huggingface.co/PotterWhite/MODNet/tree/main，请你自行去搜索，我已经做好详尽的目录结构规划，你一看就懂。
    我目前应该是卡在从ckpt -> onnx；
- [ ] Block 1.4: Verify in Netron — confirm zero `InstanceNormalization` nodes
- [ ] Block 1.4: Generate golden files via `./helmsman golden`
- [ ] Block 1.4: Build C++ native (`./helmsman build cpp cb`) and run inference
- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)
