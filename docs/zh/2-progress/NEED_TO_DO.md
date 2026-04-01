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

Apr01.2026

- [ ] Block 1.4: Compare C++ vs Python golden — 深度数值对比
    工具: `tools/MODNet/verify_golden_tensor.py`
    目标: 对比 cpp_08 (C++ 0-255 path) vs py_08 (Python [-1,1] path) 的数值分布

- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)
    打开 `build/golden/cpp/cpp_11_result.png` 目视检查发丝细节

---

Mar31.2026

- [ ] 新的 ONNX 推理结果 md5 与上一次不一致：确认 pureBN 已替换原版
    结论已知: py_01~py_07 (预处理) md5 不变；py_08 (推理输出) 不同是**正常的**
    待确认: `build/golden/python/` 里存的是 pureBN 模型的输出，后续 C++ 对比以此为基准。
    md5: py_08_inference-Output.bin = ac127a176f5e503db5ac09adf4dffed4
