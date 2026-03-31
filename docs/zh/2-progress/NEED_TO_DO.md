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
- [ ] 新的onnx的推理结果，与之前的（已经被归档）md5sum不一致，你帮我检查。
    ```bash
    md5sum build/golden/python/py_0*
    33160aa27e0a41079cb9feef750b0d65  build/golden/python/py_01_imread.bin
    cdb2d7cd5199687dac64a1c3e5d47b79  build/golden/python/py_02_cvtColor.bin
    cdb2d7cd5199687dac64a1c3e5d47b79  build/golden/python/py_03_ensure3Channel.bin
    76ccba2a08544776dcd28488664dfa2c  build/golden/python/py_04_normalized.bin
    480ee5867cbc0876b43a50a3e07ff20c  build/golden/python/py_05_resized.bin
    9ca70aaceb36bb8d51d247b64abf58b3  build/golden/python/py_06_transposed.bin
    9ca70aaceb36bb8d51d247b64abf58b3  build/golden/python/py_07_golden_reference_tensor-Input.bin
    ac127a176f5e503db5ac09adf4dffed4  build/golden/python/py_08_inference-Output.bin
    ```
    第二次 golden 用的是 modnet_bn_best_pureBN.onnx（新训练的 Pure-BN 模型）
    第一次 golden 用的是 modnet_photographic_portrait_matting.onnx（原版 IBNorm 模型）
    py_01~py_07 md5 不变（预处理一致），py_08 不同 → **正常，不是 bug**。
    两个不同的模型 → 推理权重不同 → 输出必然不同。
    待办：确认当前 build/golden/python/ 里存的是 pureBN 的输出，后续 C++ 对比用此基准。

- [ ] Block 1.4: Build C++ native (`./helmsman build cpp cb`) and run inference
    你去帮我把cpp的代码检查一下，我来执行。
- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)
