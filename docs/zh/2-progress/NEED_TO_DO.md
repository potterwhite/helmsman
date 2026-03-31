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
- [ ] load_from_string
    刚才ai说它执行通过了，我现在脚本里进行操作，根本不通过，你帮我把脚本也调试通过。
    要求一样，你让我来操作，你来说。
    ```bash
    developer@rk3588-rk3588s_ubuntu-24-04:~/primary-folder/helmsman.git$ ./helmsman

    [Helmsman] --- Helmsman Project Main Menu ---
    1. Prepare Environment
    2. Convert .ckpt model to .onnx
    3. Run Python inference
    4. Generate Golden Files (Interactive)
    5. Build C++ Helmsman Engine
    6. Clean build
    7. Clean all(build/venv/models/MODNet SDK etc)
    8. Exit
    Select [1-8]: 2
    [Helmsman] ✅ Activate Python venv Successfully.
    [Helmsman] Select MODNet variant:
    1) Original (modnet_onnx.py)
    2) Modified (modnet_onnx_modified.py) — anti-fusion, for pretrained IBNorm ckpt
    3) Pure-BN  (export_onnx_pureBN.py)  — for retrained BatchNorm-only ckpt
    Select [1-3]: 3
    [Helmsman]    Using: Pure-BN variant (retrained BatchNorm-only)
    [Helmsman] 🔎 Checking for pre-trained models (.ckpt)...
    [Helmsman]    Please choose a .ckpt model to convert to ONNX:
    1) /development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/third-party/sdk/MODNet.git/checkpoints/modnet_webcam_portrait_matting.ckpt
    2) /development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt
    3) /development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/third-party/sdk/MODNet.git/checkpoints/modnet_photographic_portrait_matting.ckpt
    #? 2
    [Helmsman]    Selected CKPT: modnet_bn_best.ckpt
    [Helmsman]    Output ONNX:   /development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/third-party/sdk/MODNet.git/pretrained/modnet_bn_best_pureBN.onnx
    [Helmsman] 🚀 Starting conversion...
    Using Device: cpu
    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

    Traceback (most recent call last):
    File "/home/developer/.pyenv/versions/3.8.10/lib/python3.8/runpy.py", line 194, in _run_module_as_main
        return _run_code(code, main_globals, None,
    File "/home/developer/.pyenv/versions/3.8.10/lib/python3.8/runpy.py", line 87, in _run_code
        exec(code, run_globals)
    File "/development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/third-party/sdk/MODNet.git/onnx/export_onnx_pureBN.py", line 152, in <module>
        torch.onnx.export(
    File "/development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/.venv/lib/python3.8/site-packages/torch/onnx/utils.py", line 506, in export
        _export(
    File "/development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/.venv/lib/python3.8/site-packages/torch/onnx/utils.py", line 1620, in _export
        proto = onnx_proto_utils._add_onnxscript_fn(
    File "/development/rk3588s_volume/src/ai/image-matting/primary-folder/helmsman.git/.venv/lib/python3.8/site-packages/torch/onnx/_internal/onnx_proto_utils.py", line 228, in _add_onnxscript_fn
        model_proto = onnx.load_from_string(model_bytes)
    AttributeError: module 'onnx' has no attribute 'load_from_string'

    developer@rk3588-rk3588s_ubuntu-24-04:~/primary-folder/helmsman.git$
    ```

- [ ] Block 1.4: Generate golden files via `./helmsman golden` 🔜 待测试图片放入 media/
- [ ] Block 1.4: Build C++ native (`./helmsman build cpp cb`) and run inference
- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)
