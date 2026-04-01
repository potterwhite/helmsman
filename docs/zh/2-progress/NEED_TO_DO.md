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

- [x] Block 1.4: cmake toolchain + rknn conditional compile fix
    commit: d9a8ce9
    - CMakePresets.json: `native-release` preset 加入 `toolchainFile` 字段，让 cmake 4.x 正确加载 conan_toolchain.cmake
    - client/src/CMakeLists.txt: rknn 源文件改为 `if(TARGET RKNNRT)` 条件加入，native 构建不再编译 rknn-*.cpp
    - scripts/cpp_build.sh: 注释掉旧的 `-DCMAKE_TOOLCHAIN_FILE=` 命令行注入逻辑（已由 preset 承接）

- [x] Block 1.4: 修复 `getInputHeight()`/`getInputWidth()` — ONNX path 缺少这两个方法
    commit: 858afc1
    - pipeline.cpp:88-94: 用 `#ifdef ENABLE_RKNN_BACKEND` 包裹，RKNN path 继续查 `engine.getInputHeight()`，
      ONNX path hardcode 512×512（与 Python golden 预处理目标尺寸一致）

- [ ] Block 1.4: Build C++ native 并生成 cpp_0*.bin
    执行: `./helmsman build cpp cb native`
    构建成功后运行推理（详见下方命令）

    **推理命令（构建完成后用）**：
    ```bash
    # 1. 确认 binary 已生成
    ls runtime/cpp/install/native/release/bin/Helmsman_Matting_Client

    # 2. 创建输出目录
    mkdir -p build/golden/cpp

    # 3. 运行推理（请替换图片路径和模型路径为你实际使用的）
    #    image_path : 与生成 Python golden 时相同的图片
    #    model_path : modnet_bn_best_pureBN.onnx（与 py_08 golden 对应的模型）
    LD_LIBRARY_PATH=runtime/cpp/install/native/release/lib:$LD_LIBRARY_PATH \
        runtime/cpp/install/native/release/bin/Helmsman_Matting_Client \
        <image_path> \
        <onnx_model_path> \
        build/golden/cpp/

    # 4. 查看生成的 bin 文件
    ls build/golden/cpp/
    md5sum build/golden/cpp/cpp_0*
    ```

    **注意**：`LD_LIBRARY_PATH` 需要包含安装目录的 lib 路径，否则会找不到共享库（libHelmsman_Utils.so 等）

- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`
    Python golden 在: `build/golden/python/py_0*.bin`
    C++ golden 在: `build/golden/cpp/cpp_0*.bin`（build 后生成）

- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)

---

Mar31.2026

- [ ] 新的 ONNX 推理结果 md5 与上一次不一致：确认 pureBN 已替换原版
    结论已知: py_01~py_07 (预处理) md5 不变；py_08 (推理输出) 不同是**正常的**
    待确认: `build/golden/python/` 里存的是 pureBN 模型的输出，后续 C++ 对比以此为基准。
    md5: py_08_inference-Output.bin = ac127a176f5e503db5ac09adf4dffed4
