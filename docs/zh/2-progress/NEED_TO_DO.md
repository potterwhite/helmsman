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
    commit: (pending — see git commit below)
    - CMakePresets.json: `native-release` preset 加入 `toolchainFile` 字段，让 cmake 4.x 正确加载 conan_toolchain.cmake
    - client/src/CMakeLists.txt: rknn 源文件改为 `if(TARGET RKNNRT)` 条件加入，native 构建不再编译 rknn-*.cpp
    - scripts/cpp_build.sh: 注释掉旧的 `-DCMAKE_TOOLCHAIN_FILE=` 命令行注入逻辑（已由 preset 承接）
    - 结果：cmake configure 成功，OpenCV 找到，rknn_api.h 编译错误消失

- [ ] Block 1.4: 修复 `getInputHeight()`/`getInputWidth()` — ONNX path 缺少这两个方法
    文件: `runtime/cpp/apps/matting/client/src/pipeline/pipeline.cpp:88-89`
    问题: `engine.getInputHeight()/getInputWidth()` 只在 `InferenceEngineRKNNZeroCP` 定义；
          `InferenceEngineONNX` 无此接口（ONNX 模型动态轴，load 后没有固定维度）
    方向: 见下方"AI 教练建议"

    **AI 教练建议（只读，用户自行决定）**：
    ONNX 模型使用动态轴 `[batch, 3, height, width]`，加载后无固定 h/w。
    有两种方案：
    - 方案 A（最小改动）：给 `InferenceEngineONNX` 加 `getInputHeight()`/`getInputWidth()`，
      从模型 input tensor shape 读取（如果模型实际有具体值），或者返回一个 hardcoded 默认值（如 512）。
      在 `onnx.h` 和 `onnx.cpp` 的 `load()` 里 parse `session_.GetInputTypeInfo(0)` 读维度。
    - 方案 B（更干净）：pipeline.cpp 里不从 engine 查 h/w，改为从外部（config / 命令行参数）传入，
      或者硬编码 512×512 用于 native 测试，RKNN path 继续用 `engine.getInputHeight()`.
      结构上更清晰，且 ONNX 模型本来就是动态轴，"从模型查 h/w"语义上不准确。
    推荐方案 B，只改 `pipeline.cpp`，改动最局限。
    ---直接用宏定义了，你可以看pipeline.cpp，然后更新docs，git commit
    然后给我运行命令，推理同一张图片，获取md5，看效果。

- [ ] Block 1.4: Build C++ native 并生成 cpp_0*.bin
    执行: `./helmsman build cpp cb native`
    检查: 对比 `py_07_golden_reference_tensor-Input.bin` vs 对应的 C++ 输出

- [ ] Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py`

- [ ] Block 1.4: Confirm visual alpha matte quality (hair detail)

---

Mar31.2026

- [ ] 新的 ONNX 推理结果 md5 与上一次不一致：确认 pureBN 已替换原版
    结论已知: py_01~py_07 (预处理) md5 不变；py_08 (推理输出) 不同是**正常的**
    （两个不同模型 → 权重不同 → 输出必然不同）
    待确认: `build/golden/python/` 里存的是 pureBN 模型的输出，后续 C++ 对比以此为基准。
    md5: py_08_inference-Output.bin = ac127a176f5e503db5ac09adf4dffed4
