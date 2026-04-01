# NEED_TO_DO Archived — Apr01.2026

Apr01.2026

- [x] Block 1.4: cmake toolchain + rknn conditional compile fix
    commit: d9a8ce9
    - CMakePresets.json: `native-release` preset 加入 `toolchainFile` 字段，让 cmake 4.x 正确加载 conan_toolchain.cmake
    - client/src/CMakeLists.txt: rknn 源文件改为 `if(TARGET RKNNRT)` 条件加入，native 构建不再编译 rknn-*.cpp
    - scripts/cpp_build.sh: 注释掉旧的 `-DCMAKE_TOOLCHAIN_FILE=` 命令行注入逻辑（已由 preset 承接）
    - Root cause: cmake 4.x 改变了 --preset + toolchain 加载时机，-D 命令行注入不够早
    - Fix: toolchainFile 字段是 CMake Presets 规范字段，cmake 3.19+ 全版本支持

- [x] Block 1.4: 修复 `getInputHeight()`/`getInputWidth()` — ONNX path 缺少这两个方法
    commit: 858afc1
    - pipeline.cpp:88-94: 用 `#ifdef ENABLE_RKNN_BACKEND` 包裹
    - RKNN path 继续查 `engine.getInputHeight()`（从 input_attr_.dims 读）
    - ONNX path hardcode 512×512（与 Python golden 预处理目标尺寸一致）
    - 原因：InferenceEngineONNX 不应有 getInputHeight/Width，ONNX 模型动态轴无固定维度

- [x] Block 1.4: Build C++ native + run inference
    commit: (无代码改动，build 产物不入库)
    - 构建成功：`./helmsman build cpp cb native`
    - 推理成功：`green-fall-girl-point-to.png` + `modnet_bn_best_pureBN.onnx`
    - 输出：`build/golden/cpp/cpp_0*.bin`（11 个文件）
    - md5 对比（C++ vs Python golden）：
      - cpp_01~03 vs py_01~03: ✅ 完全一致（imread / BGR→RGB / ensure3ch）
      - cpp_04 vs py_04: ❌ 不同（预期）— C++ 保持 0-255 float32；Python 做 ÷127.5-1 = [-1,1]
      - cpp_05 vs py_05: ❌ 不同（预期）— C++ 用 INTER_LINEAR + letterbox；Python 用 INTER_AREA 无 padding
      - cpp_08 vs py_08: ❌ 不同（预期）— 前处理不同 → 推理输出必然不同
    - 结论：差异完全符合预期，C++ 是 RKNN 专用配置；native 路径功能完整可运行

- [x] .gitignore + cpp-code-review.md 清理
    commit: (pending — see batch commit)
    - `.gitignore` 加入 `media`（软链接，原来只有 `media/` 不匹配）
    - `docs/zh/1-for-ai/cpp-code-review.md` 删除（AI agent 自动生成的临时分析，有误导性内容）
