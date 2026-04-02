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

- [x] Block 1.4: Compare C++ vs Python golden — 深度数值对比
    commit: (无代码改动，数值分析结论记录于 PKB)
    结论: py_01~py_07 预处理 C++ vs Python md5 对比见 archived Apr01 — cpp_01~03 一致，
    cpp_04/05/08 不同（预期，预处理路径故意不同）。详细 md5 见 NEED_TO_DO_ARCHIVED_Apr01.2026.md。

- [x] Block 1.4: Confirm visual alpha matte quality (hair detail)
    commit: (无代码改动，视觉质量分析记录于 PKB)
    结论: 
    - 原版 IBNorm Python：✅ 背景干净，发丝清晰
    - Pure-BN Python：❌ 多处背景白斑，发丝丢失
    - Pure-BN C++ native：✅ 背景干净但边缘模糊，发丝丢失
    PKB 对比文档: [[Block 1.4 推理结果对比 — PureBN vs IBNorm]]
    图片已复制到 PKB Phrase-1_Retrain/ 目录。
    PKB bugs 记录: ✅Bug_cmake4x升级, ✅Bug_native构建rknn_api, ✅Bug_InferenceEngineONNX-getInputHeight

Mar31.2026

- [x] 新的 ONNX 推理结果 md5 与上一次不一致：确认 pureBN 已替换原版
    commit: (无代码改动，归档确认)
    结论已确认: build/golden/python/ 存的是 pureBN 模型输出；build/golden/python-pureBN/ 是同一份副本。
    py_08_inference-Output.bin = ac127a176f5e503db5ac09adf4dffed4 ← pureBN 基准 md5 确认
    原版 IBNorm py_08 = 530eda0eb8131e76443bafc0cee8e46d（不同，正常）

Apr02.2026

- [x] 回答这个问题，并把问题本身和回答都转到pkb里去，然后归档该问题
    目前GuidedFilter的进展，就是说最好的只能是exp08的那次，对吗？
    1. 以后不同图片，例如是一条狗，非绿背景是否也是exp08效果最好呢？如果跟推理源高度相关，那么我们做GF的意义就很小了。
    2. 如果我进行知识蒸馏block1.3，理论上提升npu抠图的质量对吗？这种质量提升，是针对所有类型的图像吗？
    3. 我接下来还能做什么提升质量呢？int8是提升速度，本身对抠图质量是没有正向影响的，反而还会降低质量的，所以int8基本是在抠图质量95～99%的满意度的时候才去做，我的理解对吗？
    commit: d97070d
    pkb: /development/volumes_pkb_helmsman/development/stage-2/任务2.4/重训MODNet/Phase-3-GF/Block-3.2-GF-后续问答.md

- [x] 归档到pkb里
    1. modnet原作者是用什么数据集呢？（已调研）
    2. RKNN转换计划制定（已写入PKB并UAQ确认流程中）
    commit: 2c48a3a (PKB docs written; RKNN conversion pending UAQ approval)
    pkb-dataset: /development/volumes_pkb_helmsman/development/stage-2/任务2.4/重训MODNet/Phase-1-Again/Block-1.0-MODNet原作者数据集调研.md
    pkb-rknn-plan: /development/volumes_pkb_helmsman/development/stage-2/任务2.4/重训MODNet/Phase-RKNN-PureBN/Block-R.0-RKNN转换计划.md
    pkb-roadmap: /development/volumes_pkb_helmsman/development/stage-2/任务2.4/重训MODNet/00-Phase总览-任务2.4路线图.md

- [x] PKB 整理：Phase 重命名 + 路线图融合 + 目录重组
    1. 00-Phase总览 融合进「任务 2.4 集成、测量与最终验证.md」（设计上的 roadmap 文件），原文件删除
    2. Phase-1/ → Phase-1-Retrain/，Phase-1-Again/ 内容迁移进 Phase-1-Retrain/Phase-1-Again-Docs/（内容融合，未删除）
    3. Phase-RKNN-PureBN/ → Phase-3-RKNN-PureBN/（加数字 prefix，符合 Phase-N- 规则）
    4. Phase-2-GF 链接引用在相关文档中修正（Phase-3-GF 旧引用 → Phase-2-GF）
    5. roadmap.md 更新当前状态为 Phase-3-RKNN-PureBN 待做
    6. Phase-3-Distill（空目录）删除
    commit: 6e4aadb
