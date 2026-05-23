# 端到端数据流

## RVM 视频 matting 路径

```
┌─────────────────────────────────────────────────────────────┐
│ Pipeline::Init()                                            │
│   FrontendBase::Create() → RockchipFrontend                 │
│   createInferenceEngine() → InferenceEngineRKNNZeroCP       │
│   engine->Load(model.rknn)                                  │
│   backend_.SetOutputPath / SetBackgroundPath                │
│   注入到 RVMMode                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ RVMMode::Run()                                              │
│   _PrepareRun() → 查询模型维度 + 初始化 RNN 状态              │
│   _OutputModeProcess() → 打开 VideoWriter 或 DRM            │
│   _LoadOrCreateBackground() → 加载背景图                     │
│   _RunMainLoop()                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 每帧循环 (_RunMainLoop)                                      │
│                                                             │
│   FrontendBase::ProcessOneFrame(model_w, model_h)           │
│     ├── ReadFrame() → cv::Mat BGR                           │
│     └── Preprocessor::preprocess() → TensorData             │
│                                                             │
│   RVMMode::_ProcessOneFrame()                               │
│     ├── RecurrentStateManager::inject() → 注入 RNN 状态     │
│     ├── engine->Infer(inputs, outputs) → NPU 推理           │
│     ├── RecurrentStateManager::update() → 更新 RNN 状态     │
│     ├── Backend::Postprocess() → alpha matte                │
│     └── Composite → 混合输出                                 │
└─────────────────────────────────────────────────────────────┘
```

## 数据结构

| 结构 | 用途 |
|------|------|
| `RawPacket` | FFmpeg demux 输出的压缩包 |
| `HardwareFrame` | 硬件解码器输出的帧（NV12 DMA buffer） |
| `TensorData` | 推理引擎输入/输出（float32 数据 + shape + padding info） |
| `FrameResult` | Frontend 输出（BGR frame + hw_frame + tensor） |

## TensorData 字段

```cpp
struct TensorData {
    std::vector<float> data;           // HWC float32 数据
    std::vector<int64_t> shape;        // NHWC shape
    int orig_width, orig_height;       // 原始图像尺寸
    int pad_top, pad_bottom, pad_left, pad_right;  // padding 信息
    std::string name;                  // tensor 名称（用于 RNN 状态识别）
};
```
