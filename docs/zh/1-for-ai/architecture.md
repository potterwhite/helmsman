# 整体架构

## 核心组件

```
Pipeline (Singleton, composition root)
├── FrontendBase (abstract, platform-specific subclass)
│   ├── RockchipFrontend  — FFmpeg demux → MPP decode → RGA color convert
│   └── NoHwFrontend      — cv::VideoCapture (software decode)
├── InferenceEngine (abstract, platform-specific)
│   ├── InferenceEngineRKNNZeroCP  — RKNN zero-copy (Rockchip NPU)
│   └── InferenceEngineONNX        — ONNX Runtime (CPU/GPU)
├── MattingBackend (post-processing: alpha refinement, compositing)
└── Modes
    ├── RVMMode     — 视频 matting（递归神经网络，多帧）
    └── MODNetMode  — 图像 matting（单帧）
```

## 生命周期

```
Pipeline::Init(config)
  1. FrontendBase::Create(input_path, use_hardware, multithread_enabled)
  2. MakeEngine() → createInferenceEngine()
  3. engine->Load(model_path)
  4. backend_.SetOutputPath / SetBackgroundPath
  5. 注入到 modes: SetEngine / SetFrontend / SetBackend / SetConfig

Pipeline::Run()
  → rvm_mode_.Run() 或 modnet_mode_.Run()
  （纯执行，不再做初始化）
```

## 设计决策

### 为什么 Pipeline 拥有所有组件？
- Pipeline 是 composition root，负责创建和注入
- Modes 是纯执行器，只关心推理逻辑
- 生命周期清晰：Pipeline 销毁时所有组件一起销毁

### 为什么 Frontend 是虚基类？
- 唯一差异点：帧读取方式（硬件解码 vs OpenCV 软解）
- 共享部分：预处理、多线程模式、计时、channel 基础设施
- 4 个阶段虚方法（`_ReadInputSource01`、`_DecodeFrame02`、`_ConvertToBgr03`）提供细粒度覆盖点
- `_ReadRawPacket` 是内部 helper，被 `_ReadInputSource01` 默认实现调用

### 为什么 Backend 由 Pipeline 拥有？
- Backend 配置（输出路径、背景路径）在 Init 时设置一次
- Modes 通过指针访问，不拥有 Backend
- 与 Frontend/Engine 的生命周期管理一致

### CLI 参数
- `--multithread` 启用多线程模式（默认关闭，单线程同步模式）
- `--hwdecoder` 启用硬件解码路径

## MattingServer（独立路径）

`MattingServer` 是独立的 DMA fd API，不经过 Pipeline：
- 自己拥有 Backend + Engine + Preprocessor
- 输入：DMA buffer fd
- 输出：DMA buffer fd
- 用于实时视频管线（V4L2 camera → NPU → DRM display）
