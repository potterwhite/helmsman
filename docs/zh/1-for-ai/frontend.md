# Frontend 模块详解

## 类层次

```
FrontendBase (abstract)                    ← frontend.h
├── RockchipFrontend                       ← rockchip-frontend.h
│   ├── FfmpegInputSource (demux)
│   ├── MppFrameDecoder (VPU decode)
│   └── RgaNv12ToBgr (RGA color convert)
└── NoHwFrontend                           ← no-hw-frontend.h
    └── cv::VideoCapture (software decode)
```

## 工厂方法

`FrontendBase::Create(input_path, use_hardware, use_pipeline)` 是静态工厂方法。
两个 .cpp 文件实现它，CMake 根据 `CMAKE_PLATFORM` 选择编译哪个：
- `frontend-create-rockchip.cpp` — 可创建 RockchipFrontend 和 NoHwFrontend
- `frontend-create-nohw.cpp` — 只能创建 NoHwFrontend

## 子阶段

### 01-input-source
- `BaseInputSource` (abstract) — 读取原始压缩数据
- `FfmpegInputSource` — FFmpeg demuxer 实现

### 02-decoder
- `BaseFrameDecoder` (abstract) — 解码压缩包为帧
- `MppFrameDecoder` — MPP 硬件解码器（Rockchip VPU）

### 03-color-convert
- `BaseColorConverter` (abstract) — 硬件帧转 BGR
- `RgaNv12ToBgr` — RGA 硬件颜色转换（NV12 → BGR）

### 04-preprocess
- `BasePreprocessor` (abstract) — BGR → TensorData
- `Preprocessor` — 实现：BGR→RGB → ensure3Channel → resize/pad → float32 HWC

## 数据流

```
Hardware path:
  FfmpegInputSource::ReadRaw() → RawPacket
  MppFrameDecoder::decode() → HardwareFrame (NV12 DMA buffer)
  RgaNv12ToBgr::convert() → cv::Mat BGR
  Preprocessor::preprocess() → TensorData

OpenCV path:
  cv::VideoCapture::read() → cv::Mat BGR
  Preprocessor::preprocess() → TensorData
```

## Pipeline 模式

当 `use_pipeline=true` 时，FrontendBase 使用双缓冲流水线：
- 主线程：`ProcessOneFrame()` → pop tensor → 推理 → 合成
- 工作线程：`PrefetchWorkerLoop()` → pop raw frame → preprocess → push tensor
- Channel：`SingleSlotChannel<cv::Mat>` 和 `SingleSlotChannel<TensorData>`

当 `use_pipeline=false` 时，同步模式：read + preprocess 在调用线程上执行。
