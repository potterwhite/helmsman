# Frontend 模块详解

## 目录结构

```
frontend/
├── CMakeLists.txt
├── frontend-core/                       ← 类型体系
│   ├── frontend-base.{h,cpp}            ← 基类（export）
│   ├── impl/                            ← 实现子类（internal）
│   │   ├── no-hw-frontend.{h,cpp}
│   │   └── rockchip-frontend.{h,cpp}
│   └── factory/                         ← CMake 选择的工厂
│       ├── frontend-create-nohw.cpp
│       └── frontend-create-rockchip.cpp
├── stages/                              ← 处理流程
│   ├── 01-input-source/
│   ├── 02-decoder/
│   ├── 03-color-convert/
│   └── 04-preprocess/
```

**Export 规则**：只有 `frontend-base.h` 和 `preprocessor.h` 在 `include/`（被外部引用），其余 headers 在 `src/`（仅内部使用）。

## 类层次

```
FrontendBase (abstract)                    ← frontend-core/frontend-base.h
├── RockchipFrontend                       ← frontend-core/impl/rockchip-frontend.h
│   ├── FfmpegInputSource (demux)
│   ├── MppFrameDecoder (VPU decode)
│   └── RgaNv12ToBgr (RGA color convert)
└── NoHwFrontend                           ← frontend-core/impl/no-hw-frontend.h
    └── cv::VideoCapture (software decode)
```

## 工厂方法

`FrontendBase::Create(input_path, use_hardware, use_pipeline)` 是静态工厂方法。
两个 .cpp 文件实现它（位于 `frontend-core/factory/`），CMake 根据 `CMAKE_PLATFORM` 选择编译哪个：
- `frontend-core/factory/frontend-create-rockchip.cpp` — 可创建 RockchipFrontend 和 NoHwFrontend
- `frontend-core/factory/frontend-create-nohw.cpp` — 只能创建 NoHwFrontend

## 子阶段

### stages/01-input-source
- `BaseInputSource` (abstract) — 读取原始压缩数据
- `FfmpegInputSource` — FFmpeg demuxer 实现

### stages/02-decoder
- `BaseFrameDecoder` (abstract) — 解码压缩包为帧
- `MppFrameDecoder` — MPP 硬件解码器（Rockchip VPU）

### stages/03-color-convert
- `BaseColorConverter` (abstract) — 硬件帧转 BGR
- `RgaNv12ToBgr` — RGA 硬件颜色转换（NV12 → BGR）

### stages/04-preprocess
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
