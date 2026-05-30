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
│   ├── _ReadRawPacket (FfmpegInputSource)
│   ├── _DecodeFrame02 (MppFrameDecoder)
│   └── _ConvertToBgr03 (RgaNv12ToBgr)
└── NoHwFrontend                           ← frontend-core/impl/no-hw-frontend.h
    └── _ReadInputSource01 (cv::VideoCapture, overrides 01-03 atomically)
```

## 工厂方法

`FrontendBase::Create(input_path, use_hardware, multithread_enabled)` 是静态工厂方法。
两个 .cpp 文件实现它（位于 `frontend-core/factory/`），CMake 根据 `CMAKE_PLATFORM` 选择编译哪个：
- `frontend-core/factory/frontend-create-rockchip.cpp` — 可创建 RockchipFrontend 和 NoHwFrontend
- `frontend-core/factory/frontend-create-nohw.cpp` — 只能创建 NoHwFrontend

## 4 阶段虚方法

FrontendBase 在 protected 区暴露 4 个阶段方法，对应解码流水线的 4 个阶段：

| 方法 | 阶段 | 说明 |
|---|---|---|
| `_ReadInputSource01()` | Stage 01-03 | 读取输入源（含 decoder retry loop） |
| `_DecodeFrame02()` | Stage 02 | 解码单个压缩包 |
| `_ConvertToBgr03()` | Stage 03 | 硬件帧转 BGR |
| `_PreprocessForInference04()` | Stage 04 | BGR → TensorData（非虚，基类完成） |

- `_ReadInputSource01` 的默认实现串联 `_ReadRawPacket` → `_DecodeFrame02` → `_ConvertToBgr03`，内含 retry loop
- `RockchipFrontend` 实现 stage 02 和 03，使用默认 `_ReadInputSource01`
- `NoHwFrontend` 直接覆盖 `_ReadInputSource01`（cv::VideoCapture 一步完成 01-03）

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
Hardware path (4-stage pipeline):
  _ReadInputSource01()
    ├── _ReadRawPacket()    → RawPacket          (FfmpegInputSource::ReadRaw)
    ├── _DecodeFrame02()    → HardwareFrame       (MppFrameDecoder::decode, with retry)
    └── _ConvertToBgr03()   → cv::Mat BGR         (RgaNv12ToBgr::convert)
  _PreprocessForInference04() → TensorData        (Preprocessor::preprocess)

OpenCV path:
  _ReadInputSource01()       → cv::Mat BGR        (cv::VideoCapture::read, all-in-one)
  _PreprocessForInference04() → TensorData
```

## Multithread 模式

当 `multithread_enabled=true` 时，FrontendBase 使用双缓冲多线程模式：
- 主线程：`_ProcessMultithread()` → stages 01-03 → pop tensor → 返回结果
- 工作线程：`_MultithreadWorkerLoop()` → stage 04（preprocess）
- Channel：`SingleSlotChannel<cv::Mat>` 和 `SingleSlotChannel<TensorData>`

当 `multithread_enabled=false` 时（默认），同步模式：`_ProcessSync()` → 4 个 stages 在调用线程上顺序执行。

两种模式中，4 个 stages 都明确可见。
