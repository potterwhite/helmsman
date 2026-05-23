# CMake 构建系统

## 平台检测

使用 `CMAKE_PLATFORM` 变量（分号分隔的列表）进行显式平台检测：

```cmake
# 典型值: "aarch64;rockchip;rk3588s"
if("rockchip" IN_LIST CMAKE_PLATFORM)
    # Rockchip 特定源文件
endif()
```

**不要**用 `if(TARGET ...)` 间接推断平台。

## 条件编译模式

采用"两个 .cpp 文件，CMake 选择编译哪个"的模式，避免 `#ifdef`：

```
frontend-create-rockchip.cpp  ← CMAKE_PLATFORM 包含 "rockchip" 时编译
frontend-create-nohw.cpp      ← 其他平台编译
```

同一模式用于 InferenceEngine 工厂。

## 源文件组织

```
src/pipeline/stages/frontend/
├── CMakeLists.txt           ← 平台选择逻辑
├── frontend.cpp             ← 共享（FrontendBase 实现）
├── no-hw-frontend.cpp       ← 共享（NoHwFrontend，所有平台都需要）
├── rockchip-frontend.cpp    ← 仅 Rockchip
├── frontend-create-rockchip.cpp  ← 仅 Rockchip
├── frontend-create-nohw.cpp      ← 非 Rockchip
├── 01-input-source/
│   └── ffmpeg-input-source.cpp   ← 共享
├── 02-decoder/
│   └── mpp-frame-decoder.cpp     ← 仅 Rockchip
├── 03-color-convert/
│   └── rga-nv12-to-bgr.cpp       ← 仅 Rockchip
└── 04-preprocess/
    └── preprocessor.cpp          ← 共享
```

## 工具链

- Rockchip: `cmake/toolchains/aarch64-rockchip.cmake`
- Native: 系统默认编译器

## 构建命令

```bash
# Rockchip 板端
cmake --build build/rk3588s/release --target Matting_Server

# Native 开发机
cmake --build build/native/release --target Matting_Server
```
