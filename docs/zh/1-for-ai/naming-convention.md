# 命名规范（Google C++ Style 速查）

## 文件名

- 小写 + 连字符：`rockchip-frontend.cpp`、`base-preprocessor.h`
- 头文件：`.h`，实现文件：`.cpp`

## 类名

- PascalCase：`FrontendBase`、`RockchipFrontend`、`MattingBackend`

## 方法名

- 公有方法：PascalCase：`ProcessOneFrame()`、`ReadFrame()`、`GetInputHeight()`
- 私有方法：PascalCase + 下划线前缀：`_PrepareRun()`、`_InitRecurrentStates()`
- getter：PascalCase：`width()`、`height()`、`fps()`（不是 `getWidth()`）

## 变量名

- 成员变量：snake_case + 下划线后缀：`frontend_`、`engine_`、`config_`
- 局部变量：snake_case：`model_width`、`alpha_8u`
- 常量：k 前缀 + PascalCase：`kDefaultModelInputHeight`、`kRvmModuleName`

## 类型

- `int` 用于图像宽度/高度（行业标准：OpenCV/FFmpeg/MPPKit/RGAKit 都用 `int`）
- `size_t` 仅用于缓冲区字节数、循环索引
- `assert(w > 0 && h > 0)` 在系统入口保护

## 注释

- 不写 WHAT 注释（代码本身就是文档）
- 只写 WHY 注释：隐藏约束、微妙不变量、workaround
- 一行注释用 `//`，多行用 `/* */`
