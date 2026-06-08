# 模型输出张量选择策略

## 概述

BackEnd 模块需要从推理引擎的输出中选择正确的 alpha matte 张量。不同模型的输出结构不同，因此采用 `model_type` 进行差异化处理。

## 模型对比

| 模型 | 输出数量 | 输出张量名称 | 选择策略 |
|------|----------|--------------|----------|
| MODNet | 1 | `"output"` | 直接使用 `outputs[0]` |
| RVM | 5 | `"pha"`, `"r1o"`, `"r2o"`, `"r3o"`, `"r4o"` | 按名称 `"pha"` 查找 |

## 代码位置

- **类型定义**：`include/common/types.h` - `ModelType` 枚举
- **选择逻辑**：`src/pipeline/stages/backend/backend.cpp` - `BackEnd::Postprocess()`

## 实现细节

```cpp
const TensorData* output_tensor_ptr = nullptr;

switch (config_.model_type) {
    case ModelType::kMODNet:
        // MODNet: single output, always the first one
        if (!outputs.empty()) {
            output_tensor_ptr = &outputs[0];
        }
        break;

    case ModelType::kRVM:
        // RVM: multiple outputs, find "pha" by name
        for (const auto& td : outputs) {
            if (td.name == "pha") {
                output_tensor_ptr = &td;
                break;
            }
        }
        break;

    default:
        // Unknown model type: try "pha" first, fallback to first
        for (const auto& td : outputs) {
            if (td.name == "pha") {
                output_tensor_ptr = &td;
                break;
            }
        }
        if (!output_tensor_ptr && !outputs.empty()) {
            output_tensor_ptr = &outputs[0];
        }
        break;
}
```

## 添加新模型

当添加新模型时：

1. 在 `ModelType` 枚举中添加新类型
2. 在 `BackEnd::Postprocess()` 的 switch 语句中添加对应 case
3. 更新本文档

## MODNet 输出张量信息

**输入**：
- 名称：`"input"`
- 形状：`[1, 3, 512, 512]` (NCHW)
- 数据类型：float16 / float32

**输出**：
- 名称：`"output"`
- 形状：`[1, 1, 512, 512]` (NCHW)
- 数据类型：float16 / float32
- 含义：Alpha matte (单通道)

## RVM 输出张量信息

**输入**：
- `"src"`: `[1, 3, H, W]` - 源图像
- `"r1i"` ~ `"r4i"`: RNN 隐藏状态

**输出**：
- `"pha"`: `[1, 1, H, W]` - Alpha matte
- `"r1o"` ~ `"r4o"`: RNN 隐藏状态更新
- `"fgr"`: 跳过（在 ReadOutputBuffers3rd 中处理）
