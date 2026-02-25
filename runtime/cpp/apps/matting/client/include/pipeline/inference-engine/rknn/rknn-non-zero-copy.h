#pragma once

// 引入 RKNN API (请确保 CMake/Makefile 中链接了 librknnrt.so)
#include "rknn_api.h"

#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "pipeline/core/data_structure.h"
#include "pipeline/inference-engine/base/inference-engine.h"

#include <vector>
#include <string>

class InferenceEngineRKNN : public InferenceEngine {
public:
    InferenceEngineRKNN();
    ~InferenceEngineRKNN() override; // 注意加上 override 最佳实践

    void load(const std::string& model_path) override;
    TensorData infer(const TensorData& input) override;

private:
    // 释放 RKNN 资源
    void release();

private:
    // 成员变量
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_;

    // 缓存模型的输入输出属性
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    // const std::string kcurrent_module_name = "RKNN-Engine";
};