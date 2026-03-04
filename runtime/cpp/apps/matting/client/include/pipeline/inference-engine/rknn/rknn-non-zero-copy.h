/*
 * Copyright (c) 2026 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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