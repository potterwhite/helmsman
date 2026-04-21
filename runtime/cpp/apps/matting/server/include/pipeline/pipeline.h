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

#include <memory>
#include <string>
#include "Utils/logger/logger.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"
#include "pipeline/modes/modnet/modnet.h"
#include "pipeline/modes/rvm/rvm.h"
#include "input/input-source.h"

enum class ModelType {
    kMODNet,
    kRVM,
};

class Pipeline {
public:
    static Pipeline& GetInstance();

    void init(std::unique_ptr<InputSource> input_source, const std::string& model_path,
              const std::string& output_bin_path, const std::string& background_path = "",
              ModelType model_type = ModelType::kRVM);

    void init(const std::string& image_path, const std::string& model_path,
              const std::string& output_bin_path, const std::string& background_path = "",
              ModelType model_type = ModelType::kMODNet);

    int run();

    void setTimingEnabled(bool enabled) { timing_enabled_ = enabled; }
    bool isTimingEnabled() const { return timing_enabled_; }

private:
    Pipeline();
    ~Pipeline();

    void verify_parameters_necessary();

    static std::unique_ptr<InferenceEngine> make_engine();

private:
    std::string input_image_path_;
    std::unique_ptr<InputSource> input_source_;
    std::string model_path_;
    std::string output_bin_path_;
    std::string background_path_;
    ModelType model_type_ = ModelType::kMODNet;

    bool timing_enabled_ = true;

    std::unique_ptr<InferenceEngine> engine_;

    MODNetMode modnet_mode_;
    RVMMode rvm_mode_;
};