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
#include "common/common-define.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"
#include "pipeline/stages/frontend/frontend.h"
#include "pipeline/modes/modnet/modnet.h"
#include "pipeline/modes/rvm/rvm.h"

enum class ModelType {
    kMODNet,
    kRVM,
};

// Pipeline runtime configuration — passed from the top-level entry point.
struct PipelineConfig {
    ModelType model_type = ModelType::kMODNet;
    OutputMode output_mode = OutputMode::kMp4;
    bool timing_enabled = true;
    bool use_hardware_decoder = false;
    bool is_video = false;
    std::string input_path;
    std::string model_path;
    std::string output_bin_path;
    std::string background_path;
};

class Pipeline {
public:
    static Pipeline& GetInstance();

    void init(const PipelineConfig& config);

    int run();

    void setTimingEnabled(bool enabled) { config_.timing_enabled = enabled; }
    bool isTimingEnabled() const { return config_.timing_enabled; }

private:
    Pipeline();
    ~Pipeline();

    void verify_parameters_necessary();

    static std::unique_ptr<InferenceEngine> make_engine();

private:
    PipelineConfig config_;
    std::unique_ptr<Frontend> frontend_;
    std::unique_ptr<InferenceEngine> engine_;

    MODNetMode modnet_mode_;
    RVMMode rvm_mode_;
};