// Copyright (c) 2026 PotterWhite
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "pipeline/pipeline.h"
#include "common/common-define.h"

#if defined(INFERENCE_BACKEND_RKNN_ZEROCOPY)
#include "pipeline/stages/inference-engine/rknn/rknn-zero-copy.h"
#elif defined(INFERENCE_BACKEND_RKNN_NON_ZEROCOPY)
#include "pipeline/stages/inference-engine/rknn/rknn-non-zero-copy.h"
#else
#include "pipeline/stages/inference-engine/onnx/onnx.h"
#endif

#include "Utils/timing/timer.h"

using arcforge::utils::timing::ScopedTimer;

std::unique_ptr<InferenceEngine> Pipeline::make_engine() {
#if defined(INFERENCE_BACKEND_RKNN_ZEROCOPY)
    return std::make_unique<InferenceEngineRKNNZeroCP>();
#elif defined(INFERENCE_BACKEND_RKNN_NON_ZEROCOPY)
    return std::make_unique<InferenceEngineRKNN>();
#else
    return std::make_unique<InferenceEngineONNX>();
#endif
}

Pipeline& Pipeline::GetInstance() {
    static Pipeline instance;
    return instance;
}

Pipeline::Pipeline() {
    arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline object constructed.",
                                                      kcurrent_module_name);
}

Pipeline::~Pipeline() {
    arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline cleaned up.",
                                                      kcurrent_module_name);
}

void Pipeline::init(std::unique_ptr<InputSource> input_source, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
    this->input_source_    = std::move(input_source);
    this->model_path_      = model_path;
    this->output_bin_path_ = output_bin_path;
    this->background_path_ = background_path;
    this->model_type_      = model_type;
    this->engine_          = make_engine();
}

void Pipeline::init(const std::string& input_image_path, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
    this->input_image_path_ = input_image_path;
    this->model_path_       = model_path;
    this->output_bin_path_  = output_bin_path;
    this->background_path_  = background_path;
    this->model_type_       = model_type;
    this->engine_           = make_engine();
}

void Pipeline::verify_parameters_necessary() {
    if (input_source_ == nullptr && input_image_path_.empty()) {
        throw std::invalid_argument("No input source: neither video nor image provided.");
    }
    if (model_path_.empty()) {
        throw std::invalid_argument("Model path is empty.");
    }
    if (output_bin_path_.empty()) {
        throw std::invalid_argument("Output binary path is empty.");
    }
}

int Pipeline::run() {
    verify_parameters_necessary();

    auto& logger = arcforge::embedded::utils::Logger::GetInstance();

    ScopedTimer run_timer("Pipeline::run() total", timing_enabled_, logger, kcurrent_module_name);

    switch (model_type_) {
        case ModelType::kMODNet:
            logger.Info("Pipeline: running MODNet path (single-frame)", kcurrent_module_name);
            return modnet_mode_.run(engine_.get(), input_image_path_, model_path_,
                                     output_bin_path_, background_path_, timing_enabled_);
        case ModelType::kRVM:
            logger.Info("Pipeline: running RVM path (recurrent multi-frame)", kcurrent_module_name);
            return rvm_mode_.run(engine_.get(), std::move(input_source_), model_path_,
                                   output_bin_path_, background_path_, timing_enabled_);
        default:
            throw std::runtime_error("Unknown model type");
    }
}