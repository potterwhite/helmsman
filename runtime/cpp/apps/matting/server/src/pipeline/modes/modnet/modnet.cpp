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

#include "pipeline/modes/modnet/modnet.h"
#include <chrono>
#include "common/common-define.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"
#include "Utils/timing/timer.h"

using arcforge::utils::timing::ScopedTimer;

inline constexpr std::string_view kModnetModuleName = "MODNetMode";

int MODNetMode::run(InferenceEngine* engine,
                    const std::string& input_image_path,
                    const std::string& model_path,
                    const std::string& output_bin_path,
                    const std::string& background_path,
                    bool timing_enabled) {
    auto& logger = arcforge::embedded::utils::Logger::GetInstance();

    MattingBackend backend;

    // 1. Load model
    {
        ScopedTimer t("runMODNet: model load", timing_enabled, logger, kModnetModuleName);
        engine->setOutputBinPath(output_bin_path);
        engine->load(model_path);
    }

    const size_t model_input_height = engine->getInputHeight() > 0 ? engine->getInputHeight() : 512;
    const size_t model_input_width  = engine->getInputWidth()  > 0 ? engine->getInputWidth()  : 512;

    // 2. Frontend: preprocess image
    TensorData src;
    {
        ScopedTimer t("runMODNet: preprocess", timing_enabled, logger, kModnetModuleName);
        frontend_.setOutputBinPath(output_bin_path);
        src = frontend_.preprocess(input_image_path, model_input_width, model_input_height);
    }

    // 3. Inference: benchmark 10 iterations
    std::vector<TensorData> inputs = {src};
    std::vector<TensorData> outputs;

    {
        ScopedTimer bench_timer("runMODNet: benchmark 10x total", timing_enabled, logger, kModnetModuleName);
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            engine->infer(inputs, outputs);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> dur = end - start;
            logger.Info("[Performance Benchmark " + std::to_string(i + 1) +
                            "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) + " ms.",
                        kModnetModuleName);
        }
    }

    // 4. Backend: postprocess
    {
        ScopedTimer t("runMODNet: postprocess", timing_enabled, logger, kModnetModuleName);
        backend.setOutputPath(output_bin_path);
        backend.setBackgroundPath(background_path);
        backend.setForegroundImagePath(input_image_path);
        backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));
        backend.postprocess(outputs);
    }

    return 0;
}