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
#include <opencv2/imgcodecs.hpp>
#include "Utils/timing/timer.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using helmsman::utils::timing::ScopedTimer;

inline constexpr std::string_view kModnetModuleName = "MODNetMode";

void MODNetMode::SetEngine(InferenceEngine* engine) { engine_ = engine; }
void MODNetMode::SetBackend(MattingBackend* backend) { backend_ = backend; }
void MODNetMode::SetAppConfig(const AppConfig& config) { config_ = config; }

int MODNetMode::Run() {
	auto& logger = helmsman::utils::Logger::GetInstance();

	const int model_input_height = engine_->GetInputHeight() > 0 ? engine_->GetInputHeight() : 512;
	const int model_input_width = engine_->GetInputWidth() > 0 ? engine_->GetInputWidth() : 512;

	// 1. Frontend: preprocess image
	TensorData src;
	{
		ScopedTimer t("runMODNet: preprocess", config_.timing_enabled, logger, kModnetModuleName);
		preprocessor_.SetDumpEnabled(config_.dump_enabled);
		preprocessor_.SetOutputBinPath(config_.output_bin_path);
		cv::Mat img = cv::imread(config_.input_path, cv::IMREAD_COLOR);
		src = preprocessor_.preprocess(img, model_input_width, model_input_height);
	}

	// 2. Inference: benchmark 10 iterations
	std::vector<TensorData> inputs = {src};
	std::vector<TensorData> outputs;

	{
		ScopedTimer bench_timer("runMODNet: benchmark 10x total", config_.timing_enabled, logger,
		                        kModnetModuleName);
		for (int i = 0; i < 10; ++i) {
			auto start = std::chrono::high_resolution_clock::now();
			engine_->Infer(inputs, outputs);
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> dur = end - start;
			logger.Info("[Performance Benchmark " + std::to_string(i + 1) +
			                "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) +
			                " ms.",
			            kModnetModuleName);
		}
	}

	// 3. Backend: postprocess
	{
		ScopedTimer t("runMODNet: postprocess", config_.timing_enabled, logger, kModnetModuleName);
		backend_->SetForegroundImagePath(config_.input_path);
		backend_->SetPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));
		backend_->Postprocess(outputs);
	}

	return 0;
}
