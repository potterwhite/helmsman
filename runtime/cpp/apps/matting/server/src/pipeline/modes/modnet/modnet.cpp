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
#include "Utils/timing/timer.h"

using helmsman::utils::timing::ScopedTimer;

inline constexpr std::string_view kModnetModuleName = "MODNetMode";

void MODNetMode::SetEngine(InferenceEngine* engine) { engine_ = engine; }
void MODNetMode::SetFrontend(FrontEnd* frontend) { frontend_ = frontend; }
void MODNetMode::SetBackend(BackEnd* backend) { backend_ = backend; }
void MODNetMode::SetAppConfig(const AppConfig& config) { config_ = config; }

void MODNetMode::_InitOutputSink(int src_width, int src_height) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	if (config_.output_mode == OutputMode::kDrm) {
		if (drm_display_.Init(src_width, src_height)) {
			std::tie(drm_panel_w_, drm_panel_h_) = drm_display_.PanelSize();
			logger.Info("DRM display initialized: panel " + std::to_string(drm_panel_w_) +
			                 "x" + std::to_string(drm_panel_h_),
			            kModnetModuleName);
		} else {
			logger.Warning("DRM init failed. Falling back to mp4.", kModnetModuleName);
			config_.output_mode = OutputMode::kMp4;
		}
	}
}

void MODNetMode::_Display(const cv::Mat& result, int output_w, int output_h) {
	if (result.empty())
		return;

	if (config_.output_mode == OutputMode::kDrm) {
		// ----- DRM show Mode -----
		// Resize result to DRM panel dimensions
		cv::Mat resized;
		if (result.cols != output_w || result.rows != output_h) {
			cv::resize(result, resized, cv::Size(output_w, output_h), 0, 0, cv::INTER_LINEAR);
		} else {
			resized = result;
		}

		const int n_pixels = output_w * output_h;
		argb_buf_.resize(static_cast<size_t>(n_pixels) * 4);
		uint8_t* xrgb = argb_buf_.data();

		if (resized.channels() == 1) {
			// Single channel alpha matte - display as grayscale
			const uint8_t* alpha = resized.ptr<uint8_t>(0);
			for (int i = 0; i < n_pixels; ++i) {
				xrgb[0] = alpha[0];  // B
				xrgb[1] = alpha[0];  // G
				xrgb[2] = alpha[0];  // R
				xrgb[3] = 0xFF;      // X (padding)
				alpha++;
				xrgb += 4;
			}
		} else {
			// 3-channel BGR image
			const uint8_t* bgr = resized.ptr<uint8_t>(0);
			for (int i = 0; i < n_pixels; ++i) {
				xrgb[0] = bgr[0];  // B → B
				xrgb[1] = bgr[1];  // G → G
				xrgb[2] = bgr[2];  // R → R
				xrgb[3] = 0xFF;    // X (padding)
				bgr += 3;
				xrgb += 4;
			}
		}
		drm_display_.ShowARGB(argb_buf_.data());
	}
	// MP4 mode: no display needed (image already saved by BackEnd)
}

int MODNetMode::Run() {
	auto& logger = helmsman::utils::Logger::GetInstance();

	const int model_input_height = engine_->GetInputHeight() > 0 ? engine_->GetInputHeight() : 512;
	const int model_input_width = engine_->GetInputWidth() > 0 ? engine_->GetInputWidth() : 512;

	// 1. Frontend: preprocess image (single interface, like RVM)
	auto result = frontend_->ProcessOneFrame(model_input_width, model_input_height);
	if (!result)
		return -1;

	// 2. Inference: benchmark 10 iterations
	std::vector<TensorData> outputs;
	{
		ScopedTimer bench_timer("runMODNet: benchmark 10x total", config_.timing_enabled, logger,
		                        kModnetModuleName);
		for (int i = 0; i < 10; ++i) {
			auto start = std::chrono::high_resolution_clock::now();
			engine_->Infer({result->tensor}, outputs);
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> dur = end - start;
			logger.Info("[Performance Benchmark " + std::to_string(i + 1) +
			                "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) +
			                " ms.",
			            kModnetModuleName);
		}
	}

	// 3. BackEnd: postprocess
	cv::Mat alpha;
	{
		ScopedTimer t("runMODNet: postprocess", config_.timing_enabled, logger, kModnetModuleName);
		alpha = backend_->Postprocess(outputs, result->frame);
	}

	// 4. Output: display or save
	if (config_.output_mode == OutputMode::kDrm) {
		_InitOutputSink(result->frame.cols, result->frame.rows);
		ScopedTimer t("runMODNet: display", config_.timing_enabled, logger, kModnetModuleName);
		_Display(alpha, result->frame.cols, result->frame.rows);
	}
	// MP4 mode: image already saved by BackEnd::Postprocess()

	// 5. Cleanup DRM
	if (drm_display_.IsOpen()) {
		drm_display_.Close();
	}

	return 0;
}
