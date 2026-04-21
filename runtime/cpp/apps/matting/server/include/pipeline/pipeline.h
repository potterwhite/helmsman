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
#include <opencv2/core.hpp>
#include "Utils/logger/logger.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/recurrent-state-manager.h"
#include "pipeline/stages/frontend.h"
#include "input/input-source.h"

// ---------------------------------------------------------------------------
// ModelType — explicit model type selection
//
// The caller (main-client) chooses which model type to run.
// InferenceEngine itself is model-agnostic (N→M tensor).
// Pipeline uses this enum to decide:
//   - Whether to initialize RecurrentStateManager
//   - How many benchmark iterations to run
//   - Which backend post-processing path to take
// ---------------------------------------------------------------------------
enum class ModelType {
	kMODNet,  // Single-frame matting (1 input → 1 output)
	kRVM,     // Video matting with recurrent state (5 inputs → 6 outputs)
};

class Pipeline {
   public:
	static Pipeline& GetInstance();
	void init(std::unique_ptr<InputSource> input_source, const std::string& model_path,
	          const std::string& output_bin_path, const std::string& background_path = "",
	          ModelType model_type = ModelType::kRVM);

	// single picture init
	void init(const std::string& image_path, const std::string& model_path,
	          const std::string& output_bin_path, const std::string& background_path = "",
	          ModelType model_type = ModelType::kMODNet);

	int run();

	// Timing switch — controlled by the --timing=off CLI flag.
	// Enabled by default. Call before run().
	void setTimingEnabled(bool enabled) { timing_enabled_ = enabled; }
	bool isTimingEnabled() const        { return timing_enabled_; }

   private:
	Pipeline();
	~Pipeline();

	void verify_parameters_necessary();

	// Factory: constructs the correct InferenceEngine based on compile-time backend selection.
	// The #ifdef lives here (in the .cpp implementation) and nowhere else.
	static std::unique_ptr<InferenceEngine> make_engine();

	// MODNet path: single-frame inference with benchmarking
	int runMODNet();

	// RVM path: video inference with recurrent state + dual-buffer prefetch
	int runRVM();

	// RVM path: single-picture loop test (development / integration test only)
	int runRVM_CV_SinglePicture();

	// RVM helpers — each covers one distinct responsibility in the video loop
	void initRVMRecurrentStates(size_t model_input_height, size_t model_input_width);
	bool openVideoWriter(cv::VideoWriter& writer, const std::string& path,
	                     int width, int height, double fps);
	cv::Mat loadOrCreateBackground(int width, int height);
	cv::Mat inferOneFrame(const TensorData& src);
	void compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
	                       const cv::Mat& alpha_8u, const cv::Mat& bg_bgr);

   private:
	std::string input_image_path_;
	std::unique_ptr<InputSource> input_source_;
	std::string model_path_;
	std::string output_bin_path_;
	std::string background_path_;
	ModelType model_type_ = ModelType::kMODNet;

	bool timing_enabled_ = true;  // disabled by --timing=off CLI flag

	std::unique_ptr<InferenceEngine> engine_;   // created by make_engine() in init()
	ImageFrontend frontend_;
	ImageFrontend prefetch_frontend_;           // separate instance for async prefetch
	MattingBackend backend_;                    // video-mode backend (no GuidedFilter)
	RecurrentStateManager state_mgr_;
};
