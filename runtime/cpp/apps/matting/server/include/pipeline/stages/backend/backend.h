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
#include <opencv2/opencv.hpp>
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"
#include "Utils/timing/timer.h"
#include "pipeline/stages/backend/post-processor/base-post-processor.h"
#include "common/types.h"

class MattingBackend {
   public:
	MattingBackend();
	~MattingBackend();

	void SetOutputPath(const std::string& path);
	void SetBackgroundPath(const std::string& path);
	void SetForegroundImagePath(const std::string& path);
	void SetDumpEnabled(bool enabled);

	/**
	 * Attach an optional post-processor (e.g. GuidedFilterPostProcessor).
	 * Pass nullptr to disable post-processing and use the raw alpha mask.
	 */
	void SetPostProcessor(std::shared_ptr<BasePostProcessor> processor);

	// Timing accessors (for RVMMode accumulated stats reporting)
	const helmsman::utils::timing::StageAccumulator& postprocess_acc() const { return acc_postprocess_; }
	const helmsman::utils::timing::StageAccumulator& composite_acc() const { return acc_composite_; }

	// --- Video compositing setup ---

	/**
	 * Set the pre-computed background image at model resolution (CV_8UC3).
	 * Used by Composite() for alpha blending.
	 */
	void SetBackgroundModelImage(const cv::Mat& bg);

	// --- Inference post-processing ---

	// Accept multi-tensor output from InferenceEngine.
	// Selects the pha (alpha matte) tensor by name or position:
	//   - MODNet: outputs[0] = pha
	//   - RVM (original): outputs = [pha, r1o, r2o, r3o, r4o]
	cv::Mat Postprocess(const std::vector<TensorData>& outputs);

	/**
	 * Video-mode overload: caller supplies the original BGR frame directly
	 * so the post-processor can use it as a guide without a filesystem round-trip.
	 * When guide_bgr is non-empty it takes priority over foreground_image_path_.
	 */
	cv::Mat Postprocess(const std::vector<TensorData>& outputs, const cv::Mat& guide_bgr);

	/**
	 * No-resize RVM overload: caller supplies the src input tensor (model input)
	 * for guided filter combination. Used with no-resize RKNN model where
	 * outputs contain A(777) + b(779) + r1o~r4o instead of pha + fgr.
	 */
	cv::Mat Postprocess(const std::vector<TensorData>& outputs,
	                    const cv::Mat& guide_bgr,
	                    const TensorData& src_tensor);

	// --- Video compositing ---

	/**
	 * Composite: resize alpha + resize frame + alpha blend + upscale.
	 *
	 * Steps:
	 *   1. Resize alpha_8u to (model_w × model_h) — CPU (RGA doesn't support YUV400)
	 *   2. Resize frame to (model_w × model_h) — RGA hardware (fallback: cv::resize)
	 *   3. Alpha blend: fg * alpha + bg * (1-alpha) — CPU
	 *   4. Upscale composed to (output_w × output_h) — RGA hardware (fallback: cv::resize)
	 *
	 * @param frame     Original BGR frame from frontend (any resolution)
	 * @param alpha_8u  Alpha matte from Postprocess() (original resolution, CV_8UC1)
	 * @param model_w   Model input width (blend resolution)
	 * @param model_h   Model input height (blend resolution)
	 * @param output_w  Final output width (upscale target)
	 * @param output_h  Final output height (upscale target)
	 * @return Composited BGR frame at (output_w × output_h), CV_8UC3
	 */
	cv::Mat Composite(const cv::Mat& frame, const cv::Mat& alpha_8u,
	                   int model_w, int model_h, int output_w, int output_h);

   private:
	std::string output_path_;
	std::string background_path_;
	std::string foreground_image_path_;
	bool dump_enabled_ = false;

	std::shared_ptr<BasePostProcessor> post_processor_;  // nullptr = no post-processing
	int process_count_ = 0;  // counts postprocess() calls; used for per-frame debug dump

	// Video compositing resources (set via setters)
	cv::Mat bg_model_u8_;  // Background at model resolution (CV_8UC3)

	// Timing accumulators (top-level, reported by RVMMode)
	using sa = helmsman::utils::timing::StageAccumulator;
	sa acc_postprocess_{"  Lv03-04-01::mainloop::backend::postprocess"};
	sa acc_composite_{"  Lv03-04-02::mainloop::backend::composite"};

	cv::Mat nchwToHwc(const TensorData& tensor);
};
