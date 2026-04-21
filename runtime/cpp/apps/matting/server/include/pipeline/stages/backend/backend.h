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
#include "pipeline/stages/backend/post-processor/i-post-processor.h"
#include "common/data_structure.h"

class MattingBackend {
   public:
	MattingBackend();
	~MattingBackend();

	void setOutputPath(const std::string& path);
	void setBackgroundPath(const std::string& path);
	void setForegroundImagePath(const std::string& path);

	/**
	 * Attach an optional post-processor (e.g. GuidedFilterPostProcessor).
	 * Pass nullptr to disable post-processing and use the raw alpha mask.
	 */
	void setPostProcessor(std::shared_ptr<IPostProcessor> processor);

	// Accept multi-tensor output from InferenceEngine.
	// Selects the pha (alpha matte) tensor by name or position:
	//   - MODNet: outputs[0] = pha
	//   - RVM:    outputs[1] = pha (outputs[0] = fgr)
	cv::Mat postprocess(const std::vector<TensorData>& outputs);

	/**
	 * Video-mode overload: caller supplies the original BGR frame directly
	 * so the post-processor can use it as a guide without a filesystem round-trip.
	 * When guide_bgr is non-empty it takes priority over foreground_image_path_.
	 */
	cv::Mat postprocess(const std::vector<TensorData>& outputs, const cv::Mat& guide_bgr);

   private:
	std::string output_path_;
	std::string background_path_;
	std::string foreground_image_path_;

	std::shared_ptr<IPostProcessor> post_processor_;  // nullptr = no post-processing

	cv::Mat nchwToHwc(const TensorData& tensor);
};
