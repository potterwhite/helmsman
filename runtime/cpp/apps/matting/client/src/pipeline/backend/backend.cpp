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

#include "pipeline/backend/backend.h"
#include "common-define.h"

// #include <algorithm>
// #include <cmath>

MattingBackend::MattingBackend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("MattingBackend object constructed.",
	                                                      kcurrent_module_name);
}

MattingBackend::~MattingBackend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("MattingBackend cleaned up.",
	                                                      kcurrent_module_name);
}

void MattingBackend::setOutputPath(const std::string& path) {
	output_path_ = path;
}

void MattingBackend::setBackgroundPath(const std::string& path) {
	background_path_ = path;
}

void MattingBackend::setForegroundImagePath(const std::string& path) {
	foreground_image_path_ = path;
}

void MattingBackend::setPostProcessor(std::shared_ptr<IPostProcessor> processor) {
	post_processor_ = std::move(processor);
}

cv::Mat MattingBackend::postprocess(const TensorData& output) {

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils = arcforge::utils::FileUtils::GetInstance();

	if (output.shape.size() != 4) {
		throw std::runtime_error("Output tensor must be NCHW");
	}

	const int64_t N = output.shape[0];
	const int64_t C = output.shape[1];
	const int64_t H = output.shape[2];
	const int64_t W = output.shape[3];

	if (N != 1) {
		throw std::runtime_error("Batch size must be 1");
	}

	logger.Info("Backend processing: N=" + std::to_string(N) + " C=" + std::to_string(C) +
	                " H=" + std::to_string(H) + " W=" + std::to_string(W),
	            kcurrent_module_name);

	// -------------------------
	// dump raw output for debug
	file_utils.dumpBinary(output.data, output_path_ + "/cpp_09_backend_input.bin");

	// -----------------------------------
	// Safe cast for OpenCV (explicit!)
	const int H_int = static_cast<int>(H);
	const int W_int = static_cast<int>(W);
	const int C_int = static_cast<int>(C);

	if (H_int <= 0 || W_int <= 0) {
		throw std::runtime_error("Invalid output shape");
	}

	// -------------------------
	// 1. convert NCHW -> HWC
	cv::Mat result(H_int, W_int, (C_int == 1 ? CV_32FC1 : CV_32FC3));

	const std::vector<float>& data = output.data;

	const size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);

	for (int c = 0; c < C; ++c) {
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {

				// size_t nchw_index = c * H * W + h * W + w;
				const size_t nchw_index = static_cast<size_t>(c) * HW +
				                          static_cast<size_t>(h) * static_cast<size_t>(W) +
				                          static_cast<size_t>(w);

				float value = data[nchw_index];

				if (C == 1) {
					// result.at<float>(h, w) = value;
					result.at<float>(static_cast<int>(h), static_cast<int>(w)) = value;
				} else {
					// result.at<cv::Vec3f>(h, w)[c] = value;
					result.at<cv::Vec3f>(static_cast<int>(h),
					                     static_cast<int>(w))[static_cast<int>(c)] = value;
				}
			}
		}
	}

	// -------------------------
	// 2. clamp to [0,1]
	cv::Mat clamped;
	cv::min(cv::max(result, 0.0f), 1.0f, clamped);

	// --- ADD THIS BLOCK TO RESTORE ORIGINAL SHAPE ---
	// Step A: Crop out the letterbox padding
	int valid_width = clamped.cols - output.pad_left - output.pad_right;
	int valid_height = clamped.rows - output.pad_top - output.pad_bottom;

	cv::Rect roi(output.pad_left, output.pad_top, valid_width, valid_height);
	cv::Mat cropped_mat = clamped(roi);

	// Step B: Resize the cropped image back to the original image dimensions
	cv::Mat restored_mat;
	cv::resize(cropped_mat, restored_mat, cv::Size(output.orig_width, output.orig_height), 0, 0,
	           cv::INTER_LINEAR);  // or cv::INTER_CUBIC for better alpha edge quality
	                               // ------------------------------------------------

	// Step C: Optional post-processing (e.g. Guided Filter edge refinement).
	// Attach a processor via setPostProcessor(); leave nullptr to skip.
	if (post_processor_) {
		cv::Mat guide_bgr = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);
		if (guide_bgr.empty()) {
			logger.Warning(
			    "Post-processor skipped: cannot load guide image: " + foreground_image_path_,
			    kcurrent_module_name);
		} else {
			restored_mat = post_processor_->process(restored_mat, guide_bgr);
		}
	}

	const size_t total = HW * static_cast<size_t>(C);

	file_utils.dumpBinary(std::vector<float>((float*)clamped.data, (float*)clamped.data + total),
	                      output_path_ + "/cpp_10_clamped.bin");

	// -------------------------
	// 3. convert to 8bit image
	cv::Mat output_8u;
	// clamped.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);
	restored_mat.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);

	cv::imwrite(output_path_ + "/cpp_11_result.png", output_8u);

	// -------------------------
	// 4. Composite foreground over background (cpp_12_composed.jpg)
	//    Only executed when a background image path is provided.
	if (!background_path_.empty()) {
		// --- Load original foreground (BGR) ---
		cv::Mat fg_bgr = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);
		if (fg_bgr.empty()) {
			logger.Warning("Cannot load foreground image: " + foreground_image_path_ +
			                   ", skipping composition.",
			               kcurrent_module_name);
		} else {
			// --- Load background (BGR) and resize to match foreground ---
			cv::Mat bg_bgr_raw = cv::imread(background_path_, cv::IMREAD_COLOR);
			if (bg_bgr_raw.empty()) {
				logger.Warning("Cannot load background image: " + background_path_ +
				                   ", skipping composition.",
				               kcurrent_module_name);
			} else {
				cv::Mat bg_bgr;
				cv::resize(bg_bgr_raw, bg_bgr, cv::Size(fg_bgr.cols, fg_bgr.rows),
				           0, 0, cv::INTER_LINEAR);

				// --- Get alpha mask (output_8u is the matting result) ---
				// output_8u may be CV_8UC1 (alpha only) or CV_8UC3
				cv::Mat alpha_8u;
				if (output_8u.channels() == 1) {
					alpha_8u = output_8u;
				} else {
					// If multi-channel, use first channel as alpha
					std::vector<cv::Mat> channels;
					cv::split(output_8u, channels);
					alpha_8u = channels[0];
				}

				// Resize alpha to foreground size if needed
				if (alpha_8u.size() != fg_bgr.size()) {
					cv::resize(alpha_8u, alpha_8u, fg_bgr.size(), 0, 0, cv::INTER_LINEAR);
				}

				// --- Alpha Compositing: C = alpha * F + (1 - alpha) * B ---
				cv::Mat alpha_f32, alpha_3ch;
				alpha_8u.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);
				cv::cvtColor(alpha_f32, alpha_3ch, cv::COLOR_GRAY2BGR);  // broadcast to 3ch

				cv::Mat fg_f32, bg_f32;
				fg_bgr.convertTo(fg_f32, CV_32FC3, 1.0 / 255.0);
				bg_bgr.convertTo(bg_f32, CV_32FC3, 1.0 / 255.0);

				cv::Mat composed_f32 = alpha_3ch.mul(fg_f32) +
				                      (cv::Scalar(1.0, 1.0, 1.0) - alpha_3ch).mul(bg_f32);

				cv::Mat composed_8u;
				composed_f32.convertTo(composed_8u, CV_8UC3, 255.0);

				cv::imwrite(output_path_ + "/cpp_12_composed.jpg", composed_8u);
				logger.Info("Background composition saved: cpp_12_composed.jpg",
				            kcurrent_module_name);
			}
		}
	}



	return output_8u;
}
