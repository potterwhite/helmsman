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

#include "pipeline/stages/backend/backend.h"
#include "common/common-define.h"

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

cv::Mat MattingBackend::postprocess(const std::vector<TensorData>& outputs) {
	return postprocess(outputs, cv::Mat{});
}

cv::Mat MattingBackend::postprocess(const std::vector<TensorData>& outputs,
                                    const cv::Mat& guide_bgr_override) {

	const int current_frame = process_count_++;  // 0-indexed frame number

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils = arcforge::utils::FileUtils::GetInstance();

	if (outputs.empty()) {
		throw std::runtime_error("Backend received empty outputs");
	}

	// -------------------------
	// Select the pha (alpha matte) tensor from outputs.
	// Strategy: first try by name ("pha"), then fall back to positional convention.
	//   - MODNet: outputs[0] = pha (only output)
	//   - RVM:    outputs[0] = fgr, outputs[1] = pha
	const TensorData* pha_ptr = nullptr;
	for (const auto& td : outputs) {
		if (td.name == "pha") {
			pha_ptr = &td;
			break;
		}
	}
	if (!pha_ptr) {
		// Fall back: if single output, use [0]; otherwise use [1] (RVM convention)
		pha_ptr = (outputs.size() == 1) ? &outputs[0] : &outputs[1];
	}

	const TensorData& output = *pha_ptr;

	logger.Info("Backend: selected pha tensor '" + output.name + "' from " +
	            std::to_string(outputs.size()) + " outputs", kcurrent_module_name);

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
	if (isDumpEnabled()) {
		file_utils.dumpBinary(output.data, output_path_ + "/cpp_09_backend_input.bin");
	}

	// -----------------------------------
	// Safe cast for OpenCV (explicit!)
	const int H_int = static_cast<int>(H);
	const int W_int = static_cast<int>(W);
	const int C_int = static_cast<int>(C);

	if (H_int <= 0 || W_int <= 0) {
		throw std::runtime_error("Invalid output shape");
	}

	const std::vector<float>& data = output.data;

	// -------------------------
	// 1. Convert NCHW float32 → HWC cv::Mat
	//
	// For C==1 (alpha matte): NCHW is already a contiguous H×W float array.
	//   → single memcpy, no loop needed.
	// For C==3 (fgr): merge three planar float channels into interleaved HWC.
	//   → cv::merge() is SIMD-optimised; much faster than at<>() indexing.
	cv::Mat result;

	if (C_int == 1) {
		// Alpha matte: plane 0 IS the H×W float array → wrap without copy
		result = cv::Mat(H_int, W_int, CV_32FC1,
		                 const_cast<float*>(data.data())).clone();
	} else {
		// Multi-channel: split planes and merge
		std::vector<cv::Mat> planes(static_cast<size_t>(C_int));
		const size_t plane_size = static_cast<size_t>(H_int) * static_cast<size_t>(W_int);
		for (size_t c = 0; c < static_cast<size_t>(C_int); ++c) {
			planes[c] = cv::Mat(H_int, W_int, CV_32FC1,
			                    const_cast<float*>(data.data() + c * plane_size));
		}
		cv::merge(planes, result);
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
	// guide priority: caller-supplied frame (video mode) > imread from path (image mode).

	// --- Diagnostics §10: per-frame pha mean log + frame 50-70 PNG dump ---
	{
		cv::Scalar mean_val = cv::mean(restored_mat);
		logger.Info("PHA_MEAN frame=" + std::to_string(current_frame) +
		            " mean=" + std::to_string(mean_val[0]), kcurrent_module_name);

		if (current_frame >= 50 && current_frame <= 70 && !output_path_.empty()) {
			cv::Mat pha_debug_u8;
			restored_mat.convertTo(pha_debug_u8, CV_8UC1, 255.0);
			char fname[64];
			snprintf(fname, sizeof(fname), "/debug_pha_rknn_frame%04d.png", current_frame);
			cv::imwrite(output_path_ + fname, pha_debug_u8);
		}
	}
	if (post_processor_) {
		cv::Mat guide;
		if (!guide_bgr_override.empty()) {
			guide = guide_bgr_override;  // video mode: use the decoded frame directly
		} else {
			guide = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);  // image mode
		}
		if (guide.empty()) {
			logger.Warning(
			    "Post-processor skipped: guide image unavailable (path='" +
			        foreground_image_path_ + "')",
			    kcurrent_module_name);
		} else {
			restored_mat = post_processor_->process(restored_mat, guide);
		}
	}

	if (isDumpEnabled()) {
		const size_t total = static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C);
		file_utils.dumpBinary(std::vector<float>((float*)clamped.data, (float*)clamped.data + total),
		                      output_path_ + "/cpp_10_clamped.bin");
	}

	// -------------------------
	// 3. convert to 8bit image
	cv::Mat output_8u;
	// clamped.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);
	restored_mat.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);

	if (isDumpEnabled()) {
		cv::imwrite(output_path_ + "/cpp_11_result.png", output_8u);
	}

	// -------------------------
	// 4. Composite foreground over background (cpp_12_composed.jpg)
	//    Only executed when BOTH foreground and background paths are provided.
	//    For video mode, compositing is handled by Pipeline::compositeAndWrite().
	if (!background_path_.empty() && !foreground_image_path_.empty()) {
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
