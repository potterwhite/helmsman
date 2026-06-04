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

#include <cassert>
#include <cstdio>
#include "RGAKit/rga_resize.h"
#include "common/common-define.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;
using helmsman::rgakit::RgaResize;

// #include <algorithm>
// #include <cmath>

MattingBackend::MattingBackend() {
	helmsman::utils::Logger::GetInstance().Info("MattingBackend object constructed.",
	                                                      kcurrent_module_name);
}

MattingBackend::~MattingBackend() {
	helmsman::utils::Logger::GetInstance().Info("MattingBackend cleaned up.",
	                                                      kcurrent_module_name);
}

void MattingBackend::SetAppConfig(const AppConfig& config) {
	config_ = &config;
}

void MattingBackend::SetForegroundImagePath(const std::string& path) {
	foreground_image_path_ = path;
}

void MattingBackend::SetPostProcessor(std::shared_ptr<BasePostProcessor> processor) {
	post_processor_ = std::move(processor);
}

cv::Mat MattingBackend::Postprocess(const std::vector<TensorData>& outputs) {
	return Postprocess(outputs, cv::Mat{});
}

cv::Mat MattingBackend::Postprocess(const std::vector<TensorData>& outputs,
                                    const cv::Mat& guide_bgr_override) {
	helmsman::utils::timing::ManualTimer t;
	t.start();

	const int current_frame = process_count_++;  // 0-indexed frame number

	auto& logger = helmsman::utils::Logger::GetInstance();
	auto& file_utils = helmsman::utils::FileUtils::GetInstance();

	if (outputs.empty()) {
		throw std::runtime_error("Backend received empty outputs");
	}

	// -------------------------
	// Select the pha (alpha matte) tensor from outputs.
	// Strategy: first try by name ("pha"), then fall back to first output.
	//   - MODNet: outputs[0] = pha (only output)
	//   - RVM:    fgr is skipped at ReadOutputBuffers3rd (§5.5),
	//             so outputs = [pha, r1o, r2o, r3o, r4o]
	const TensorData* pha_ptr = nullptr;
	for (const auto& td : outputs) {
		if (td.name == "pha") {
			pha_ptr = &td;
			break;
		}
	}
	if (!pha_ptr) {
		throw std::runtime_error(
		    "Backend: output tensor 'pha' not found in " +
		    std::to_string(outputs.size()) + " outputs");
	}

	const TensorData& output = *pha_ptr;

	if (config_->dump_enabled)
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

	if (config_->dump_enabled)
		logger.Info("Backend processing: N=" + std::to_string(N) + " C=" + std::to_string(C) +
		                " H=" + std::to_string(H) + " W=" + std::to_string(W),
		            kcurrent_module_name);

	// -------------------------
	// dump raw output for debug
	if (config_->dump_enabled) {
		file_utils.dumpBinary(output.data, config_->output_bin_path + "/cpp_09_backend_input.bin");
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
	// Attach a processor via SetPostProcessor(); leave nullptr to skip.
	// guide priority: caller-supplied frame (video mode) > imread from path (image mode).

	// --- Diagnostics §10: per-frame pha mean log + frame 50-70 PNG dump ---
	{
		cv::Scalar mean_val = cv::mean(restored_mat);
		if (config_->dump_enabled)
			logger.Info("PHA_MEAN frame=" + std::to_string(current_frame) +
			            " mean=" + std::to_string(mean_val[0]), kcurrent_module_name);

		if (current_frame >= 50 && current_frame <= 70 && !config_->output_bin_path.empty()) {
			cv::Mat pha_debug_u8;
			restored_mat.convertTo(pha_debug_u8, CV_8UC1, 255.0);
			char fname[64];
			snprintf(fname, sizeof(fname), "/debug_pha_rknn_frame%04d.png", current_frame);
			cv::imwrite(config_->output_bin_path + fname, pha_debug_u8);
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

	if (config_->dump_enabled) {
		const size_t total = static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C);
		file_utils.dumpBinary(std::vector<float>((float*)clamped.data, (float*)clamped.data + total),
		                      config_->output_bin_path + "/cpp_10_clamped.bin");
	}

	// -------------------------
	// 3. convert to 8bit image
	cv::Mat output_8u;
	// clamped.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);
	restored_mat.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);

	if (config_->dump_enabled) {
		cv::imwrite(config_->output_bin_path + "/cpp_11_result.png", output_8u);
	}

	// -------------------------
	// 4. Composite foreground over background (cpp_12_composed.jpg)
	//    Only executed when BOTH foreground and background paths are provided.
	//    For video mode, compositing is handled by Pipeline::_compositeAndWrite().
	if (!config_->background_path.empty() && !foreground_image_path_.empty()) {
		// --- Load original foreground (BGR) ---
		cv::Mat fg_bgr = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);
		if (fg_bgr.empty()) {
			logger.Warning("Cannot load foreground image: " + foreground_image_path_ +
			                   ", skipping composition.",
			               kcurrent_module_name);
		} else {
			// --- Load background (BGR) and resize to match foreground ---
			cv::Mat bg_bgr_raw = cv::imread(config_->background_path, cv::IMREAD_COLOR);
			if (bg_bgr_raw.empty()) {
				logger.Warning("Cannot load background image: " + config_->background_path +
				                   ", skipping composition.",
				               kcurrent_module_name);
			} else {
				cv::Mat bg_bgr;
				bg_bgr.create(fg_bgr.rows, fg_bgr.cols, CV_8UC3);
				ImageDescriptor bg_src(bg_bgr_raw.data, bg_bgr_raw.cols, bg_bgr_raw.rows, RgaPixelFormat::kBgr888);
				ImageDescriptor bg_dst(bg_bgr.data, fg_bgr.cols, fg_bgr.rows, RgaPixelFormat::kBgr888);
				if (!RgaResize::Instance().Execute(bg_src, bg_dst)) {
					fprintf(stderr, "[FATAL] RGA resize failed for background — hardware error\n");
					std::abort();
				}

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

				cv::imwrite(config_->output_bin_path + "/cpp_12_composed.jpg", composed_8u);
				logger.Info("Background composition saved: cpp_12_composed.jpg",
				            kcurrent_module_name);
			}
		}
	}

	acc_postprocess_.record(t.stop());
	return output_8u;
}

// ============================================================================
// Postprocess — no-resize RVM model (low-res guided filter combination)
//
// Background:
//   RVM's DeepGuidedFilterRefiner refines coarse decoder output to full
//   resolution. In the "no-resize" ONNX model, the refiner's final bilinear
//   upsampling + combination was removed from the NPU graph. The NPU outputs
//   A and b at low resolution (272×480).
//
//   Original approach: upsample A(4ch) and b(4ch) to full res, compute
//   guided filter at 544×960. This was slow (~25ms for float32 resize).
//
//   Optimized approach: downsample src to 272×480 via RGA hardware, compute
//   guided filter at low res (A, b, fine_x all at 272×480), then upsample
//   only the 1ch alpha result. Eliminates the 4ch+4ch float32 upsampling.
//
// Guided filter formula:
//   out = A * fine_x + b
//   fine_x = [R, G, B, mean(R,G,B)]  (4ch guided image)
//   pha = clip(out[:,:,3], 0, 1)      (extract alpha channel)
//
// References:
//   - He et al., "Guided Image Filtering", IEEE TPAMI 2013
//   - deep_guided_filter.py (lines 24-43)
// ============================================================================
cv::Mat MattingBackend::Postprocess(const std::vector<TensorData>& outputs,
                                    const cv::Mat& guide_bgr_override,
                                    const TensorData& src_tensor) {
	helmsman::utils::timing::ManualTimer t;
	t.start();

	const int current_frame = process_count_++;

	auto& logger = helmsman::utils::Logger::GetInstance();

	// --- Find A and b tensors by name ---
	const TensorData* td_A = nullptr;
	const TensorData* td_b = nullptr;
	for (const auto& td : outputs) {
		if (td.name == "777") td_A = &td;
		if (td.name == "779") td_b = &td;
	}
	if (!td_A || !td_b) {
		throw std::runtime_error(
		    "Backend(no-resize): A('777') or b('779') not found in " +
		    std::to_string(outputs.size()) + " outputs");
	}

	// --- Validate shapes ---
	// A/b: NCHW (RKNN output), src: NHWC (Preprocessor stores HWC layout)
	if (td_A->shape.size() != 4 || td_b->shape.size() != 4 ||
	    td_A->shape[1] != 4 || td_b->shape[1] != 4) {
		throw std::runtime_error("Backend(no-resize): A and b must be NCHW with C=4");
	}
	if (src_tensor.shape.size() != 4 || src_tensor.shape[3] != 3) {
		throw std::runtime_error("Backend(no-resize): src must be NHWC with C=3");
	}

	const int dH = static_cast<int>(td_A->shape[2]);
	const int dW = static_cast<int>(td_A->shape[3]);
	const int H = static_cast<int>(src_tensor.shape[1]);
	const int W = static_cast<int>(src_tensor.shape[2]);

	if (config_->dump_enabled)
		logger.Info("Backend(no-resize): A=" + std::to_string(dH) + "x" + std::to_string(dW) +
		                " src=" + std::to_string(H) + "x" + std::to_string(W),
		            kcurrent_module_name);

	// --- 1. NCHW → HWC for A and b (4 channels) ---
	// A and b are the outputs of DeepGuidedFilterRefiner's learned linear model.
	// A: 4ch coefficients (one per guided channel: R, G, B, gray_mean)
	// b: 4ch bias
	// Both at low resolution (dH×dW = 272×480), NCHW layout from RKNN.
	const size_t plane_size_d = static_cast<size_t>(dH) * static_cast<size_t>(dW);

	auto nchw4_to_hwc4 = [&](const std::vector<float>& data) -> cv::Mat {
		std::vector<cv::Mat> planes(4);
		for (size_t c = 0; c < 4; ++c) {
			planes[c] = cv::Mat(dH, dW, CV_32FC1,
			                    const_cast<float*>(data.data() + c * plane_size_d));
		}
		cv::Mat result;
		cv::merge(planes, result);
		return result;
	};

	cv::Mat A_hwc = nchw4_to_hwc4(td_A->data);  // CV_32FC4, dH×dW
	cv::Mat b_hwc = nchw4_to_hwc4(td_b->data);  // CV_32FC4, dH×dW

	if (config_->diag_enabled)
		logger.Info("Backend(no-resize) data: A.data=" + std::to_string(td_A->data.size()) +
		            " b.data=" + std::to_string(td_b->data.size()) +
		            " src.data=" + std::to_string(src_tensor.data.size()) +
		            " A_hwc=" + std::to_string(A_hwc.cols) + "x" + std::to_string(A_hwc.rows) +
		            "ch" + std::to_string(A_hwc.channels()) +
		            " b_hwc=" + std::to_string(b_hwc.cols) + "x" + std::to_string(b_hwc.rows) +
		            "ch" + std::to_string(b_hwc.channels()),
		            kcurrent_module_name);

	// =========================================================================
	// Low-resolution guided filter combination (optimized path)
	//
	// Instead of upsampling A(4ch) and b(4ch) to full resolution and computing
	// the guided filter at 544×960, we downsample src to 272×480, compute the
	// guided filter at low resolution, and only upsample the 1ch alpha result.
	//
	// Cost comparison:
	//   Old: resize A(4ch)+b(4ch) 544×960 + mul/add 4ch@544×960 ≈ 25ms
	//   New: resize src(3ch) 272×480 + mul/add 4ch@272×480 + resize pha(1ch) ≈ 3ms
	// =========================================================================

	// --- 2. Downsample src to low resolution (dH×dW) via RGA hardware ---
	// Preprocessor stores data as HWC interleaved float32 [0,255].
	// Convert to uint8 for RGA, resize, then convert back to float32 [0,1].
	cv::Mat src_u8(H, W, CV_8UC3);
	{
		cv::Mat src_hwc(H, W, CV_32FC3,
		                const_cast<float*>(src_tensor.data.data()));
		src_hwc.convertTo(src_u8, CV_8UC3);  // float32 [0,255] → uint8
	}

	cv::Mat src_lr_u8(dH, dW, CV_8UC3);
	{
		ImageDescriptor rga_src(src_u8.data, W, H, RgaPixelFormat::kBgr888);
		ImageDescriptor rga_dst(src_lr_u8.data, dW, dH, RgaPixelFormat::kBgr888);
		if (!RgaResize::Instance().Execute(rga_src, rga_dst)) {
			fprintf(stderr, "[FATAL] RGA resize failed for src downsample\n");
			std::abort();
		}
	}

	// Convert downsampled src to float32 [0,1] for guided filter
	cv::Mat src_lr_f32;
	src_lr_u8.convertTo(src_lr_f32, CV_32FC3, 1.0 / 255.0);

	// --- 3. Compute fine_x_lr = [src_lr, mean(src_lr)] at low resolution ---
	cv::Mat src_lr_mean;
	{
		std::vector<cv::Mat> ch(3);
		cv::split(src_lr_f32, ch);
		src_lr_mean = (ch[0] + ch[1] + ch[2]) / 3.0f;  // CV_32FC1, dH×dW
	}

	cv::Mat fine_x_lr;
	{
		std::vector<cv::Mat> src_ch(3);
		cv::split(src_lr_f32, src_ch);
		std::vector<cv::Mat> fine_ch = {src_ch[0], src_ch[1], src_ch[2], src_lr_mean};
		cv::merge(fine_ch, fine_x_lr);  // CV_32FC4, dH×dW
	}

	// --- 4. Guided filter combination at low resolution ---
	// out_lr = A * fine_x_lr + b  (all at dH×dW, no upsampling needed!)
	cv::Mat out_lr = A_hwc.mul(fine_x_lr) + b_hwc;  // CV_32FC4, dH×dW

	// Extract pha channel only (fgr is not used — we only return alpha)
	std::vector<cv::Mat> out_ch(4);
	cv::split(out_lr, out_ch);
	cv::Mat pha_lr;
	cv::min(cv::max(out_ch[3], 0.0f), 1.0f, pha_lr);  // clip to [0,1]

	// --- 5. Upsample alpha to model resolution (1ch float32) ---
	cv::Mat pha_model;
	cv::resize(pha_lr, pha_model, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

	// --- 6. Crop letterbox padding + resize to original resolution (uint8) ---
	// Note: RGA does not support kYuv400 for resize, so we use CPU here.
	// The uint8 crop+resize is fast (~1ms) — the main savings come from
	// the low-res guided filter (step 2-5) and RGA src downsample (step 2).
	const int valid_w = W - src_tensor.pad_left - src_tensor.pad_right;
	const int valid_h = H - src_tensor.pad_top - src_tensor.pad_bottom;
	cv::Mat pha_model_u8;
	pha_model.convertTo(pha_model_u8, CV_8UC1, 255.0);
	cv::Rect roi(src_tensor.pad_left, src_tensor.pad_top, valid_w, valid_h);
	cv::Mat cropped = pha_model_u8(roi);
	cv::Mat output_8u;
	cv::resize(cropped, output_8u,
	           cv::Size(src_tensor.orig_width, src_tensor.orig_height),
	           0, 0, cv::INTER_LINEAR);

	// --- 8. Diagnostics ---
	{
		cv::Scalar mean_val = cv::mean(output_8u);
		if (config_->dump_enabled)
			logger.Info("PHA_MEAN frame=" + std::to_string(current_frame) +
			            " mean=" + std::to_string(mean_val[0] / 255.0), kcurrent_module_name);

		if (current_frame >= 50 && current_frame <= 70 && !config_->output_bin_path.empty()) {
			char fname[64];
			snprintf(fname, sizeof(fname), "/debug_pha_rknn_frame%04d.png", current_frame);
			cv::imwrite(config_->output_bin_path + fname, output_8u);
		}
	}

	// --- 9. Post-processing (GuidedFilter edge refinement) ---
	// The post-processor operates on float32 alpha, so convert back temporarily.
	if (post_processor_) {
		cv::Mat guide;
		if (!guide_bgr_override.empty()) {
			guide = guide_bgr_override;
		} else {
			guide = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);
		}
		if (guide.empty()) {
			logger.Warning(
			    "Post-processor skipped: guide image unavailable (path='" +
			        foreground_image_path_ + "')",
			    kcurrent_module_name);
		} else {
			cv::Mat alpha_f32;
			output_8u.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);
			alpha_f32 = post_processor_->process(alpha_f32, guide);
			alpha_f32.convertTo(output_8u, CV_8UC1, 255.0);
		}
	}

	if (config_->dump_enabled) {
		cv::imwrite(config_->output_bin_path + "/cpp_11_result.png", output_8u);
	}

	// --- 14. Background composition (same as original Postprocess) ---
	if (!config_->background_path.empty() && !foreground_image_path_.empty()) {
		cv::Mat fg_bgr = cv::imread(foreground_image_path_, cv::IMREAD_COLOR);
		if (fg_bgr.empty()) {
			logger.Warning("Cannot load foreground image: " + foreground_image_path_ +
			                   ", skipping composition.",
			               kcurrent_module_name);
		} else {
			cv::Mat bg_bgr_raw = cv::imread(config_->background_path, cv::IMREAD_COLOR);
			if (bg_bgr_raw.empty()) {
				logger.Warning("Cannot load background image: " + config_->background_path +
				                   ", skipping composition.",
				               kcurrent_module_name);
			} else {
				cv::Mat bg_bgr;
				bg_bgr.create(fg_bgr.rows, fg_bgr.cols, CV_8UC3);
				ImageDescriptor bg_src(bg_bgr_raw.data, bg_bgr_raw.cols, bg_bgr_raw.rows, RgaPixelFormat::kBgr888);
				ImageDescriptor bg_dst(bg_bgr.data, fg_bgr.cols, fg_bgr.rows, RgaPixelFormat::kBgr888);
				if (!RgaResize::Instance().Execute(bg_src, bg_dst)) {
					fprintf(stderr, "[FATAL] RGA resize failed for background — hardware error\n");
					std::abort();
				}

				cv::Mat alpha_8u;
				if (output_8u.channels() == 1) {
					alpha_8u = output_8u;
				} else {
					std::vector<cv::Mat> channels;
					cv::split(output_8u, channels);
					alpha_8u = channels[0];
				}

				if (alpha_8u.size() != fg_bgr.size()) {
					cv::resize(alpha_8u, alpha_8u, fg_bgr.size(), 0, 0, cv::INTER_LINEAR);
				}

				cv::Mat alpha_f32, alpha_3ch;
				alpha_8u.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);
				cv::cvtColor(alpha_f32, alpha_3ch, cv::COLOR_GRAY2BGR);

				cv::Mat fg_f32, bg_f32;
				fg_bgr.convertTo(fg_f32, CV_32FC3, 1.0 / 255.0);
				bg_bgr.convertTo(bg_f32, CV_32FC3, 1.0 / 255.0);

				cv::Mat composed_f32 = alpha_3ch.mul(fg_f32) +
				                      (cv::Scalar(1.0, 1.0, 1.0) - alpha_3ch).mul(bg_f32);

				cv::Mat composed_8u;
				composed_f32.convertTo(composed_8u, CV_8UC3, 255.0);

				cv::imwrite(config_->output_bin_path + "/cpp_12_composed.jpg", composed_8u);
				logger.Info("Background composition saved: cpp_12_composed.jpg",
				            kcurrent_module_name);
			}
		}
	}

	acc_postprocess_.record(t.stop());
	return output_8u;
}

void MattingBackend::SetBackgroundModelImage(const cv::Mat& bg) {
	bg_model_u8_ = bg;
}

// ---------------------------------------------------------------------------
// ReportAccumulatedTimers — report all timing stats for this stage
// ---------------------------------------------------------------------------
void MattingBackend::ReportAccumulatedTimers(bool timing_enabled,
                                              helmsman::utils::Logger& logger,
                                              std::string_view module) const {
	acc_postprocess_.report(timing_enabled, logger, module);
	acc_composite_.report(timing_enabled, logger, module);
	acc_display_.report(timing_enabled, logger, module);

	logger.Info("", module);    // blank line for separation
	acc_total_.report(timing_enabled, logger, module);
	logger.Info("", module);    // blank line for separation
}


cv::Mat MattingBackend::Composite(const cv::Mat& frame, const cv::Mat& alpha_8u,
                                   int model_w, int model_h, int output_w, int output_h) {
	auto& logger = helmsman::utils::Logger::GetInstance();
	helmsman::utils::timing::ManualTimer t;
	t.start();

	if (alpha_8u.empty() || bg_model_u8_.empty()) {
		logger.Warning("Composite skipped: alpha or background is empty.", kcurrent_module_name);
		acc_composite_.record(t.stop());
		return frame;
	}

	// 1. Resize alpha to model resolution (CPU — RGA doesn't support single-channel YUV400)
	cv::Mat alpha_model;
	{
		alpha_model.create(model_h, model_w, CV_8UC1);
		cv::resize(alpha_8u, alpha_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
	}

	// 2. Resize frame to model resolution (RGA hardware, fallback: cv::resize)
	cv::Mat frame_model;
	{
		frame_model.create(model_h, model_w, CV_8UC3);
		ImageDescriptor src(frame.data, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(frame_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		if (!RgaResize::Instance().Execute(src, dst)) {
			fprintf(stderr, "[FATAL] RGA resize failed for frame — hardware error\n");
			std::abort();
		}
	}

	// 3. CPU alpha blend: fg_bgr * alpha + bg_bgr * (1-alpha) → composed_bgr
	cv::Mat composed_model(model_h, model_w, CV_8UC3);
	{
		const int pixels = model_h * model_w;
		const uint8_t* fg_ptr = frame_model.ptr<uint8_t>(0);
		const uint8_t* bg_ptr = bg_model_u8_.ptr<uint8_t>(0);
		const uint8_t* a_ptr = alpha_model.ptr<uint8_t>(0);
		uint8_t* out = composed_model.ptr<uint8_t>(0);
		for (int i = 0; i < pixels; ++i) {
			const uint16_t alpha = a_ptr[i];
			const uint16_t inv = 255 - alpha;
			out[0] = static_cast<uint8_t>((fg_ptr[0] * alpha + bg_ptr[0] * inv + 1 +
			                               ((fg_ptr[0] * alpha + bg_ptr[0] * inv) >> 8)) >>
			                              8);
			out[1] = static_cast<uint8_t>((fg_ptr[1] * alpha + bg_ptr[1] * inv + 1 +
			                               ((fg_ptr[1] * alpha + bg_ptr[1] * inv) >> 8)) >>
			                              8);
			out[2] = static_cast<uint8_t>((fg_ptr[2] * alpha + bg_ptr[2] * inv + 1 +
			                               ((fg_ptr[2] * alpha + bg_ptr[2] * inv) >> 8)) >>
			                              8);
			fg_ptr += 3;
			bg_ptr += 3;
			out += 3;
		}
	}

	// 4. Upscale to output resolution (RGA hardware, fallback: cv::resize)
	cv::Mat composed_output;
	{
		composed_output.create(output_h, output_w, CV_8UC3);
		ImageDescriptor src(composed_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(composed_output.data, output_w, output_h, RgaPixelFormat::kBgr888);
		if (!RgaResize::Instance().Execute(src, dst)) {
			fprintf(stderr, "[FATAL] RGA resize failed for upscale — hardware error\n");
			std::abort();
		}
	}

	acc_composite_.record(t.stop());
	return composed_output;
}
