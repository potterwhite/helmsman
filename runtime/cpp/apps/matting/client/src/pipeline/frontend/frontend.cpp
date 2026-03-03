// ============================================================================
// License Section
// ============================================================================
// Copyright (c) 2026 PotterWhite
//
// This file is released under the MIT License.
// You are free to use, modify, distribute, and sublicense this software,
// provided that the copyright notice and permission notice are included.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
// The author is not responsible for any damage or liability.
//
// ============================================================================

#include "pipeline/frontend/frontend.h"
#include "common-define.h"

// ============================================================================
// ImageFrontend Class
// Responsibility:
//   This class performs image preprocessing before inference.
//
// High-Level Pipeline:
//
//   Step 1  - Load image from disk
//   Step 2  - Convert color space (BGR → RGB)
//   Step 3  - Ensure 3-channel format
//   Step 4  - Convert data type to float32 (no normalization)
//   Step 5  - Resize image to model reference size
//   Step 6  - Convert image memory layout for inference engine
//   Step 7  - Fill TensorData structure (data + shape)
//
// This preprocessing must match Python/NumPy logic exactly
// to guarantee inference consistency.
// ============================================================================

// ImageFrontend& ImageFrontend::GetInstance() {
// 	static ImageFrontend instance;
// 	return instance;
// }

// ============================================================================
// Constructor
// Purpose:
//   Construct frontend processing object
// ============================================================================
ImageFrontend::ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend object constructed.",
	                                                      kcurrent_module_name);
}

// ============================================================================
// Destructor
// Purpose:
//   Cleanup logging only (no dynamic resources owned here)
// ============================================================================
ImageFrontend::~ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend cleaned up.",
	                                                      kcurrent_module_name);
}

// ============================================================================
// Configure output binary dump directory
// ============================================================================
void ImageFrontend::setOutputBinPath(const std::string& path) {
	outputBinPath_ = path;
}

// ============================================================================
// Preprocess Function
//
// Complete Image → Tensor Pipeline
//
// Major Phases:
//
//   Phase 1 - Image Loading & Basic Format Conversion
//   Phase 2 - Data Type Conversion
//   Phase 3 - Resize to Model Reference Size
//   Phase 4 - Memory Layout Preparation
//   Phase 5 - Construct TensorData (data + shape)
//
// This function guarantees:
//   • Bit-level debug capability via dumpBinary()
//   • Controlled numeric range (0~255 float32)
//   • Layout compatibility with RKNN inference
// ============================================================================
TensorData ImageFrontend::preprocess(const std::string& image_path) {

	auto& logger_ = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();
	// auto& math_utils_ = arcforge::utils::MathUtils::GetInstance();
	// auto& runtime_ = arcforge::runtime::RuntimeONNX::GetInstance();
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();

	TensorData tensor_data;

	// =========================================================================
	// Phase 1 - Image Loading & Color Handling
	// =========================================================================

	// Step 1.1 - Load image from disk (OpenCV default: BGR format)
	cv::Mat img = cvkit_obj->loadImage(image_path);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_01_loadimage.bin");

	// Step 1.2 - Convert BGR to RGB
	// Reason:
	//   Most deep learning frameworks expect RGB ordering.
	img = cvkit_obj->bgrToRgb(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_02_bgrToRgb.bin");

	// Step 1.3 - Ensure image has exactly 3 channels
	// Reason:
	//   Some images may be grayscale or RGBA.
	img = cvkit_obj->ensure3Channel(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_03_ensure3Channel.bin");

	// /*
	//      * NOTE:
	//      * DO NOT use cv::normalize / convertTo here.
	//      * This must be bitwise identical to NumPy preprocessing.
	//      *
	//      * Date: Feb03.2026
	//      * Author: PotterWhite
	//      *
	//      * img = cvkit_obj->normalizeToMinusOneToOne(img);
	//      */
	// img = cvkit_obj->normalize_exact_numpy(img);
	// cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_04_normalized.bin");

	// =========================================================================
	// Phase 2 - Convert Data Type (uint8 → float32)
	// =========================================================================
	// Important Design Decision:
	//   We DO NOT normalize to [-1, 1] or [0, 1].
	//   We ONLY convert type:
	//
	//   uint8  (0~255)  →  float32 (0.0~255.0)
	//
	// Reason:
	//   RKNN driver will internally handle quantization or FP conversion.
	//
	// 改为：仅仅把类型从 uint8 转成 float32，保留 0.0 ~ 255.0 的数值范围，喂给 RKNN 驱动
	img.convertTo(img, CV_32FC3);

	// 依然可以 dump 出来确认，里面的值应该是 0~255 的浮点数
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_04_converted_float.bin");

	// =========================================================================
	// Phase 3 - Resize to Model Reference Size
	// =========================================================================
	// CRITICAL OPTIMIZATION:
	//   Use fixed 512x512 size instead of aspect-ratio-preserving resize.
	//
	// Reason:
	//   Non-standard sizes like 512x896 cause NPU multi-core scheduling failure,
	//   forcing layers to run on single core instead of 3-core parallel mode.
	//   This results in 3x+ performance degradation (760ms -> ~200ms expected).
	//
	// Trade-off:
	//   Slight aspect ratio distortion vs. massive performance gain.
	//
	constexpr int target_width = 512;
	constexpr int target_height = 512;

	logger_.Info("Original size: Width=" + std::to_string(img.cols) +
	                 ", Height=" + std::to_string(img.rows),
	             kcurrent_module_name);

	cv::resize(img,
	           img,
	           cv::Size(target_width, target_height),
	           0,
	           0,
	           cv::INTER_AREA);

	logger_.Info("Resized to fixed size: Width=" + std::to_string(img.cols) +
	                 ", Height=" + std::to_string(img.rows),
	             kcurrent_module_name);

	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_05_resized.bin");

	// =========================================================================
	// Phase 4 - Prepare Memory Layout for Inference Engine
	// =========================================================================
	// Originally:
	//   Convert HWC → NCHW manually.
	//
	// Now:
	//   Directly copy HWC continuous memory to inference engine.
	//
	// Reason:
	//   Current RKNN model expects NHWC format.
	//
	// ---------------------------------
	// 2. convert to NCHW
	// tensor_data.data = cvkit_obj->hwcToNchw(img, 3);
	// file_utils_.dumpBinary(tensor_data.data, outputBinPath_ + "/cpp_06-07_hwcToNchw.bin");

	// 改为：直接拷贝 HWC 格式的连续内存给 inference 引擎

	cv::Mat continuous_img;
	if (img.isContinuous()) {
		continuous_img = img;
	} else {
		continuous_img = img.clone();
	}

	CV_Assert(continuous_img.isContinuous());

	size_t total = continuous_img.total() * static_cast<size_t>(continuous_img.channels());
	float* ptr = continuous_img.ptr<float>(0);

	tensor_data.data.assign(ptr, ptr + total);

	file_utils_.dumpBinary(tensor_data.data, outputBinPath_ + "/cpp_06-07_hwc_direct.bin");

	// =========================================================================
	// Phase 5 - Construct Tensor Shape
	// =========================================================================
	// Final Tensor Layout:
	//
	//   NHWC  →  {1, Height, Width, Channels}
	//
	// 3. NWHC
	// tensor_data.shape = {1, 3, static_cast<int64_t>(img.rows), static_cast<int64_t>(img.cols)};
	tensor_data.shape = {1,
	                     static_cast<int64_t>(img.rows),
	                     static_cast<int64_t>(img.cols),
	                     3};

	return tensor_data;
}
