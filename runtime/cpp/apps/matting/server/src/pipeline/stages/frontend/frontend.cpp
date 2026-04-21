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

#include "pipeline/stages/frontend.h"
#include "common/common-define.h"

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

TensorData ImageFrontend::_preprocessCore(cv::Mat img, size_t model_width, size_t model_height) {
	TensorData tensor_data;

	auto& logger_ = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();
	// auto& math_utils_ = arcforge::utils::MathUtils::GetInstance();
	// auto& runtime_ = arcforge::runtime::RuntimeONNX::GetInstance();

	// Step 1.2 - Convert BGR to RGB
	// Reason:
	//   Most deep learning frameworks expect RGB ordering.
	img = cvkit_.bgrToRgb(img);
	if (isDumpEnabled()) cvkit_.dumpBinary(img, outputBinPath_ + "/cpp_02_bgrToRgb.bin");

	// Step 1.3 - Ensure image has exactly 3 channels
	// Reason:
	//   Some images may be grayscale or RGBA.
	img = cvkit_.ensure3Channel(img);
	if (isDumpEnabled()) cvkit_.dumpBinary(img, outputBinPath_ + "/cpp_03_ensure3Channel.bin");

	// /*
	//      * NOTE:
	//      * DO NOT use cv::normalize / convertTo here.
	//      * This must be bitwise identical to NumPy preprocessing.
	//      *
	//      * Date: Feb03.2026
	//      * Author: PotterWhite
	//      *
	//      * img = cvkit_.normalizeToMinusOneToOne(img);
	//      */
	// img = cvkit_.normalize_exact_numpy(img);
	// cvkit_.dumpBinary(img, outputBinPath_ + "/cpp_04_normalized.bin");

	// =========================================================================
	// Phase 2 - Resize FIRST (cheaper on uint8), then convert to float32
	// =========================================================================
	// Critical order: resize the uint8 image to model size before type conversion.
	// Resizing CV_8UC3 (1 byte/element) is ~4× faster than resizing CV_32FC3
	// (4 bytes/element). For a 1920×1080→288×512 downscale the saving is substantial.

	// save original cols/rows before any resize
	int original_w = img.cols;
	int original_h = img.rows;

	logger_.Info(
	    "Original size: Width=" + std::to_string(img.cols) + ", Height=" + std::to_string(img.rows),
	    kcurrent_module_name);

	// Step 3.1: Compute scale factors and target size (letterbox)
	// Use the MINIMUM of the two scale factors to preserve aspect ratio.
	// Using independent scale_width / scale_height would non-uniformly stretch
	// the image (e.g. 1920×1080 → 256×256 squashes 16:9 to 1:1, corrupting inference).
	double scale_width  = static_cast<double>(model_width)  / img.cols;
	double scale_height = static_cast<double>(model_height) / img.rows;
	double scale = std::min(scale_width, scale_height);  // preserve aspect ratio

	int new_width  = static_cast<int>(img.cols * scale);
	int new_height = static_cast<int>(img.rows * scale);

	logger_.Info("ScaleOfWidth factor: " + std::to_string(scale_width) +
	                 ", ScaleOfHeight factor: " + std::to_string(scale_height) +
	                 ", New size before padding: " + std::to_string(new_width) + "x" +
	                 std::to_string(new_height),
	             kcurrent_module_name);

	// Step 3.2: Resize while still uint8 (fast path)
	cv::Mat resized_u8;
	cv::resize(img, resized_u8, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

	// Step 3.3: Letterbox padding (still uint8)
	int pad_top    = (static_cast<int>(model_height) - new_height) / 2;
	int pad_bottom = static_cast<int>(model_height) - new_height - pad_top;
	int pad_left   = (static_cast<int>(model_width)  - new_width)  / 2;
	int pad_right  = static_cast<int>(model_width)  - new_width  - pad_left;

	cv::Mat padded_u8;
	cv::copyMakeBorder(resized_u8, padded_u8, pad_top, pad_bottom, pad_left, pad_right,
	                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	logger_.Info("Final size with letterbox: Width=" + std::to_string(padded_u8.cols) +
	                 ", Height=" + std::to_string(padded_u8.rows) +
	                 ", Padding: top=" + std::to_string(pad_top) +
	                 ", bottom=" + std::to_string(pad_bottom) +
	                 ", left=" + std::to_string(pad_left) + ", right=" + std::to_string(pad_right),
	             kcurrent_module_name);

	// Step 3.4: Convert to float32 AFTER resize (only model-sized image is converted)
	// Normalize to [0.0, 1.0]: divide by 255.
	// CRITICAL: The RKNN model was compiled with std_values=[[255,255,255]], which means
	// it expects float input in [0,1]. The RKNN runtime only applies mean/std normalization
	// for RKNN_TENSOR_UINT8 inputs; for RKNN_TENSOR_FLOAT32 the data is passed through as-is.
	// Therefore we MUST normalize here in C++ before feeding to the engine.
	// padded_u8.convertTo(img, CV_32FC3, 1.0 / 255.0);
	padded_u8.convertTo(img, CV_32FC3);

	if (isDumpEnabled()) cvkit_.dumpBinary(img, outputBinPath_ + "/cpp_05_resized.bin");

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
	// tensor_data.data = cvkit_.hwcToNchw(img, 3);
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

	if (isDumpEnabled()) file_utils_.dumpBinary(tensor_data.data, outputBinPath_ + "/cpp_06-07_hwc_direct.bin");

	// =========================================================================
	// Phase 5 - Construct Tensor Shape
	// =========================================================================
	// Final Tensor Layout:
	//
	//   NHWC  →  {1, Height, Width, Channels}
	//
	// 3. NWHC
	// tensor_data.shape = {1, 3, static_cast<int64_t>(img.rows), static_cast<int64_t>(img.cols)};
	tensor_data.shape = {1, static_cast<int64_t>(img.rows), static_cast<int64_t>(img.cols), 3};

	// Save original dimensions and letterbox paddings into tensor metadata
	// (backend uses these to crop + resize alpha matte back to source resolution).
	tensor_data.orig_width = original_w;
	tensor_data.orig_height = original_h;
	tensor_data.pad_top = pad_top;
	tensor_data.pad_bottom = pad_bottom;
	tensor_data.pad_left = pad_left;
	tensor_data.pad_right = pad_right;
	// ----------------------
	return tensor_data;
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
//
// Parameters:
//   image_path: Path to input image
//   target_size: Target size for model input (default 512, dynamically set from model)
// ============================================================================
TensorData ImageFrontend::preprocess(const std::string& image_path, size_t model_width,
                                     size_t model_height) {

	// =========================================================================
	// Phase 1 - Image Loading & Color Handling
	// =========================================================================

	// Step 1.1 - Load image from disk (OpenCV default: BGR format)
	cv::Mat img = cvkit_.loadImage(image_path);
	if (isDumpEnabled()) cvkit_.dumpBinary(img, outputBinPath_ + "/cpp_01_loadimage.bin");

	return _preprocessCore(img, model_width, model_height);
}

TensorData ImageFrontend::preprocess(const cv::Mat& bgr_frame, size_t model_width,
                                     size_t model_height) {
	return _preprocessCore(bgr_frame.clone(), model_width, model_height);
}
