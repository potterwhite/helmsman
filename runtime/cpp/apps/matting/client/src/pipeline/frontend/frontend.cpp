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

#include "pipeline/frontend/frontend.h"

// ImageFrontend& ImageFrontend::GetInstance() {
// 	static ImageFrontend instance;

// 	return instance;
// }

ImageFrontend::ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend object constructed.");
}

ImageFrontend::~ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend cleaned up.");
}

void ImageFrontend::setOutputBinPath(const std::string& path) {
	outputBinPath_ = path;
}

TensorData ImageFrontend::preprocess(const std::string& image_path) {

	auto& logger_ = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();
	auto& math_utils_ = arcforge::utils::MathUtils::GetInstance();
	// auto& runtime_ = arcforge::runtime::RuntimeONNX::GetInstance();
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();

	TensorData tensor_data;

	cv::Mat img = cvkit_obj->loadImage(image_path);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_01_loadimage.bin");

	img = cvkit_obj->bgrToRgb(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_02_bgrToRgb.bin");

	img = cvkit_obj->ensure3Channel(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_03_ensure3Channel.bin");
	/*
	     * NOTE:
	     * DO NOT use cv::normalize / convertTo here.
	     * This must be bitwise identical to NumPy preprocessing.
	     *
	     * Date: Feb03.2026
	     * Author: PotterWhite
	     *
	     * img = cvkit_obj->normalizeToMinusOneToOne(img);
	     */
	img = cvkit_obj->normalize_exact_numpy(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_04_normalized.bin");

	// 1. resize to fit model input size
	constexpr int ref_size = 512;
	auto scale_factor = math_utils_.getScaleFactor(img.rows, img.cols, ref_size);
	std::cout << std::setprecision(17) << "x_scale_factor=" << scale_factor.x
	          << ", y_scale_factor=" << scale_factor.y << std::endl;
	cv::resize(img,             // src
	           img,             // dst（可以原地）
	           cv::Size(),      // dsize 为空
	           scale_factor.x,  // fx
	           scale_factor.y,  // fy
	           cv::INTER_AREA   // interpolation
	);
	logger_.Info("Resized Width=" + std::to_string(img.cols) +
	             ", Resized Height=" + std::to_string(img.rows));
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_05_resized.bin");

	// 2. convert to NCHW
	// std::vector<float> result = hwcToNchw(img);
	tensor_data.data = cvkit_obj->hwcToNchw(img, 3);
	//the number of 06 & 07 is according to python debug file naming
	file_utils_.dumpBinary(tensor_data.data, outputBinPath_ + "/cpp_06-07_hwcToNchw.bin");
	// NCHW
	tensor_data.shape = {1, 3, static_cast<int64_t>(img.rows), static_cast<int64_t>(img.cols)};
	// tensor_data.height = static_cast<int64_t>(img.rows);
	// tensor_data.width = static_cast<int64_t>(img.cols);

	return tensor_data;
}