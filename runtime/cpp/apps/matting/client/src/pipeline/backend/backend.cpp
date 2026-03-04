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

	const size_t total = HW * static_cast<size_t>(C);

	file_utils.dumpBinary(std::vector<float>((float*)clamped.data, (float*)clamped.data + total),
	                      output_path_ + "/cpp_10_clamped.bin");

	// -------------------------
	// 3. convert to 8bit image
	cv::Mat output_8u;
	clamped.convertTo(output_8u, (C_int == 1 ? CV_8UC1 : CV_8UC3), 255.0);

	cv::imwrite(output_path_ + "/cpp_11_result.png", output_8u);

	logger.Info("Backend postprocess finished.", kcurrent_module_name);

	return output_8u;
}
