// Copyright (c) 2025 PotterWhite
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

// libs/network/src/base/impl/base-impl.cpp
#include "CVKit/base/impl/base-impl.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace cvkit {

// #define DEBUG
/*===================================================
 * constructors and operators
 *===================================================*/
BaseImpl::BaseImpl() {
	arcforge::embedded::utils::Logger::GetInstance().Info("BaseImpl object constructed.",
	                                                      kcurrent_lib_name);
}

BaseImpl::~BaseImpl() {
	arcforge::embedded::utils::Logger::GetInstance().Info("BaseImpl cleaned up.",
	                                                      kcurrent_lib_name);
}

/**
 * @brief Load image from disk using OpenCV
 */
cv::Mat BaseImpl::loadImage(const std::string& imagePath) {
	cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
	if (img.empty()) {
		throw std::runtime_error("Failed to load image: " + imagePath);
	}
	return img;
}

cv::Mat BaseImpl::bgrToRgb(const cv::Mat& input) {
	cv::Mat out;
	cv::cvtColor(input, out, cv::COLOR_BGR2RGB);
	return out;
}

cv::Mat BaseImpl::ensure3Channel(const cv::Mat& input) {
	cv::Mat out;

	if (input.channels() == 1) {
		cv::cvtColor(input, out, cv::COLOR_GRAY2RGB);
	} else if (input.channels() == 4) {
		// 等价于 Python 的 im[:, :, 0:3]
		cv::cvtColor(input, out, cv::COLOR_BGRA2BGR);
	} else {
		out = input;
	}

	return out;
}

void BaseImpl::echoImg(cv::Mat& input) {
	float* p = input.ptr<float>(0);
	std::cout << std::fixed << std::setprecision(8);
	for (int i = 0; i < 10; ++i) {
		std::cout << "v[" << i << "] = " << p[i] << std::endl;
	}

	unsigned char* bytes = reinterpret_cast<unsigned char*>(p);
	for (int i = 0; i < 10 * 4; ++i) {
		printf("%02x ", bytes[i]);
	}
	printf("\n");
}

/**
 * @brief Normalize image to range [-1, 1]
 * Formula: (pixel - 127.5) / 127.5
 */
cv::Mat BaseImpl::normalizeToMinusOneToOne(const cv::Mat& input) {
	cv::Mat floatImg;
	/*
     * Method A: Wrong
     */
	input.convertTo(floatImg, CV_32FC3, 1.0 / 127.5, -1.0);

	// /*
	//  * Method B: the same as python
	//  */
	// input.convertTo(floatImg, CV_32FC3);   // uint8 → float32

	// floatImg = floatImg - 127.5f;          // 第一次 rounding
	// floatImg = floatImg / 127.5f;          // 第二次 rounding

	// if (!floatImg.isContinuous()) {
	//     std::cout << "isContinuous: " << floatImg.isContinuous() << std::endl;
	//     floatImg = floatImg.clone();
	// }

	echoImg(floatImg);

	return floatImg;
}

cv::Mat BaseImpl::normalize_exact_numpy(const cv::Mat& input) {
	CV_Assert(input.type() == CV_8UC3);

	cv::Mat out(input.rows, input.cols, CV_32FC3);

	const float scale = static_cast<float>(1.0 / 127.5);  // 注意：double → float
	const float bias = -1.0f;

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b* src = input.ptr<cv::Vec3b>(y);
		cv::Vec3f* dst = out.ptr<cv::Vec3f>(y);

		for (int x = 0; x < input.cols; ++x) {
			dst[x][0] = src[x][0] * scale + bias;
			dst[x][1] = src[x][1] * scale + bias;
			dst[x][2] = src[x][2] * scale + bias;
		}
	}

	echoImg(out);

	return out;
}

/**
 * @brief Dump raw float data to binary file
 */
void BaseImpl::dumpBinary(const cv::Mat& mat, const std::string& outputPath) {
	if (!mat.isContinuous()) {
		throw std::runtime_error("Mat is not continuous in memory");
	}

	std::ofstream ofs(outputPath, std::ios::binary);
	if (!ofs) {
		throw std::runtime_error("Failed to open output file: " + outputPath);
	}

	ofs.write(reinterpret_cast<const char*>(mat.ptr<float>(0)),
	          static_cast<std::streamsize>(mat.total() * mat.elemSize()));

	ofs.close();
}

}  // namespace cvkit
}  // namespace arcforge