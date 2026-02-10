/*
 * Copyright (c) 2025 PotterWhite
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

// include/CVKit/base/impl/base-impl.h
#pragma once

#include <opencv2/opencv.hpp>
#include "CVKit/common/common-types.h"
#include "CVKit/pch.h"

namespace arcforge {
namespace cvkit {

inline constexpr int killegal_fd_value = -1;

class BaseImpl;

class BaseImpl {
   public:
	BaseImpl();
	~BaseImpl();

	// forbitd copy and assignment
	BaseImpl(const BaseImpl&) = delete;
	BaseImpl& operator=(const BaseImpl&) = delete;

	// permit move semantics
	BaseImpl(BaseImpl&&) noexcept = default;
	BaseImpl& operator=(BaseImpl&&) noexcept = default;

	cv::Mat loadImage(const std::string& imagePath);
	cv::Mat bgrToRgb(const cv::Mat& input);
	cv::Mat ensure3Channel(const cv::Mat& input);
	void echoImg(cv::Mat& input);
	cv::Mat normalizeToMinusOneToOne(const cv::Mat& input);
	cv::Mat normalize_exact_numpy(const cv::Mat& input);
	void dumpBinary(const cv::Mat& mat, const std::string& outputPath);
	std::vector<float> hwcToNchw(const cv::Mat& origin_img, const size_t channels);

   private:
	// member functions

   private:
	// member variables
	int socketfd_ = -1;
	std::string socketpath_;
	std::unique_ptr<std::mutex> socket_mutex_;
	std::unique_ptr<std::mutex> log_mutex_;
};

}  // namespace cvkit
}  // namespace arcforge