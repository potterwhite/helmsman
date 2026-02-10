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

#pragma once

#include <opencv2/opencv.hpp>
#include "CVKit/common/common-types.h"
#include "CVKit/pch.h"

namespace arcforge {
namespace cvkit {

// forward declaration of PIMPL implementation class
class BaseImpl;

static const std::string k_empty_socket_path_sentinel = "";

class Base {
   public:
	explicit Base();
	virtual ~Base();

	// copy constructor and operator
	Base(const Base&) = delete;
	Base& operator=(const Base&) = delete;

	// std::move constructor and operator
	Base(Base&&) noexcept;
	Base& operator=(Base&&) noexcept;

	cv::Mat loadImage(const std::string& imagePath);
	cv::Mat bgrToRgb(const cv::Mat& input);
	cv::Mat ensure3Channel(const cv::Mat& input);
	void echoImg(cv::Mat& input);
	cv::Mat normalizeToMinusOneToOne(const cv::Mat& input);
	cv::Mat normalize_exact_numpy(const cv::Mat& input);
	void dumpBinary(const cv::Mat& mat, const std::string& outputPath);
	std::vector<float> hwcToNchw(const cv::Mat& origin_img, const size_t channels);

   private:
	explicit Base(std::unique_ptr<BaseImpl>);
	std::unique_ptr<BaseImpl> impl_;
};

}  // namespace cvkit
}  // namespace arcforge