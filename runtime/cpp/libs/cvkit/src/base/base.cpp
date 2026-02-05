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

// cvkit/src/base/base.cpp

#include "CVKit/base/base.h"
#include "impl/base-impl.h"

namespace arcforge {
namespace cvkit {

Base::Base() : impl_(std::make_unique<BaseImpl>()) {}

Base::Base(std::unique_ptr<BaseImpl> param_impl) : impl_(std::move(param_impl)) {}

Base::~Base() {}

Base::Base(Base&& other) noexcept = default;

Base& Base::operator=(Base&& other) noexcept = default;

cv::Mat Base::loadImage(const std::string& imagePath) {
	if (impl_ != nullptr) {
		return impl_->loadImage(imagePath);
	}

	return cv::Mat{};
}

cv::Mat Base::bgrToRgb(const cv::Mat& input) {
	if (impl_ != nullptr) {
		return impl_->bgrToRgb(input);
	}

	return cv::Mat{};
}

cv::Mat Base::ensure3Channel(const cv::Mat& input) {
	if (impl_ != nullptr) {
		return impl_->ensure3Channel(input);
	}

	return cv::Mat{};
}

void Base::echoImg(cv::Mat& input) {
	if (impl_ != nullptr) {
		impl_->echoImg(input);
	}
}

cv::Mat Base::normalizeToMinusOneToOne(const cv::Mat& input) {
	if (impl_ != nullptr) {
		return impl_->normalizeToMinusOneToOne(input);
	}

	return cv::Mat{};
}

cv::Mat Base::normalize_exact_numpy(const cv::Mat& input) {
	if (impl_ != nullptr) {
		return impl_->normalize_exact_numpy(input);
	}

	return cv::Mat{};
}

void Base::dumpBinary(const cv::Mat& mat, const std::string& outputPath) {
	if (impl_ != nullptr) {
		impl_->dumpBinary(mat, outputPath);
	}
}

}  // namespace cvkit
}  // namespace arcforge