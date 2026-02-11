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

#include "Utils/math/math-utils.h"
#include <iostream>
#include <sstream>

namespace arcforge {
namespace utils {

MathUtils& MathUtils::GetInstance() {
	static MathUtils instance;
	return instance;
}

MathUtils::~MathUtils() {}

MathUtils::MathUtils() {}

ScaleFactor MathUtils::getScaleFactor(int im_h, int im_w, int ref_size) {
	int im_rh;
	int im_rw;

	if (std::max(im_h, im_w) < ref_size || std::min(im_h, im_w) > ref_size) {
		if (im_w >= im_h) {
			im_rh = ref_size;
			im_rw = static_cast<int>(static_cast<double>(im_w) / im_h * ref_size);
		} else {
			im_rw = ref_size;
			im_rh = static_cast<int>(static_cast<double>(im_h) / im_w * ref_size);
		}
	} else {
		im_rh = im_h;
		im_rw = im_w;
	}

	// 对齐到 32 倍数
	im_rw = im_rw - (im_rw % 32);
	im_rh = im_rh - (im_rh % 32);

	double x_scale_factor = static_cast<double>(im_rw) / im_w;

	double y_scale_factor = static_cast<double>(im_rh) / im_h;

	return {x_scale_factor, y_scale_factor};
}

}  // namespace utils
}  // namespace arcforge