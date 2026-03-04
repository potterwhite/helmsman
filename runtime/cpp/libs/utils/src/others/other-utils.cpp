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

#include "Utils/other/other-utils.h"
#include <algorithm>  // std::min
#include <sstream>
#include <string>
#include <vector>

namespace arcforge {
namespace utils {

OtherUtils& OtherUtils::GetInstance() {
	static OtherUtils instance;
	return instance;
}

OtherUtils::~OtherUtils() {}

OtherUtils::OtherUtils() {}

/* =============================================================================
 * Return a string representation of the first N elements, suitable for logging
 *
 * Usage:
 *
 * logger.Info("input data = " + format_vector_preview(input.data, 10, 6));
 * logger.Info("weights    = " + format_vector_preview(weights, 6));
 */

std::string OtherUtils::format_vector_preview(const std::vector<float>& vec, size_t max_count,
                                              int precision) {
	if (vec.empty()) {
		return "[empty]";
	}

	std::ostringstream oss;
	oss.precision(precision);
	oss << std::fixed;

	size_t show_cnt = std::min(vec.size(), max_count);

	oss << "[size=" << vec.size() << "] ";
	oss << "precision=" << precision;
	oss << ":\n";
	oss << "[0] " << vec[0];

	for (size_t i = 1; i < show_cnt; ++i) {
		if (i % 4 == 0) {
			oss << "\n";
		} else {
			oss << ", ";
		}
		oss << "[" << std::to_string(i) << "] " << vec[i];
	}

	if (vec.size() > max_count) {
		oss << ", ... (total " << vec.size() << ")";
	}

	return oss.str();
}

}  // namespace utils
}  // namespace arcforge