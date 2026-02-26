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