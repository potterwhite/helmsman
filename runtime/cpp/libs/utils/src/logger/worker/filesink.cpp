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

#include "Utils/logger/worker/filesink.h"

#include "Utils/logger/logger.h"  // For LoggerLevel enum

namespace arcforge {
namespace embedded {
namespace utils {

static std::string format_timestamp_for_file(const std::chrono::system_clock::time_point& tp) {
	std::time_t time = std::chrono::system_clock::to_time_t(tp);
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;

	std::ostringstream oss;
	oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
	oss << '.' << std::setw(3) << std::setfill('0') << ms.count();
	return oss.str();
}

FileSink::FileSink(const std::string& filename) {
	// std::cout << "[FileSink Constructor] sizeof(arcforge::embedded::utils::FileSink) = "
	//           << sizeof(arcforge::embedded::utils::FileSink) << std::endl;

	file_stream_.open(filename, std::ios::app | std::ios::out);
}

FileSink::~FileSink() {
	if (file_stream_.is_open()) {
		file_stream_.close();
	}
}

void FileSink::log(const LogEntry& entry) {

	if (!file_stream_.is_open()) {
		return;
	}

	std::string level_str;
	switch (entry.level) {
		case LoggerLevel::kdebug:
			level_str = "[DEBUG]";
			break;
		default:
		case LoggerLevel::kinfo:
			level_str = "[INFO]";
			break;
		case LoggerLevel::kwarning:
			level_str = "[WARNING]";
			break;
		case LoggerLevel::kerror:
			level_str = "[ERROR]";
			break;
		case LoggerLevel::kcritical:
			level_str = "[CRITICAL]";
			break;
	}

	std::ostringstream tag_info;
	if (entry.tag.empty() == false) {
		tag_info << " [" << entry.tag << "] ";
	}

	std::lock_guard<std::mutex> lock(mutex_);  // Lock the mutex for the duration of this function

	file_stream_ << tag_info.str() << format_timestamp_for_file(entry.timestamp) << " " << level_str
	             << " " << entry.message << std::endl;
}

}  // namespace utils
}  // namespace embedded
}  // namespace arcforge