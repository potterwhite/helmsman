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

#include "Utils/file/file-utils.h"
#include <iostream>
#include <sstream>

namespace arcforge {
namespace utils {

FileUtils& FileUtils::GetInstance() {
	static FileUtils instance;
	return instance;
}

FileUtils::~FileUtils() {}

FileUtils::FileUtils() {}

/**
 * @brief Dump raw float data to binary file
 */
void FileUtils::dumpBinary(const std::vector<float>& vec, const std::string& outputPath) {

	std::ofstream ofs(outputPath, std::ios::binary);
	if (!ofs) {
		throw std::runtime_error("Failed to open output file: " + outputPath);
	}

	ofs.write(reinterpret_cast<const char*>(vec.data()),
	          static_cast<std::streamsize>(vec.size() * sizeof(float)));

	ofs.close();
}

}  // namespace utils
}  // namespace arcforge