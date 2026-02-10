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

#include "Utils/common/system-info.h"

// IMPORTANT: Import from CMakeLists.txt externally
#include "system-info.h"

namespace arcforge {
namespace embedded {
namespace utils {


std::string get_utils_library_version() {
	// These macros come from CMakeLists externally
	return VERSION_STRING;
}

std::string get_utils_library_author() {
	// These macros come from CMakeLists externally
	return AUTHOR_NAME;
}

std::string get_utils_library_author_email() {
	// These macros come from CMakeLists externally
	return AUTHOR_EMAIL;
}

std::string get_utils_library_build_timestamp() {
	// These macros come from CMakeLists externally
	return BUILD_TIMESTAMP;
}

}  // namespace utils
}  // namespace embedded
}  // namespace arcforge