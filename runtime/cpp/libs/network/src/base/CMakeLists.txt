# Copyright (c) 2025 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# base subdirectory CMakeLists.txt
#

set(COMMON_SOURCES
	"${CMAKE_CURRENT_SOURCE_DIR}/base.cpp"
	# "${CMAKE_CURRENT_SOURCE_DIR}/base-config.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/exception.cpp" )

set(BASE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/impl/base-impl.cpp" ${COMMON_SOURCES})

set(BASE_PUBLIC_HEADERS
	"${INCLUDE_DIR}/Network/base/base.h"
	# "${INCLUDE_DIR}/Network/base/base-config.h"
	"${INCLUDE_DIR}/Network/base/exception.h" )

target_sources(${PROJECT_NAME}
    PRIVATE
        ${COMMON_SOURCES}
		${BASE_SOURCES}
)
