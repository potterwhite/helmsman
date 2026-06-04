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

# third_party/CMakeLists.txt
#
# Dynamically include dependencies based on INFERENCE_BACKEND and CMAKE_PLATFORM.

# Default INFERENCE_BACKEND if not set
if(NOT DEFINED INFERENCE_BACKEND OR INFERENCE_BACKEND STREQUAL "")
    if("rockchip" IN_LIST CMAKE_PLATFORM)
        set(INFERENCE_BACKEND "rknn-zerocopy")
    else()
        set(INFERENCE_BACKEND "onnx")
    endif()
    message(STATUS "[ThirdParty] Defaulting INFERENCE_BACKEND to '${INFERENCE_BACKEND}'")
endif()

message(STATUS "[ThirdParty] INFERENCE_BACKEND = ${INFERENCE_BACKEND}")

# ONNX Runtime: always needed (as fallback or primary)
add_subdirectory(onnxruntime)

# RKNN: only on rockchip platforms AND when RKNN backend is requested
if(INFERENCE_BACKEND MATCHES "^rknn")
    if("rockchip" IN_LIST CMAKE_PLATFORM)
        add_subdirectory(librknnrt)
    else()
        message(FATAL_ERROR "RKNN backend requires rockchip platform (CMAKE_PLATFORM contains 'rockchip')")
    endif()
endif()

# ---------------------------------
#               End
# ---------------------------------
