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

#pragma once

#include <vector>
#include "rknn_api.h"

namespace helmsman::rknnkit {

// ============================================================================
// RKNNQuery — stateless utility class for RKNN SDK query operations.
//
// All methods are static and take rknn_context as an explicit parameter.
// Instances cannot be created (deleted constructor, private destructor).
// Thread safety: methods are stateless; ctx thread-safety is caller's duty.
// ============================================================================
class RKNNQuery {
public:
	RKNNQuery() = delete;
	RKNNQuery(const RKNNQuery&) = delete;
	RKNNQuery& operator=(const RKNNQuery&) = delete;

	// ===== Query (returns data) =====

	static rknn_sdk_version SdkVersion(rknn_context ctx);
	static rknn_input_output_num IoNum(rknn_context ctx);
	static std::vector<rknn_tensor_attr> InputAttrs(rknn_context ctx, uint32_t n_input,
	                                                 bool log = true);
	static std::vector<rknn_tensor_attr> OutputAttrs(rknn_context ctx, uint32_t n_output,
	                                                  bool log = true);
	static rknn_perf_run PerfRun(rknn_context ctx);
	static rknn_perf_detail PerfDetail(rknn_context ctx);

	// ===== Diagnostic (log-only, no return value) =====

	static void LogNativeInputAttrs(rknn_context ctx, uint32_t n_input);
	static void LogNativeOutputAttrs(rknn_context ctx, uint32_t n_output);
	static void LogNhwcInputAttrs(rknn_context ctx, uint32_t n_input);
	static void LogNhwcOutputAttrs(rknn_context ctx, uint32_t n_output);
	static void LogMemSize(rknn_context ctx);
	static void LogCustomString(rknn_context ctx);
	static void LogInputDynamicRange(rknn_context ctx, uint32_t n_input);
	static void LogCurrentInputAttrs(rknn_context ctx, uint32_t n_input);
	static void LogCurrentOutputAttrs(rknn_context ctx, uint32_t n_output);
	static void LogCurrentNativeInputAttrs(rknn_context ctx, uint32_t n_input);
	static void LogCurrentNativeOutputAttrs(rknn_context ctx, uint32_t n_output);

private:
	~RKNNQuery() = delete;
};

}  // namespace helmsman::rknnkit
