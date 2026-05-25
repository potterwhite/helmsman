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

#include <string>
#include <vector>
#include "rknn_api.h"

namespace helmsman::rknnkit {

// ============================================================================
// RKNNQuery — stateless utility class for RKNN SDK query operations.
//
// Method names carry the doc ordinal (1st, 2nd, …) so that callers can
// immediately see which batch a query belongs to.  Log messages use the
// matching "[RKNN Nth]" prefix, matching the original inline style.
//
// All methods are static and take rknn_context as an explicit parameter.
// Instances cannot be created (deleted constructor, private destructor).
// Thread safety: methods are stateless; ctx thread-safety is caller's duty.
//
// Query groups:
//   1st–4th   : Load() Phase I/II — essential model metadata
//   5th–6th   : DoInfer() — per-frame NPU performance
//   7th–8th   : Load() post-bind — model memory & identity
//   9th–18th  : Load() post-bind — tensor layout diagnostics
// ============================================================================
class RKNNQuery {
public:
	RKNNQuery() = delete;
	RKNNQuery(const RKNNQuery&) = delete;
	RKNNQuery& operator=(const RKNNQuery&) = delete;

	// ===== Group 1: Load() Phase I/II — essential model metadata =====

	static rknn_sdk_version SdkVersion1st(rknn_context ctx);
	static rknn_input_output_num IoNum2nd(rknn_context ctx);
	static std::vector<rknn_tensor_attr> InputAttrs3rd(rknn_context ctx, uint32_t n_input,
	                                                    bool log = true);
	static std::vector<rknn_tensor_attr> OutputAttrs4th(rknn_context ctx, uint32_t n_output,
	                                                     bool log = true);

	// ===== Group 2: DoInfer() — per-frame NPU performance =====

	static rknn_perf_run PerfRun5th(rknn_context ctx);
	static rknn_perf_detail PerfDetail6th(rknn_context ctx);

	// ===== Group 3: Load() post-bind — model memory & identity =====

	static void LogMemSize7th(rknn_context ctx);
	static void LogCustomString8th(rknn_context ctx);

	// ===== Group 4: Load() post-bind — tensor layout diagnostics =====

	static void LogNativeInputAttrs9th(rknn_context ctx, uint32_t n_input);
	static void LogNativeOutputAttrs10th(rknn_context ctx, uint32_t n_output);
	static void LogNhwcInputAttrs11th(rknn_context ctx, uint32_t n_input);
	static void LogNhwcOutputAttrs12th(rknn_context ctx, uint32_t n_output);
	static void LogInputDynamicRange14th(rknn_context ctx, uint32_t n_input);
	static void LogCurrentInputAttrs15th(rknn_context ctx, uint32_t n_input);
	static void LogCurrentOutputAttrs16th(rknn_context ctx, uint32_t n_output);
	static void LogCurrentNativeInputAttrs17th(rknn_context ctx, uint32_t n_input);
	static void LogCurrentNativeOutputAttrs18th(rknn_context ctx, uint32_t n_output);

private:
	~RKNNQuery() = delete;
};

}  // namespace helmsman::rknnkit
