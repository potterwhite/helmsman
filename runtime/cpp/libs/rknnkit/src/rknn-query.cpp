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

#include "RKNNKit/rknn-query.h"
#include <cstring>
#include <string>
#include "RKNNKit/utils.h"
#include "Utils/logger/logger.h"

namespace {

constexpr std::string_view kModuleName = "rknnkit";

inline auto& GetLogger() { return helmsman::utils::Logger::GetInstance(); }

}  // namespace

namespace helmsman::rknnkit {

// ============================================================================
// Query (returns data)
// ============================================================================

rknn_sdk_version RKNNQuery::SdkVersion(rknn_context ctx) {
	rknn_sdk_version sdk_ver;
	memset(&sdk_ver, 0, sizeof(sdk_ver));
	int ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
	if (ret != RKNN_SUCC) {
		GetLogger().Error("[RKNNQuery] SdkVersion query failed (ret=" + std::to_string(ret) + ")",
		                  kModuleName);
	} else {
		GetLogger().Info("[RKNNQuery] SDK api: " + std::string(sdk_ver.api_version) +
		                     "  drv: " + std::string(sdk_ver.drv_version),
		                 kModuleName);
	}
	return sdk_ver;
}

rknn_input_output_num RKNNQuery::IoNum(rknn_context ctx) {
	rknn_input_output_num io_num;
	memset(&io_num, 0, sizeof(io_num));
	int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		GetLogger().Error("[RKNNQuery] IoNum query failed (ret=" + std::to_string(ret) + ")",
		                  kModuleName);
	} else {
		GetLogger().Info("[RKNNQuery] Input num: " + std::to_string(io_num.n_input), kModuleName);
		GetLogger().Info("[RKNNQuery] Output num: " + std::to_string(io_num.n_output), kModuleName);
	}
	return io_num;
}

std::vector<rknn_tensor_attr> RKNNQuery::InputAttrs(rknn_context ctx, uint32_t n_input, bool log) {
	std::vector<rknn_tensor_attr> attrs(n_input);
	for (uint32_t i = 0; i < n_input; ++i) {
		memset(&attrs[i], 0, sizeof(rknn_tensor_attr));
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			GetLogger().Error("[RKNNQuery] InputAttrs[" + std::to_string(i) + "] failed (ret=" +
			                      std::to_string(ret) + ")",
			                  kModuleName);
		} else if (log) {
			GetLogger().Info("[RKNNQuery] Input[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		}
	}
	return attrs;
}

std::vector<rknn_tensor_attr> RKNNQuery::OutputAttrs(rknn_context ctx, uint32_t n_output, bool log) {
	std::vector<rknn_tensor_attr> attrs(n_output);
	for (uint32_t i = 0; i < n_output; ++i) {
		memset(&attrs[i], 0, sizeof(rknn_tensor_attr));
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			GetLogger().Error("[RKNNQuery] OutputAttrs[" + std::to_string(i) + "] failed (ret=" +
			                      std::to_string(ret) + ")",
			                  kModuleName);
		} else if (log) {
			GetLogger().Info("[RKNNQuery] Output[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		}
	}
	return attrs;
}

rknn_perf_run RKNNQuery::PerfRun(rknn_context ctx) {
	rknn_perf_run perf_run;
	memset(&perf_run, 0, sizeof(perf_run));
	int ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
	if (ret == RKNN_SUCC) {
		GetLogger().Info("   [RKNN] perf_run (NPU only, us): " + std::to_string(perf_run.run_duration),
		                 kModuleName);
	} else {
		GetLogger().Warning("   [RKNN] RKNN_QUERY_PERF_RUN failed", kModuleName);
	}
	return perf_run;
}

rknn_perf_detail RKNNQuery::PerfDetail(rknn_context ctx) {
	rknn_perf_detail perf_detail;
	memset(&perf_detail, 0, sizeof(perf_detail));
	int ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
	if (ret == RKNN_SUCC && perf_detail.data_len > 0) {
		std::string detail_str(perf_detail.perf_data, perf_detail.data_len);
		GetLogger().Info("   [RKNN] perf_detail:\n" + detail_str, kModuleName);
	} else {
		GetLogger().Warning("   [RKNN] RKNN_QUERY_PERF_DETAIL failed or empty", kModuleName);
	}
	return perf_detail;
}

// ============================================================================
// Diagnostic (log-only)
// ============================================================================

void RKNNQuery::LogNativeInputAttrs(rknn_context ctx, uint32_t n_input) {
	std::vector<rknn_tensor_attr> attrs(n_input);
	memset(attrs.data(), 0, n_input * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_input; ++i) {
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Native Input[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] NATIVE_INPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogNativeOutputAttrs(rknn_context ctx, uint32_t n_output) {
	std::vector<rknn_tensor_attr> attrs(n_output);
	memset(attrs.data(), 0, n_output * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_output; ++i) {
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Native Output[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] NATIVE_OUTPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogNhwcInputAttrs(rknn_context ctx, uint32_t n_input) {
	std::vector<rknn_tensor_attr> attrs(n_input);
	memset(attrs.data(), 0, n_input * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_input; ++i) {
		attrs[i].index = i;
		int ret =
		    rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] NHWC Input[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] NHWC_INPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogNhwcOutputAttrs(rknn_context ctx, uint32_t n_output) {
	std::vector<rknn_tensor_attr> attrs(n_output);
	memset(attrs.data(), 0, n_output * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_output; ++i) {
		attrs[i].index = i;
		int ret =
		    rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] NHWC Output[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] NHWC_OUTPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogMemSize(rknn_context ctx) {
	rknn_mem_size mem_size;
	memset(&mem_size, 0, sizeof(mem_size));
	int ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
	if (ret == RKNN_SUCC) {
		GetLogger().Info("[RKNNQuery] mem_size: weight=" + std::to_string(mem_size.total_weight_size) +
		                     " internal=" + std::to_string(mem_size.total_internal_size) +
		                     " dma_total=" + std::to_string(mem_size.total_dma_allocated_size) +
		                     " sram_total=" + std::to_string(mem_size.total_sram_size) +
		                     " sram_free=" + std::to_string(mem_size.free_sram_size),
		                 kModuleName);
	} else {
		GetLogger().Error("[RKNNQuery] MEM_SIZE failed", kModuleName);
	}
}

void RKNNQuery::LogCustomString(rknn_context ctx) {
	rknn_custom_string custom_str;
	memset(&custom_str, 0, sizeof(custom_str));
	int ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_str, sizeof(custom_str));
	if (ret == RKNN_SUCC) {
		GetLogger().Info("[RKNNQuery] custom_str: " + std::string(custom_str.string), kModuleName);
	} else {
		GetLogger().Error("[RKNNQuery] CUSTOM_STRING failed", kModuleName);
	}
}

void RKNNQuery::LogInputDynamicRange(rknn_context ctx, uint32_t n_input) {
	std::vector<rknn_input_range> dyn_ranges(n_input);
	memset(dyn_ranges.data(), 0, n_input * sizeof(rknn_input_range));
	for (uint32_t i = 0; i < n_input; ++i) {
		dyn_ranges[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_INPUT_DYNAMIC_RANGE, &dyn_ranges[i],
		                     sizeof(rknn_input_range));
		if (ret == RKNN_SUCC) {
			for (uint32_t s = 0; s < dyn_ranges[i].shape_number; ++s) {
				std::string dims_str;
				for (uint32_t d = 0; d < dyn_ranges[i].n_dims; ++d) {
					if (d > 0) dims_str += ", ";
					dims_str += std::to_string(dyn_ranges[i].dyn_range[s][d]);
				}
				GetLogger().Info("[RKNNQuery] InputDynamicRange[" + std::to_string(i) + "][" +
				                     std::to_string(s) + "]: {" + dims_str + "}",
				                 kModuleName);
			}
		} else {
			GetLogger().Error("[RKNNQuery] INPUT_DYNAMIC_RANGE[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogCurrentInputAttrs(rknn_context ctx, uint32_t n_input) {
	std::vector<rknn_tensor_attr> attrs(n_input);
	memset(attrs.data(), 0, n_input * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_input; ++i) {
		attrs[i].index = i;
		int ret =
		    rknn_query(ctx, RKNN_QUERY_CURRENT_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Current Input[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] CURRENT_INPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogCurrentOutputAttrs(rknn_context ctx, uint32_t n_output) {
	std::vector<rknn_tensor_attr> attrs(n_output);
	memset(attrs.data(), 0, n_output * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_output; ++i) {
		attrs[i].index = i;
		int ret =
		    rknn_query(ctx, RKNN_QUERY_CURRENT_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Current Output[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] CURRENT_OUTPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogCurrentNativeInputAttrs(rknn_context ctx, uint32_t n_input) {
	std::vector<rknn_tensor_attr> attrs(n_input);
	memset(attrs.data(), 0, n_input * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_input; ++i) {
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR, &attrs[i],
		                     sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Current Native Input[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] CURRENT_NATIVE_INPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

void RKNNQuery::LogCurrentNativeOutputAttrs(rknn_context ctx, uint32_t n_output) {
	std::vector<rknn_tensor_attr> attrs(n_output);
	memset(attrs.data(), 0, n_output * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < n_output; ++i) {
		attrs[i].index = i;
		int ret = rknn_query(ctx, RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR, &attrs[i],
		                     sizeof(rknn_tensor_attr));
		if (ret == RKNN_SUCC) {
			GetLogger().Info("[RKNNQuery] Current Native Output[" + std::to_string(i) +
			                     "] attr: " + helmsman::rknnkit::to_string(attrs[i]),
			                 kModuleName);
		} else {
			GetLogger().Error("[RKNNQuery] CURRENT_NATIVE_OUTPUT_ATTR[" + std::to_string(i) +
			                      "] failed (ret=" + std::to_string(ret) + ")",
			                  kModuleName);
		}
	}
}

}  // namespace helmsman::rknnkit
