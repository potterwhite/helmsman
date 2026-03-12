/*
 * Copyright (c) 2026 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// rknn_struct_to_string.hpp
// =============================================
// Helper functions to convert RKNN API structs to human-readable std::string
// All code is in English. No Chinese comments or identifiers.
// Perfect for logging, debugging, or printing tensors/attributes.
// =============================================

#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "rknn_api.h"

namespace arcforge {
namespace runtime {

// -----------------------------------------------------------------------------
// Forward declarations (so we can define them in any order)
std::string to_string(const rknn_tensor_attr& attr);
std::string to_string(const rknn_input_output_num& num);
std::string to_string(const rknn_input_range& range);
std::string to_string(const rknn_perf_detail& detail);
std::string to_string(const rknn_perf_run& run);
std::string to_string(const rknn_sdk_version& ver);
std::string to_string(const rknn_mem_size& mem);
std::string to_string(const rknn_custom_string& custom);
std::string to_string(const rknn_tensor_mem& mem);
std::string to_string(const rknn_input& input);
std::string to_string(const rknn_output& output);
std::string to_string(const rknn_init_extend& ext);
std::string to_string(const rknn_run_extend& ext);
std::string to_string(const rknn_output_extend& ext);

// -----------------------------------------------------------------------------
// Helper for printing arrays nicely
template <typename T>
std::string array_to_string(const T* arr, uint32_t count, const std::string& separator = ", ") {
	if (count == 0)
		return "[]";
	std::stringstream ss;
	ss << "[";
	for (uint32_t i = 0; i < count; ++i) {
		if (i > 0)
			ss << separator;
		ss << arr[i];
	}
	ss << "]";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_input_output_num
inline std::string to_string(const rknn_input_output_num& num) {
	std::stringstream ss;
	ss << "rknn_input_output_num { "
	   << "n_input = " << num.n_input << ", "
	   << "n_output = " << num.n_output << " }";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_tensor_attr  (the one you asked for specifically)
inline std::string to_string(const rknn_tensor_attr& attr) {
	std::stringstream ss;
	ss << "rknn_tensor_attr {\n";
	ss << "    index                = " << attr.index << "\n";
	ss << "    n_dims               = " << attr.n_dims << "\n";
	ss << "    dims                 = " << array_to_string(attr.dims, attr.n_dims) << "\n";
	ss << "    name                 = \"" << attr.name << "\"\n";
	ss << "    n_elems              = " << attr.n_elems << "\n";
	ss << "    size                 = " << attr.size << " bytes\n";
	ss << "    fmt                  = " << get_format_string(attr.fmt) << "\n";
	ss << "    type                 = " << get_type_string(attr.type) << "\n";
	ss << "    qnt_type             = " << get_qnt_type_string(attr.qnt_type) << "\n";
	ss << "    fl                   = " << (int)attr.fl << "\n";
	ss << "    zp                   = " << attr.zp << "\n";
	ss << "    scale                = " << std::fixed << std::setprecision(6) << attr.scale << "\n";
	ss << "    w_stride             = " << attr.w_stride << "\n";
	ss << "    size_with_stride     = " << attr.size_with_stride << " bytes\n";
	ss << "    pass_through         = " << (attr.pass_through ? "true" : "false") << "\n";
	ss << "    h_stride             = " << attr.h_stride << "\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_input_range
inline std::string to_string(const rknn_input_range& range) {
	std::stringstream ss;
	ss << "rknn_input_range {\n";
	ss << "    index         = " << range.index << "\n";
	ss << "    shape_number  = " << range.shape_number << "\n";
	ss << "    fmt           = " << get_format_string(range.fmt) << "\n";
	ss << "    name          = \"" << range.name << "\"\n";
	ss << "    n_dims        = " << range.n_dims << "\n";
	ss << "    dyn_range     = [";
	for (uint32_t i = 0; i < range.shape_number; ++i) {
		if (i > 0)
			ss << ", ";
		ss << array_to_string(range.dyn_range[i], range.n_dims);
	}
	ss << "]\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_perf_detail
inline std::string to_string(const rknn_perf_detail& detail) {
	std::stringstream ss;
	ss << "rknn_perf_detail { data_len = " << detail.data_len << " bytes, perf_data = \""
	   << (detail.perf_data ? detail.perf_data : "<null>") << "\" }";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_perf_run
inline std::string to_string(const rknn_perf_run& run) {
	std::stringstream ss;
	ss << "rknn_perf_run { run_duration = " << run.run_duration << " us }";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_sdk_version
inline std::string to_string(const rknn_sdk_version& ver) {
	std::stringstream ss;
	ss << "rknn_sdk_version {\n";
	ss << "    api_version = \"" << ver.api_version << "\"\n";
	ss << "    drv_version = \"" << ver.drv_version << "\"\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_mem_size
inline std::string to_string(const rknn_mem_size& mem) {
	std::stringstream ss;
	ss << "rknn_mem_size {\n";
	ss << "    total_weight_size       = " << mem.total_weight_size << " bytes\n";
	ss << "    total_internal_size     = " << mem.total_internal_size << " bytes\n";
	ss << "    total_dma_allocated_size= " << mem.total_dma_allocated_size << " bytes\n";
	ss << "    total_sram_size         = " << mem.total_sram_size << " bytes\n";
	ss << "    free_sram_size          = " << mem.free_sram_size << " bytes\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_custom_string
inline std::string to_string(const rknn_custom_string& custom) {
	std::stringstream ss;
	ss << "rknn_custom_string { string = \"" << custom.string << "\" }";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_tensor_mem
inline std::string to_string(const rknn_tensor_mem& mem) {
	std::stringstream ss;
	ss << "rknn_tensor_mem {\n";
	ss << "    virt_addr  = " << mem.virt_addr << "\n";
	ss << "    phys_addr  = 0x" << std::hex << mem.phys_addr << std::dec << "\n";
	ss << "    fd         = " << mem.fd << "\n";
	ss << "    offset     = " << mem.offset << "\n";
	ss << "    size       = " << mem.size << " bytes\n";
	ss << "    flags      = " << mem.flags << "\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_input
inline std::string to_string(const rknn_input& input) {
	std::stringstream ss;
	ss << "rknn_input {\n";
	ss << "    index        = " << input.index << "\n";
	ss << "    buf          = " << input.buf << "\n";
	ss << "    size         = " << input.size << " bytes\n";
	ss << "    pass_through = " << (input.pass_through ? "true" : "false") << "\n";
	ss << "    type         = " << get_type_string(input.type) << "\n";
	ss << "    fmt          = " << get_format_string(input.fmt) << "\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_output
inline std::string to_string(const rknn_output& output) {
	std::stringstream ss;
	ss << "rknn_output {\n";
	ss << "    want_float   = " << (output.want_float ? "true" : "false") << "\n";
	ss << "    is_prealloc  = " << (output.is_prealloc ? "true" : "false") << "\n";
	ss << "    index        = " << output.index << "\n";
	ss << "    buf          = " << output.buf << "\n";
	ss << "    size         = " << output.size << " bytes\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_init_extend
inline std::string to_string(const rknn_init_extend& ext) {
	std::stringstream ss;
	ss << "rknn_init_extend {\n";
	ss << "    ctx                 = " << ext.ctx << "\n";
	ss << "    real_model_offset   = " << ext.real_model_offset << "\n";
	ss << "    real_model_size     = " << ext.real_model_size << "\n";
	ss << "    model_buffer_fd     = " << ext.model_buffer_fd << "\n";
	ss << "    model_buffer_flags  = " << ext.model_buffer_flags << "\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_run_extend
inline std::string to_string(const rknn_run_extend& ext) {
	std::stringstream ss;
	ss << "rknn_run_extend {\n";
	ss << "    frame_id   = " << ext.frame_id << "\n";
	ss << "    non_block  = " << ext.non_block << "\n";
	ss << "    timeout_ms = " << ext.timeout_ms << "\n";
	ss << "    fence_fd   = " << ext.fence_fd << "\n";
	ss << "}";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_output_extend
inline std::string to_string(const rknn_output_extend& ext) {
	std::stringstream ss;
	ss << "rknn_output_extend { frame_id = " << ext.frame_id << " }";
	return ss.str();
}

// -----------------------------------------------------------------------------
// rknn_core_mask
inline std::string to_string(rknn_core_mask mask) {
	std::stringstream ss;

	ss << "rknn_core_mask {\n";
	ss << " value = " << static_cast<int>(mask) << "\n";

	ss << " cores = [";

	bool first = true;

	auto append = [&](const char* name) {
		if (!first)
			ss << ", ";
		ss << name;
		first = false;
	};

	if (mask == RKNN_NPU_CORE_AUTO) {
		append("AUTO");
	} else {
		if (mask & RKNN_NPU_CORE_0)
			append("CORE_0");

		if (mask & RKNN_NPU_CORE_1)
			append("CORE_1");

		if (mask & RKNN_NPU_CORE_2)
			append("CORE_2");
	}

	ss << "]\n";
	ss << "}";

	return ss.str();
}

// -----------------------------------------------------------------------------
// Bonus: one-liner for printing a vector of tensors (very useful)
template <typename T>
std::string to_string(const std::vector<T>& vec, const std::string& name = "vector") {
	std::stringstream ss;
	ss << name << " (" << vec.size() << " elements):\n";
	for (size_t i = 0; i < vec.size(); ++i) {
		ss << "  [" << i << "] " << to_string(vec[i]) << "\n";
	}
	return ss.str();
}

}  // namespace runtime
}  // namespace arcforge