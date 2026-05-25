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

#include "pipeline/stages/inference-engine/rknn/rknn-zero-copy.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include "common/common-define.h"
#include "Runtime/rknn.h/rknn.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/other/other-utils.h"

// ============================================================================
// InferenceEngineRKNNZeroCP — N-input / M-output Zero-Copy Mode
//
// Changes from the original 1-in/1-out version:
//   - input_mem_ / output_mem_ (single) → input_mems_ / output_mems_ (vector)
//   - input_attr_ / output_attr_ (single) → input_attrs_ / output_attrs_ (vector)
//   - load() queries ALL input/output attrs and allocates ALL buffers
//   - infer() iterates over all inputs/outputs
//   - MODNet (1-in/1-out) is the trivial special case of this general path
// ============================================================================

InferenceEngineRKNNZeroCP::InferenceEngineRKNNZeroCP() {
	helmsman::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNNZeroCP constructed. (Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNNZeroCP::~InferenceEngineRKNNZeroCP() {
	ReleaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

// ============================================================================
// Release all zero-copy buffers
// ============================================================================
void InferenceEngineRKNNZeroCP::ReleaseBuffers() {
	for (auto* mem : input_mems_) {
		if (mem) rknn_destroy_mem(ctx_, mem);
	}
	input_mems_.clear();

	for (auto* mem : output_mems_) {
		if (mem) rknn_destroy_mem(ctx_, mem);
	}
	output_mems_.clear();
}

// ============================================================================
// Load Model and Initialize Zero-Copy Environment
//
// Phase I   - Initialize RKNN context
// Phase II  - Query ALL input/output metadata
// Phase III - Allocate zero-copy memory for EACH input and output
// Phase IV  - Bind memory to tensors
// Phase V   - Set NPU core mask
// ============================================================================
void InferenceEngineRKNNZeroCP::Load(const std::string& model_path) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// ----------------------------------------------------------------
	// Phase I - RKNN Context Initialization (file path mode)
	// ----------------------------------------------------------------
	// perf_enabled_ is set via --perf-enabled CLI flag (SetPerfEnabled()).
	// When enabled, pass RKNN_FLAG_COLLECT_PERF_MASK to rknn_init() so that
	// per-layer NPU profiling data is available via RKNN_QUERY_PERF_DETAIL.
	// Some SDK versions reject this flag in file-path mode — retry without it.
	uint32_t init_flags = perf_enabled_ ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
	int ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
	                    init_flags, nullptr);

	if (ret < 0 && init_flags != 0) {
		logger.Warning("rknn_init with COLLECT_PERF_MASK failed (ret=" + std::to_string(ret) +
		               "), retrying without it.", kcurrent_module_name);
		init_flags = 0;
		perf_enabled_ = false;
		ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
		                init_flags, nullptr);
	}
	if (ret < 0) {
		throw std::runtime_error("rknn_init failed! (ret=" + std::to_string(ret) + ")");
	}
	logger.Info("RKNN model loaded and context initialized. (COLLECT_PERF_MASK " +
	            std::string(perf_enabled_ ? "enabled" : "disabled") + ")", kcurrent_module_name);

	// --- One-time: SDK version ---
	rknn_sdk_version sdk_ver;
	memset(&sdk_ver, 0, sizeof(sdk_ver));
	rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
	logger.Info("[RKNN] SDK api: " + std::string(sdk_ver.api_version) +
	            "  drv: " + std::string(sdk_ver.drv_version), kcurrent_module_name);

	// ----------------------------------------------------------------
	// Phase II - Query model input/output counts and attributes
	// ----------------------------------------------------------------

	// ---------------------------------
	// 2nd in docs: Query number of inputs and outputs
	ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
	if (ret != RKNN_SUCC) {
		logger.Error("[RKNN 2nd Query] Failed to query input/output number! (ret=" +
		             std::to_string(ret) + ")", kcurrent_module_name);
	} else {
		logger.Info("[RKNN 2nd Query] Input num: " + std::to_string(io_num_.n_input),
		            kcurrent_module_name);
		logger.Info("[RKNN 2nd Query] Output num: " + std::to_string(io_num_.n_output),
		            kcurrent_module_name);
	}

	// ---------------------------------
	// 3rd in docs: Query each input attribute
	// NOTE: query itself runs unconditionally - Phase III buffer allocation needs input_attrs_[i].size.
	input_attrs_.resize(io_num_.n_input);
	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
		input_attrs_[i].index = i;
		int ret_in = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i],
		                        sizeof(rknn_tensor_attr));
		if (ret_in != RKNN_SUCC) {
			logger.Error("[RKNN 3rd Query] Failed to query input attribute! (ret=" +
			             std::to_string(ret_in) + ")", kcurrent_module_name);
		} else if (perf_enabled_) {
			logger.Info("[RKNN 3rd Query] Input[" + std::to_string(i) +
			            "] attr: " + helmsman::runtime::to_string(input_attrs_[i]),
			            kcurrent_module_name);
		}
	}

	// ---------------------------------
	// 4th in docs: Query each output attribute
	// NOTE: query itself runs unconditionally - Phase III buffer allocation needs output_attrs_[i].size.
	output_attrs_.resize(io_num_.n_output);
	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
		output_attrs_[i].index = i;
		int ret_out = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i],
		                         sizeof(rknn_tensor_attr));
		if (ret_out != RKNN_SUCC) {
			logger.Error("[RKNN 4th Query] Failed to query output attribute! (ret=" +
			             std::to_string(ret_out) + ")", kcurrent_module_name);
		} else if (perf_enabled_) {
			logger.Info("[RKNN 4th Query] Output[" + std::to_string(i) +
			            "] attr: " + helmsman::runtime::to_string(output_attrs_[i]),
			            kcurrent_module_name);
		}
	}

	// ----------------------------------------------------------------
	// Phase III - Allocate zero-copy buffers for all inputs and outputs
	// ----------------------------------------------------------------
	input_mems_.resize(io_num_.n_input, nullptr);
	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		input_mems_[i] = rknn_create_mem(ctx_, static_cast<uint32_t>(input_attrs_[i].size));
		if (!input_mems_[i]) {
			throw std::runtime_error("rknn_create_mem failed for input[" + std::to_string(i) + "]");
		}
	}

	output_mems_.resize(io_num_.n_output, nullptr);
	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		output_mems_[i] = rknn_create_mem(ctx_, static_cast<uint32_t>(output_attrs_[i].size));
		if (!output_mems_[i]) {
			throw std::runtime_error("rknn_create_mem failed for output[" + std::to_string(i) + "]");
		}
	}

	// ----------------------------------------------------------------
	// Phase IV - Bind memory to tensor descriptors
	// ----------------------------------------------------------------
	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
	}
	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
	}

	// ----------------------------------------------------------------
	// Phase V - Set NPU core mask
	// ----------------------------------------------------------------
	rknn_core_mask mask = (core_mask_ < 0) ? RKNN_NPU_CORE_ALL
	                                        : static_cast<rknn_core_mask>(core_mask_);
	auto retval = rknn_set_core_mask(ctx_, mask);
	if (retval == RKNN_SUCC) {
		logger.Info("Set core to " + helmsman::runtime::to_string(mask) + " Successfully",
		            kcurrent_module_name);
	} else {
		logger.Warning("Set core to " + helmsman::runtime::to_string(mask) + " Failed",
		               kcurrent_module_name);
	}

	logger.Info("Zero-copy buffers allocated and bound: " +
	            std::to_string(io_num_.n_input) + " inputs, " +
	            std::to_string(io_num_.n_output) + " outputs.", kcurrent_module_name);

	// ----------------------------------------------------------------
	// Diagnostic queries (9th-18th) — all require Phase III buffers to be allocated first
	// ----------------------------------------------------------------

	// ---------------------------------
	// 9th in docs: query native input attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> native_input_attrs(io_num_.n_input);
		memset(native_input_attrs.data(), 0, io_num_.n_input * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_input; i++) {
			native_input_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &(native_input_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 9th Query] Native Input[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(native_input_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 9th Query] RKNN_QUERY_NATIVE_INPUT_ATTR failed for input[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 10th in docs: query native output attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> native_output_attrs(io_num_.n_output);
		memset(native_output_attrs.data(), 0, io_num_.n_output * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_output; i++) {
			native_output_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(native_output_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 10th Query] Native Output[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(native_output_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 10th Query] RKNN_QUERY_NATIVE_OUTPUT_ATTR failed for output[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 11th in docs: query nhwc input attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> local_input_attrs(io_num_.n_input);
		memset(local_input_attrs.data(), 0, io_num_.n_input * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_input; i++) {
			local_input_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &(local_input_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 11th Query] NHWC Input[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(local_input_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 11th Query] RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR failed for input[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 12th in docs: query nhwc output attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> local_output_attrs(io_num_.n_output);
		memset(local_output_attrs.data(), 0, io_num_.n_output * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_output; i++) {
			local_output_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(local_output_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 12th Query] NHWC Output[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(local_output_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 12th Query] RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR failed for output[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 7th in docs: memory info
	if (perf_enabled_) {
		rknn_mem_size mem_size;
		memset(&mem_size, 0, sizeof(mem_size));
		ret = rknn_query(ctx_, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
		if (ret == RKNN_SUCC) {
			logger.Info("[RKNN 7th Query] mem_size: weight=" +
			            std::to_string(mem_size.total_weight_size) +
			            " internal=" + std::to_string(mem_size.total_internal_size) +
			            " dma_total=" + std::to_string(mem_size.total_dma_allocated_size) +
			            " sram_total=" + std::to_string(mem_size.total_sram_size) +
			            " sram_free=" + std::to_string(mem_size.free_sram_size),
			            kcurrent_module_name);
		} else {
			logger.Error("[RKNN 7th Query] RKNN_QUERY_MEM_SIZE failed",
			             kcurrent_module_name);
		}
	}

	// ---------------------------------
	// 8th in docs: custom string
	if (perf_enabled_) {
		rknn_custom_string custom_str;
		memset(&custom_str, 0, sizeof(custom_str));
		ret = rknn_query(ctx_, RKNN_QUERY_CUSTOM_STRING, &custom_str, sizeof(custom_str));
		if (ret == RKNN_SUCC) {
			logger.Info("[RKNN 8th Query] custom_str: " + std::string(custom_str.string),
			            kcurrent_module_name);
		} else {
			logger.Error("[RKNN 8th Query] RKNN_QUERY_CUSTOM_STRING failed",
			             kcurrent_module_name);
		}
	}

	// ---------------------------------
	// 14th in docs: dynamic input shape range
	if (perf_enabled_) {
		std::vector<rknn_input_range> dyn_ranges(io_num_.n_input);
		memset(dyn_ranges.data(), 0, io_num_.n_input * sizeof(rknn_input_range));
		for (uint32_t i = 0; i < io_num_.n_input; i++) {
			dyn_ranges[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_INPUT_DYNAMIC_RANGE, &dyn_ranges[i],
			                 sizeof(rknn_input_range));
			if (ret == RKNN_SUCC) {
				for (uint32_t s = 0; s < dyn_ranges[i].shape_number; s++) {
					std::string dims_str;
					for (uint32_t d = 0; d < dyn_ranges[i].n_dims; d++) {
						if (d > 0) dims_str += ", ";
						dims_str += std::to_string(dyn_ranges[i].dyn_range[s][d]);
					}
					logger.Info("[RKNN 14th Query] Input[" + std::to_string(i) + "][" +
					            std::to_string(s) + "]: {" + dims_str + "}",
					            kcurrent_module_name);
				}
			} else {
				logger.Error("[RKNN 14th Query] RKNN_QUERY_INPUT_DYNAMIC_RANGE failed for input[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 15th in docs: current input attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> cur_input_attrs(io_num_.n_input);
		memset(cur_input_attrs.data(), 0, io_num_.n_input * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_input; i++) {
			cur_input_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_CURRENT_INPUT_ATTR, &(cur_input_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 15th Query] Current Input[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(cur_input_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 15th Query] RKNN_QUERY_CURRENT_INPUT_ATTR failed for input[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 16th in docs: current output attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> cur_output_attrs(io_num_.n_output);
		memset(cur_output_attrs.data(), 0, io_num_.n_output * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_output; i++) {
			cur_output_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_CURRENT_OUTPUT_ATTR, &(cur_output_attrs[i]),
			                 sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 16th Query] Current Output[" + std::to_string(i) +
				            "] attr: " + helmsman::runtime::to_string(cur_output_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 16th Query] RKNN_QUERY_CURRENT_OUTPUT_ATTR failed for output[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 17th in docs: current native input attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> cur_native_input_attrs(io_num_.n_input);
		memset(cur_native_input_attrs.data(), 0, io_num_.n_input * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_input; i++) {
			cur_native_input_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR,
			                 &(cur_native_input_attrs[i]), sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 17th Query] Current Native Input[" + std::to_string(i) +
				            "] attr: " +
				            helmsman::runtime::to_string(cur_native_input_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 17th Query] RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR failed for input[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}

	// ---------------------------------
	// 18th in docs: current native output attributes
	if (perf_enabled_) {
		std::vector<rknn_tensor_attr> cur_native_output_attrs(io_num_.n_output);
		memset(cur_native_output_attrs.data(), 0, io_num_.n_output * sizeof(rknn_tensor_attr));
		for (uint32_t i = 0; i < io_num_.n_output; i++) {
			cur_native_output_attrs[i].index = i;
			ret = rknn_query(ctx_, RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR,
			                 &(cur_native_output_attrs[i]), sizeof(rknn_tensor_attr));
			if (ret == RKNN_SUCC) {
				logger.Info("[RKNN 18th Query] Current Native Output[" + std::to_string(i) +
				            "] attr: " +
				            helmsman::runtime::to_string(cur_native_output_attrs[i]),
				            kcurrent_module_name);
			} else {
				logger.Error("[RKNN 18th Query] RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR failed for output[" +
				             std::to_string(i) + "], ret=" + std::to_string(ret),
				             kcurrent_module_name);
			}
		}
	}
}

// ============================================================================
// Inference — N inputs → M outputs (Zero-Copy)
//
// For each input tensor:
//   - inputs[0] (image/src): FLOAT32 0~255 → type-dependent conversion
//   - inputs[1..N-1] (recurrent states): FLOAT32 → type-dependent conversion
//
// Type conversion adapts to the model's compiled precision per tensor:
//   - INT8:  normalize [0,255]→[-1,1], quantize with scale/zp (image only)
//            For recurrent states: direct quantize (already in model range)
//   - FP16:  FLOAT32 → __fp16
//   - FP32:  direct memcpy
// ============================================================================
void InferenceEngineRKNNZeroCP::DoInfer(
    const std::vector<TensorData>& inputs,
          std::vector<TensorData>& outputs)
{
	auto& logger = helmsman::utils::Logger::GetInstance();
	auto& file_utils_ = helmsman::utils::FileUtils::GetInstance();

	if (inputs.size() != static_cast<std::size_t>(io_num_.n_input)) {
		throw std::runtime_error(
		    "Input count mismatch: model expects " + std::to_string(io_num_.n_input) +
		    " inputs, got " + std::to_string(inputs.size()));
	}

	auto t0 = std::chrono::high_resolution_clock::now();

	// ================================================================
	// INFERENCE ENGINE SCOPE — Step 1: Input Data Transfer (CPU → NPU)
	//
	// Convert float32 tensors into the model's native precision and
	// write directly into RKNN zero-copy DMA buffers.
	// This is the CPU→device data transfer; no model execution yet.
	// ================================================================
	// Step 1 - Write data into zero-copy input buffers
	// ----------------------------------------------------------------
	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		const TensorData& td    = inputs[i];
		const rknn_tensor_attr& attr = input_attrs_[i];

		bool is_int8  = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8);
		bool is_fp16  = (attr.type == RKNN_TENSOR_FLOAT16);

		if (is_int8) {
			// INT8 model: quantize input
			int8_t* dst = reinterpret_cast<int8_t*>(input_mems_[i]->virt_addr);
			float scale = attr.scale;
			float zp    = static_cast<float>(attr.zp);

			if (i == 0) {
				// Image tensor: normalize [0,255] → [-1,1] then quantize
				for (size_t j = 0; j < td.data.size(); ++j) {
					float normalized = (td.data[j] - 127.5f) / 127.5f;
					float quantized  = (normalized / scale) + zp;
					quantized = std::max(-128.0f, std::min(127.0f, quantized));
					dst[j] = static_cast<int8_t>(std::round(quantized));
				}
			} else {
				// Recurrent state: direct quantize (values already in model range)
				for (size_t j = 0; j < td.data.size(); ++j) {
					float quantized = (td.data[j] / scale) + zp;
					quantized = std::max(-128.0f, std::min(127.0f, quantized));
					dst[j] = static_cast<int8_t>(std::round(quantized));
				}
			}
		} else if (is_fp16) {
			// FP16 model: FLOAT32 → __fp16
			__fp16* dst = reinterpret_cast<__fp16*>(input_mems_[i]->virt_addr);
			for (size_t j = 0; j < td.data.size(); ++j) {
				dst[j] = static_cast<__fp16>(td.data[j]);
			}
		} else {
			// FP32 model: direct copy
			float* dst = reinterpret_cast<float*>(input_mems_[i]->virt_addr);
			std::memcpy(dst, td.data.data(), td.data.size() * sizeof(float));
		}
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	// ================================================================
	// INFERENCE ENGINE SCOPE — Step 2: NPU Execution
	//
	// rknn_run() dispatches the model graph to the NPU hardware.
	// The per-layer profiling queries (PERF_RUN, PERF_DETAIL) are
	// also engine-internal — they measure NPU kernel timing only.
	// ================================================================
	// Step 2 - Execute inference on NPU
	// ----------------------------------------------------------------
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	// --- Per-frame: RKNN pure run time (us) ---
	rknn_perf_run perf_run;
	memset(&perf_run, 0, sizeof(perf_run));
	ret = rknn_query(ctx_, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
	if (ret == 0) {
		logger.Info("   [RKNN] perf_run (NPU only, us): " +
		            std::to_string(perf_run.run_duration), kcurrent_module_name);
	} else {
		logger.Warning("   [RKNN] RKNN_QUERY_PERF_RUN failed", kcurrent_module_name);
	}

	// --- Per-frame: per-layer op timing (only when COLLECT_PERF_MASK was accepted) ---
	if (perf_enabled_) {
		rknn_perf_detail perf_detail;
		memset(&perf_detail, 0, sizeof(perf_detail));
		ret = rknn_query(ctx_, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
		if (ret == 0 && perf_detail.data_len > 0) {
			std::string detail_str(perf_detail.perf_data, perf_detail.data_len);
			logger.Info("   [RKNN] perf_detail:\n" + detail_str, kcurrent_module_name);
		} else {
			logger.Warning("   [RKNN] RKNN_QUERY_PERF_DETAIL failed or empty", kcurrent_module_name);
		}
	}

	auto t2 = std::chrono::high_resolution_clock::now();

	// ================================================================
	// INFERENCE ENGINE SCOPE — Step 3: Output Data Transfer (NPU → CPU)
	//
	// Read raw NPU output buffers and convert from model precision
	// (INT8/FP16) back to float32 for the downstream MattingBackend.
	// After this step, ownership of `outputs` transfers to the caller
	// (InferenceEngine::Infer → MattingBackend::Postprocess).
	// ================================================================
	// Step 3 - Read outputs from zero-copy buffers
	// ----------------------------------------------------------------
	outputs.clear();
	outputs.reserve(io_num_.n_output);

	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		const rknn_tensor_attr& attr = output_attrs_[i];

		bool is_int8_out = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8);
		bool is_fp16_out = (attr.type == RKNN_TENSOR_FLOAT16);

		size_t element_count;
		if (is_int8_out)       element_count = attr.size / sizeof(int8_t);
		else if (is_fp16_out)  element_count = attr.size / sizeof(__fp16);
		else                   element_count = attr.size / sizeof(float);

		std::vector<float> out_data(element_count);

		if (is_int8_out) {
			// INT8 → FLOAT32 dequantization
			int8_t* src = reinterpret_cast<int8_t*>(output_mems_[i]->virt_addr);
			float scale = attr.scale;
			float zp    = static_cast<float>(attr.zp);
			for (size_t j = 0; j < element_count; ++j) {
				out_data[j] = (static_cast<float>(src[j]) - zp) * scale;
			}
		} else if (is_fp16_out) {
			// FP16 → FLOAT32
			__fp16* src = reinterpret_cast<__fp16*>(output_mems_[i]->virt_addr);
			for (size_t j = 0; j < element_count; ++j) {
				out_data[j] = static_cast<float>(src[j]);
			}
		} else {
			// FP32 direct copy
			float* src = reinterpret_cast<float*>(output_mems_[i]->virt_addr);
			std::memcpy(out_data.data(), src, element_count * sizeof(float));
		}

		TensorData td;
		td.name = attr.name;
		td.data = std::move(out_data);

		// Assemble shape from tensor metadata
		td.shape.clear();
		for (uint32_t d = 0; d < attr.n_dims; ++d) {
			td.shape.push_back(attr.dims[d]);
		}

		// Propagate letterbox metadata from the primary input (src)
		td.orig_width  = inputs[0].orig_width;
		td.orig_height = inputs[0].orig_height;
		td.pad_top     = inputs[0].pad_top;
		td.pad_bottom  = inputs[0].pad_bottom;
		td.pad_left    = inputs[0].pad_left;
		td.pad_right   = inputs[0].pad_right;

		outputs.push_back(std::move(td));
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	// Debug dump for primary output (index 0)
	if (IsDumpEnabled() && !output_bin_path_.empty()) {
		file_utils_.dumpBinary(outputs[0].data,
		    output_bin_path_ + "cpp_08_inference-Output.bin");
	}

	auto t4 = std::chrono::high_resolution_clock::now();

	// ----------------------------------------------------------------
	// INFERENCE ENGINE SCOPE — Profiling timestamps (engine-internal)
	// ----------------------------------------------------------------
	std::chrono::duration<double, std::milli> cast_in_time  = t1 - t0;
	std::chrono::duration<double, std::milli> npu_run_time  = t2 - t1;
	std::chrono::duration<double, std::milli> cast_out_time = t3 - t2;
	std::chrono::duration<double, std::milli> dump_time     = t4 - t3;

	bool is_int8_primary = (input_attrs_[0].type == RKNN_TENSOR_INT8 ||
	                        input_attrs_[0].type == RKNN_TENSOR_UINT8);
	bool is_fp16_primary = (input_attrs_[0].type == RKNN_TENSOR_FLOAT16);
	std::string precision_str = is_int8_primary ? "INT8" : (is_fp16_primary ? "FP16" : "FP32");

	logger.Info("   [Profiler] Input conversion (" + precision_str +
	                ") cost: " + std::to_string(cast_in_time.count()) + " ms.",
	            kcurrent_module_name);
	logger.Info(
	    "   [Profiler] Pure RKNN Run cost:   " + std::to_string(npu_run_time.count()) + " ms.",
	    kcurrent_module_name);
	logger.Info(
	    "   [Profiler] Output conversion cost: " + std::to_string(cast_out_time.count()) + " ms.",
	    kcurrent_module_name);
	logger.Info("   [Profiler] Binary Dump cost:     " + std::to_string(dump_time.count()) + " ms.",
	            kcurrent_module_name);

	logger.Info("infer() complete: " + std::to_string(io_num_.n_input) + " in / " +
	            std::to_string(io_num_.n_output) + " out", kcurrent_module_name);
}
