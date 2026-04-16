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

#include "pipeline/inference-engine/rknn/rknn-zero-copy.h"
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
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNNZeroCP constructed. (Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNNZeroCP::~InferenceEngineRKNNZeroCP() {
	releaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

// ============================================================================
// Release all zero-copy buffers
// ============================================================================
void InferenceEngineRKNNZeroCP::releaseBuffers() {
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
void InferenceEngineRKNNZeroCP::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// ----------------------------------------------------------------
	// Phase I - RKNN Context Initialization (file path mode)
	// ----------------------------------------------------------------
	int ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
	                    0, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_init failed!");
	}
	logger.Info("RKNN model loaded and context initialized.", kcurrent_module_name);

	// ----------------------------------------------------------------
	// Phase II - Query model input/output counts and attributes
	// ----------------------------------------------------------------
	rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

	const uint32_t n_in  = io_num_.n_input;
	const uint32_t n_out = io_num_.n_output;

	logger.Info("Input num: "  + std::to_string(n_in),  kcurrent_module_name);
	logger.Info("Output num: " + std::to_string(n_out), kcurrent_module_name);

	// Query each input attribute
	input_attrs_.resize(n_in);
	for (uint32_t i = 0; i < n_in; ++i) {
		memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
		input_attrs_[i].index = i;
		rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
		logger.Info("Input[" + std::to_string(i) + "] attr: " +
		            arcforge::runtime::to_string(input_attrs_[i]));
	}

	// Query each output attribute
	output_attrs_.resize(n_out);
	for (uint32_t i = 0; i < n_out; ++i) {
		memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
		output_attrs_[i].index = i;
		rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
		logger.Info("Output[" + std::to_string(i) + "] attr: " +
		            arcforge::runtime::to_string(output_attrs_[i]));
	}

	// ----------------------------------------------------------------
	// Phase III - Allocate zero-copy buffers for all inputs and outputs
	// ----------------------------------------------------------------
	input_mems_.resize(n_in, nullptr);
	for (uint32_t i = 0; i < n_in; ++i) {
		input_mems_[i] = rknn_create_mem(ctx_, static_cast<uint32_t>(input_attrs_[i].size));
		if (!input_mems_[i]) {
			throw std::runtime_error("rknn_create_mem failed for input[" + std::to_string(i) + "]");
		}
	}

	output_mems_.resize(n_out, nullptr);
	for (uint32_t i = 0; i < n_out; ++i) {
		output_mems_[i] = rknn_create_mem(ctx_, static_cast<uint32_t>(output_attrs_[i].size));
		if (!output_mems_[i]) {
			throw std::runtime_error("rknn_create_mem failed for output[" + std::to_string(i) + "]");
		}
	}

	// ----------------------------------------------------------------
	// Phase IV - Bind memory to tensor descriptors
	// ----------------------------------------------------------------
	for (uint32_t i = 0; i < n_in; ++i) {
		rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
	}
	for (uint32_t i = 0; i < n_out; ++i) {
		rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
	}

	// ----------------------------------------------------------------
	// Phase V - Set NPU core mask
	// ----------------------------------------------------------------
	rknn_core_mask mask = RKNN_NPU_CORE_ALL;
	auto retval = rknn_set_core_mask(ctx_, mask);
	if (retval == RKNN_SUCC) {
		logger.Info("Set core to " + arcforge::runtime::to_string(mask) + " Successfully");
	} else {
		logger.Warning("Set core to " + arcforge::runtime::to_string(mask) + " Failed");
	}

	logger.Info("Zero-copy buffers allocated and bound: " +
	            std::to_string(n_in) + " inputs, " +
	            std::to_string(n_out) + " outputs.", kcurrent_module_name);
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
void InferenceEngineRKNNZeroCP::infer(
    const std::vector<TensorData>& inputs,
          std::vector<TensorData>& outputs)
{
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	const uint32_t n_in  = io_num_.n_input;
	const uint32_t n_out = io_num_.n_output;

	if (inputs.size() != static_cast<std::size_t>(n_in)) {
		throw std::runtime_error(
		    "Input count mismatch: model expects " + std::to_string(n_in) +
		    " inputs, got " + std::to_string(inputs.size()));
	}

	auto t0 = std::chrono::high_resolution_clock::now();

	// ----------------------------------------------------------------
	// Step 1 - Write data into zero-copy input buffers
	// ----------------------------------------------------------------
	for (uint32_t i = 0; i < n_in; ++i) {
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

	// ----------------------------------------------------------------
	// Step 2 - Execute inference on NPU
	// ----------------------------------------------------------------
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	auto t2 = std::chrono::high_resolution_clock::now();

	// ----------------------------------------------------------------
	// Step 3 - Read outputs from zero-copy buffers
	// ----------------------------------------------------------------
	outputs.clear();
	outputs.reserve(n_out);

	for (uint32_t i = 0; i < n_out; ++i) {
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
	if (isDumpEnabled() && !output_bin_path_.empty()) {
		file_utils_.dumpBinary(outputs[0].data,
		    output_bin_path_ + "cpp_08_inference-Output.bin");
	}

	auto t4 = std::chrono::high_resolution_clock::now();

	// ----------------------------------------------------------------
	// Profiling timestamps
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

	logger.Info("infer() complete: " + std::to_string(n_in) + " in / " +
	            std::to_string(n_out) + " out", kcurrent_module_name);
}
