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
#include "Runtime/rknn.h/rknn.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/other/other-utils.h"

// ============================================================================
// InferenceEngineRKNNZeroCP
//
// This class implements RKNN inference using Zero-Copy mode.
//
// Zero-Copy Design Philosophy:
//   • Allocate NPU-visible buffers once
//   • Bind buffers directly to input/output tensors
//   • Avoid rknn_inputs_set / rknn_outputs_get
//   • CPU writes directly into NPU memory
//   • CPU reads directly from NPU output buffer
//
// High-Level Lifecycle:
//
//   Step 1 - Initialize RKNN context
//   Step 2 - Query input/output attributes
//   Step 3 - Allocate zero-copy buffers
//   Step 4 - Bind buffers to tensors
//   Step 5 - Run inference
//   Step 6 - Read output directly from mapped memory
// ============================================================================

InferenceEngineRKNNZeroCP::InferenceEngineRKNNZeroCP() {
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNNZeroCP constructed. (Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNNZeroCP::~InferenceEngineRKNNZeroCP() {
	// Ensure buffers are released before destroying context
	releaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

// ============================================================================
// Release Zero-Copy Buffers
//
// Must be called before destroying context to avoid memory leak.
// ============================================================================
void InferenceEngineRKNNZeroCP::releaseBuffers() {
	if (input_mem_) {
		rknn_destroy_mem(ctx_, input_mem_);
		input_mem_ = nullptr;
	}

	if (output_mem_) {
		rknn_destroy_mem(ctx_, output_mem_);
		output_mem_ = nullptr;
	}
}

// ============================================================================
// Load Model and Initialize Zero-Copy Environment
//
// Major Phases:
//
//   Phase I   - Initialize RKNN context
//   Phase II  - Query input/output metadata
//   Phase III - Allocate zero-copy memory
//   Phase IV  - Bind memory to tensors
// ============================================================================
void InferenceEngineRKNNZeroCP::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// ------------------------------------------------------------------------
	// Phase I - RKNN Context Initialization
	// ------------------------------------------------------------------------

	// // I. Read RKNN model from binary buffer (disabled version)
	// std::ifstream file(model_path, std::ios::binary | std::ios::ate);
	// size_t model_size = static_cast<size_t>(file.tellg());
	// file.seekg(0);
	// std::vector<uint8_t> model_data(model_size);
	// file.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(model_size));
	// file.close();

	/* II. RKNN context initialization explanation:
	 *
	 * rknn_context *context:
	 *     Pointer to RKNN runtime context.
	 *
	 * void *model:
	 *     Either:
	 *       - Pointer to binary RKNN model data
	 *       - OR file path string to RKNN model
	 *
	 * uint32_t size:
	 *     > 0 : model is binary data
	 *     = 0 : model is file path
	 *
	 * uint32_t flag:
	 *     Initialization flags (default behavior = 0)
	 *
	 * rknn_init_extend:
	 *     Optional extended initialization config.
	 *     Pass NULL when not needed.
	 */

	// ---------------
	// Option 1: Binary buffer mode (disabled here)
	// int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);

	// ---------------
	// Option 2: File path mode (used here)
	int ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
	                    0, nullptr);

	if (ret < 0) {
		throw std::runtime_error("rknn_init failed!");
	} else {
		logger.Info("RKNN model loaded and context initialized successfully.",
		            kcurrent_module_name);
	}

	// ------------------------------------------------------------------------
	// Phase II - Query Model Input/Output Information
	// ------------------------------------------------------------------------

	// Step 2.1 Query number of input/output tensors
	rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

	logger.Info("Input num: " + std::to_string(io_num_.n_input), kcurrent_module_name);
	logger.Info("Output num: " + std::to_string(io_num_.n_output), kcurrent_module_name);

	// Step 2.2 Query input and output tensor attributes
	memset(&input_attr_, 0, sizeof(input_attr_));
	input_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));

	memset(&output_attr_, 0, sizeof(output_attr_));
	output_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));

	// Step 2.3 Echo input/output information for debugging
	input_size_ = input_attr_.size;
	output_size_ = output_attr_.size;

	logger.Info("Input size bytes: " + std::to_string(input_size_), kcurrent_module_name);
	logger.Info("Output size bytes: " + std::to_string(output_size_), kcurrent_module_name);

	logger.Info("Input attr: " + arcforge::runtime::to_string(input_attr_));
	logger.Info("Output attr: " + arcforge::runtime::to_string(output_attr_));

	// ------------------------------------------------------------------------
	// Phase III - Create Zero-Copy Buffers
	// ------------------------------------------------------------------------
	// rknn_create_mem allocates NPU-accessible memory.
	// This memory is shared between CPU and NPU.
	// No additional copy is needed during inference.
	// ------------------------------------------------------------------------
	input_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(input_size_));
	output_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(output_size_));

	if (!input_mem_ || !output_mem_) {
		throw std::runtime_error("rknn_create_mem failed!");
	}

	// ------------------------------------------------------------------------
	// Phase IV - Bind Memory to Tensor Descriptors
	// ------------------------------------------------------------------------
	// rknn_set_io_mem connects allocated memory with model input/output.
	// After binding:
	//   • Writing into input_mem_->virt_addr feeds the model
	//   • Reading from output_mem_->virt_addr retrieves results
	// ------------------------------------------------------------------------
	rknn_set_io_mem(ctx_, input_mem_, &input_attr_);
	rknn_set_io_mem(ctx_, output_mem_, &output_attr_);

	logger.Info("Zero-copy buffers allocated and bound.", kcurrent_module_name);
}

// ============================================================================
// Inference (Zero-Copy Mode, FP16 Model)
//
// Execution Steps:
//
//   Step 1 - Validate input element count (FP16-based validation)
//   Step 2 - Convert FLOAT32 → FP16 and write into zero-copy buffer
//   Step 3 - Run inference on NPU
//   Step 4 - Convert FP16 → FLOAT32 directly from output buffer
//   Step 5 - Construct TensorData structure
//
// Important:
//   • Model internal precision: FP16
//   • Frontend provides: FLOAT32
//   • Conversion happens explicitly in this function
// ============================================================================

TensorData InferenceEngineRKNNZeroCP::infer(const TensorData& input) {

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// ------------------------------------------------------------------------
	// Step 1 - Validate input size and detect model precision
	// ------------------------------------------------------------------------
	// Determine model precision based on input tensor type
	bool is_int8_model =
	    (input_attr_.type == RKNN_TENSOR_INT8 || input_attr_.type == RKNN_TENSOR_UINT8);
	bool is_fp16_model = (input_attr_.type == RKNN_TENSOR_FLOAT16);

	size_t expected_element_count;
	if (is_int8_model) {
		// INT8 model: 1 byte per element
		expected_element_count = input_size_ / sizeof(int8_t);
	} else if (is_fp16_model) {
		// FP16 model: 2 bytes per element
		expected_element_count = input_size_ / sizeof(__fp16);
	} else {
		// FP32 model: 4 bytes per element
		expected_element_count = input_size_ / sizeof(float);
	}

	if (input.data.size() != expected_element_count) {
		logger.Error("Input size mismatch! Expected " + std::to_string(expected_element_count) +
		                 " elements, but Frontend provided " + std::to_string(input.data.size()) +
		                 " elements.",
		             kcurrent_module_name);
		throw std::runtime_error("Input size mismatch with RKNN model.");
	}

	auto t0 = std::chrono::high_resolution_clock::now();

	// ------------------------------------------------------------------------
	// Step 2 - Write data into zero-copy input buffer (adaptive precision)
	// ------------------------------------------------------------------------
	if (is_int8_model) {
		// INT8 quantized model: FLOAT32 → Normalize → INT8
		//
		// CRITICAL OPTIMIZATION:
		// When RKNN config has normalization enabled, the toolkit inserts a normalization
		// operator at the beginning of the model graph. For INT8 models, this causes:
		//   INT8 input → dequantize to FP16 → normalize → quantize back to INT8
		// This adds ~100ms overhead (35% slower than FP16!) due to extra conversions.
		//
		// Solution: Normalize BEFORE quantization in C++ code, so the model receives
		// pre-normalized INT8 data directly, avoiding internal FP16 conversions.
		//
		// Normalization formula: normalized = (pixel - 127.5) / 127.5
		// This converts [0, 255] → [-1, 1]
		//
		// Then quantize: value_int8 = (normalized / scale) + zero_point
		//
		int8_t* input_int8_ptr = reinterpret_cast<int8_t*>(input_mem_->virt_addr);
		float scale = input_attr_.scale;
		float zp = static_cast<float>(input_attr_.zp);

		for (size_t i = 0; i < input.data.size(); ++i) {
			// Step 1: Normalize [0, 255] → [-1, 1]
			float normalized = (input.data[i] - 127.5f) / 127.5f;

			// Step 2: Quantize to INT8
			float quantized = (normalized / scale) + zp;

			// Step 3: Clamp to INT8 range [-128, 127]
			quantized = std::max(-128.0f, std::min(127.0f, quantized));
			input_int8_ptr[i] = static_cast<int8_t>(std::round(quantized));
		}
	} else if (is_fp16_model) {
		// FP16 model: FLOAT32 → FP16
		__fp16* input_fp16_ptr = reinterpret_cast<__fp16*>(input_mem_->virt_addr);
		for (size_t i = 0; i < input.data.size(); ++i) {
			input_fp16_ptr[i] = static_cast<__fp16>(input.data[i]);
		}
	} else {
		// FP32 model: FLOAT32 → FLOAT32 (direct copy)
		float* input_fp32_ptr = reinterpret_cast<float*>(input_mem_->virt_addr);
		std::memcpy(input_fp32_ptr, input.data.data(), input.data.size() * sizeof(float));
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	// ------------------------------------------------------------------------
	// Step 3 - Execute inference on NPU
	// ------------------------------------------------------------------------
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	// ------------------------------------------------------------------------
	// Step 4 - Read output directly from zero-copy buffer (adaptive precision)
	// ------------------------------------------------------------------------
	bool is_int8_output =
	    (output_attr_.type == RKNN_TENSOR_INT8 || output_attr_.type == RKNN_TENSOR_UINT8);
	bool is_fp16_output = (output_attr_.type == RKNN_TENSOR_FLOAT16);

	size_t output_element_count;
	if (is_int8_output) {
		output_element_count = output_size_ / sizeof(int8_t);
	} else if (is_fp16_output) {
		output_element_count = output_size_ / sizeof(__fp16);
	} else {
		output_element_count = output_size_ / sizeof(float);
	}

	std::vector<float> output_vector(output_element_count);

	if (is_int8_output) {
		// INT8 output: dequantize INT8 → FLOAT32
		// value_float32 = (value_int8 - zero_point) * scale
		int8_t* output_int8_ptr = reinterpret_cast<int8_t*>(output_mem_->virt_addr);
		float scale = output_attr_.scale;
		float zp = static_cast<float>(output_attr_.zp);

		for (size_t i = 0; i < output_element_count; ++i) {
			output_vector[i] = (static_cast<float>(output_int8_ptr[i]) - zp) * scale;
		}
	} else if (is_fp16_output) {
		// FP16 output: FP16 → FLOAT32
		__fp16* output_fp16_ptr = reinterpret_cast<__fp16*>(output_mem_->virt_addr);
		for (size_t i = 0; i < output_element_count; ++i) {
			output_vector[i] = static_cast<float>(output_fp16_ptr[i]);
		}
	} else {
		// FP32 output: direct copy
		float* output_fp32_ptr = reinterpret_cast<float*>(output_mem_->virt_addr);
		std::memcpy(output_vector.data(), output_fp32_ptr, output_element_count * sizeof(float));
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	// Dump output for binary-level debugging and verification
	file_utils_.dumpBinary(output_vector, output_bin_path_ + "cpp_08_inference-Output.bin");

	auto t4 = std::chrono::high_resolution_clock::now();
	// --- 打印内部精细耗时 ---
	std::chrono::duration<double, std::milli> cast_in_time = t1 - t0;
	std::chrono::duration<double, std::milli> npu_run_time = t2 - t1;
	std::chrono::duration<double, std::milli> cast_out_time = t3 - t2;
	std::chrono::duration<double, std::milli> dump_time = t4 - t3;

	std::string precision_str = is_int8_model ? "INT8" : (is_fp16_model ? "FP16" : "FP32");
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

	// ------------------------------------------------------------------------
	// Step 5 - Construct TensorData structure
	// ------------------------------------------------------------------------
	TensorData output;
	output.data = output_vector;

	// Assemble output shape from tensor metadata
	output.shape.clear();
	for (uint32_t i = 0; i < output_attr_.n_dims; i++) {
		output.shape.push_back(output_attr_.dims[i]);
	}

	// --- ADD THIS BLOCK ---
	// Inherit metadata from input tensor to output tensor
	output.orig_width = input.orig_width;
	output.orig_height = input.orig_height;
	output.pad_top = input.pad_top;
	output.pad_bottom = input.pad_bottom;
	output.pad_left = input.pad_left;
	output.pad_right = input.pad_right;
	// ----------------------

	return output;
}