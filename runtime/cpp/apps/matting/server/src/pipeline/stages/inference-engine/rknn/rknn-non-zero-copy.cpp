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

#include "pipeline/stages/inference-engine/rknn/rknn-non-zero-copy.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "RKNNKit/rknn-query.h"
#include "RKNNKit/utils.h"
#include "common/common-define.h"

using helmsman::rknnkit::RKNNQuery;

// ============================================================================
// InferenceEngineRKNN — N-input / M-output Non-Zero-Copy Mode
// ============================================================================

InferenceEngineRKNN::InferenceEngineRKNN() {
	helmsman::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNN object constructed. (Non-Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNN::~InferenceEngineRKNN() {
	release();
	helmsman::utils::Logger::GetInstance().Info("InferenceEngineRKNN cleaned up.",
	                                                      kcurrent_module_name);
}

void InferenceEngineRKNN::release() {
	if (ctx_ > 0) {
		rknn_destroy(ctx_);
		ctx_ = 0;
	}
}

// ============================================================================
// Load Model — same as original (already supports multi-tensor query)
// ============================================================================
void InferenceEngineRKNN::Load(const std::string& model_path) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// Step 1: Read RKNN model file into memory
	std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
	if (!ifs.is_open()) {
		logger.Error("Failed to open RKNN model file: " + model_path, kcurrent_module_name);
		throw std::runtime_error("Failed to open RKNN model");
	}

	ifs.seekg(0, std::ios::end);
	size_t model_size = static_cast<size_t>(ifs.tellg());
	ifs.seekg(0, std::ios::beg);

	std::vector<unsigned char> model_data(model_size);
	ifs.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(model_size));
	ifs.close();

	logger.Info("RKNN model loaded into memory, size: " + std::to_string(model_size) + " bytes.",
	            kcurrent_module_name);

	// Step 2: Initialize RKNN runtime context
	int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);
	if (ret < 0) {
		logger.Error("rknn_init fail! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_init failed");
	}

	// Step 3: Query model metadata via RKNNKit
	io_num_ = RKNNQuery::IoNum(ctx_);
	input_attrs_ = RKNNQuery::InputAttrs(ctx_, io_num_.n_input);
	output_attrs_ = RKNNQuery::OutputAttrs(ctx_, io_num_.n_output);
}

// ============================================================================
// Inference — N inputs → M outputs (Non-Zero-Copy)
// ============================================================================
void InferenceEngineRKNN::DoInfer(
    const std::vector<TensorData>& inputs,
          std::vector<TensorData>& outputs)
{
	auto& logger = helmsman::utils::Logger::GetInstance();
	auto& file_utils_ = helmsman::utils::FileUtils::GetInstance();

	if (ctx_ == 0) {
		throw std::runtime_error("RKNN Context is null, please load model first.");
	}

	const uint32_t n_in  = io_num_.n_input;
	const uint32_t n_out = io_num_.n_output;

	if (inputs.size() != static_cast<std::size_t>(n_in)) {
		throw std::runtime_error(
		    "Input count mismatch: model expects " + std::to_string(n_in) +
		    " inputs, got " + std::to_string(inputs.size()));
	}

	// ----------------------------------------------------------------
	// Step 1: Prepare and set input tensors
	// ----------------------------------------------------------------
	std::vector<rknn_input> rknn_inputs(n_in);
	memset(rknn_inputs.data(), 0, n_in * sizeof(rknn_input));

	for (uint32_t i = 0; i < n_in; ++i) {
		rknn_inputs[i].index       = i;
		rknn_inputs[i].type        = RKNN_TENSOR_FLOAT32;
		rknn_inputs[i].fmt         = input_attrs_[i].fmt;
		rknn_inputs[i].size        = static_cast<uint32_t>(inputs[i].data.size() * sizeof(float));
		rknn_inputs[i].buf         = const_cast<void*>(
		    reinterpret_cast<const void*>(inputs[i].data.data()));
		rknn_inputs[i].pass_through = 0;
	}

	int ret = rknn_inputs_set(ctx_, n_in, rknn_inputs.data());
	if (ret < 0) {
		logger.Error("rknn_inputs_set failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_inputs_set failed");
	}

	// ----------------------------------------------------------------
	// Step 2: Execute inference
	// ----------------------------------------------------------------
	ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		logger.Error("rknn_run failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_run failed");
	}

	// ----------------------------------------------------------------
	// Step 3: Retrieve output tensors
	// ----------------------------------------------------------------
	std::vector<rknn_output> rknn_outputs(n_out);
	memset(rknn_outputs.data(), 0, n_out * sizeof(rknn_output));

	for (uint32_t i = 0; i < n_out; ++i) {
		rknn_outputs[i].want_float  = 1;   // Request dequantized FLOAT32
		rknn_outputs[i].is_prealloc = 0;   // Let RKNN allocate output buffer
	}

	ret = rknn_outputs_get(ctx_, n_out, rknn_outputs.data(), nullptr);
	if (ret < 0) {
		logger.Error("rknn_outputs_get failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_outputs_get failed");
	}

	// ----------------------------------------------------------------
	// Step 4: Convert RKNN outputs to TensorData
	// ----------------------------------------------------------------
	outputs.clear();
	outputs.reserve(n_out);

	for (uint32_t i = 0; i < n_out; ++i) {
		TensorData td;
		td.name = output_attrs_[i].name;

		float* out_ptr = reinterpret_cast<float*>(rknn_outputs[i].buf);
		size_t elements = rknn_outputs[i].size / sizeof(float);
		td.data.assign(out_ptr, out_ptr + elements);

		td.shape.clear();
		for (uint32_t d = 0; d < output_attrs_[i].n_dims; ++d) {
			td.shape.push_back(output_attrs_[i].dims[d]);
		}

		// Propagate letterbox metadata from primary input
		td.orig_width  = inputs[0].orig_width;
		td.orig_height = inputs[0].orig_height;
		td.pad_top     = inputs[0].pad_top;
		td.pad_bottom  = inputs[0].pad_bottom;
		td.pad_left    = inputs[0].pad_left;
		td.pad_right   = inputs[0].pad_right;

		outputs.push_back(std::move(td));
	}

	// Debug dump for primary output
	if (IsDumpEnabled() && !output_bin_path_.empty()) {
		file_utils_.dumpBinary(outputs[0].data,
		    output_bin_path_ + "cpp_08_inference-Output-RKNN.bin");
	}

	// ----------------------------------------------------------------
	// Step 5: Release output buffers
	// ----------------------------------------------------------------
	rknn_outputs_release(ctx_, n_out, rknn_outputs.data());

	logger.Info("infer() complete: " + std::to_string(n_in) + " in / " +
	            std::to_string(n_out) + " out", kcurrent_module_name);
}
