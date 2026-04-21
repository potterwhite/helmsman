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

// ============================================================================
// ONNX Inference Engine
//
// Processing Flow:
//
//   Step 1  - Create ONNX Runtime session
//   Step 2  - Translate data layout (NHWC → NCHW)
//   Step 3  - Restore normalization (0~255 → 0~1)
//   Step 4  - Create ONNX input tensor
//   Step 5  - Run inference
//   Step 6  - Extract output tensor
//   Step 7  - Assemble TensorData structure
//
// Important:
//   • Frontend provides image in NHWC format (0~255 range)
//   • ONNX model expects NCHW format (0~1 normalized)
//   • This engine performs layout translation + normalization recovery
// ============================================================================

#include "pipeline/stages/inference-engine/onnx/onnx.h"
#include "common/common-define.h"

// InferenceEngineONNX& InferenceEngineONNX::GetInstance() {
// 	static InferenceEngineONNX instance;
// 	return instance;
// }

InferenceEngineONNX::InferenceEngineONNX()
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx-inference-engine") {

	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineONNX object constructed. (CPU Mode)", kcurrent_module_name);
}

InferenceEngineONNX::~InferenceEngineONNX() {

	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineONNX cleaned up.");
}

// ============================================================================
// Step 1 - Load ONNX Model and Create Session
// ============================================================================
void InferenceEngineONNX::load(const std::string& model_path) {

	auto& logger  = arcforge::embedded::utils::Logger::GetInstance();
	auto& runtime = arcforge::runtime::RuntimeONNX::GetInstance();

	session_ = std::make_unique<Ort::Session>(
	    env_, model_path.c_str(), runtime.init_session_option());

	Ort::AllocatorWithDefaultOptions allocator;

	const std::size_t n_inputs  = session_->GetInputCount();
	const std::size_t n_outputs = session_->GetOutputCount();

	input_names_.clear();
	output_names_.clear();

	for (std::size_t i = 0; i < n_inputs; ++i) {
		input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
	}
	for (std::size_t i = 0; i < n_outputs; ++i) {
		output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
	}

	logger.Info("Loaded ONNX model: " + model_path, kcurrent_module_name);
	logger.Info("  inputs:  " + std::to_string(n_inputs),  kcurrent_module_name);
	logger.Info("  outputs: " + std::to_string(n_outputs), kcurrent_module_name);
	for (auto& n : input_names_)  logger.Info("    in  " + n, kcurrent_module_name);
	for (auto& n : output_names_) logger.Info("    out " + n, kcurrent_module_name);
}

// ============================================================================
// Step 2~7 - Run Inference (N inputs → M outputs)
//
// Input layout contract:
//   - inputs[0] (image/src): NHWC float32, range 0~255
//     → converted to NCHW + normalized to 0~1 before ORT run
//   - inputs[1..N-1] (recurrent states r*i): already NCHW float32, raw values
//     → passed through as-is (no layout conversion, no normalization)
//
// Output layout:
//   - all outputs are NCHW float32 as returned by ORT
// ============================================================================
void InferenceEngineONNX::infer(
    const std::vector<TensorData>& inputs,
          std::vector<TensorData>& outputs)
{
	auto& logger     = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils = arcforge::utils::FileUtils::GetInstance();

	const std::size_t n_in  = inputs.size();
	const std::size_t n_out = output_names_.size();

	// ----------------------------------------------------------------
	// Build ORT input tensors
	// ----------------------------------------------------------------
	Ort::MemoryInfo mem_info =
	    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// We need stable pointers: keep converted buffers alive until Run().
	std::vector<std::vector<float>> converted_bufs(n_in);
	std::vector<Ort::Value>         ort_inputs;
	ort_inputs.reserve(n_in);

	for (std::size_t i = 0; i < n_in; ++i) {
		const TensorData& td = inputs[i];

		if (i == 0) {
			// --- Image input: NHWC 0~255 → NCHW 0~1 ---
			const int64_t N = td.shape[0];
			const int64_t H = td.shape[1];
			const int64_t W = td.shape[2];
			const int64_t C = td.shape[3];
			const std::size_t sN = static_cast<std::size_t>(N);
			const std::size_t sH = static_cast<std::size_t>(H);
			const std::size_t sW = static_cast<std::size_t>(W);
			const std::size_t sC = static_cast<std::size_t>(C);
			const std::size_t total = sN * sC * sH * sW;

			converted_bufs[i].resize(total);
			for (std::size_t c = 0; c < sC; ++c)
				for (std::size_t h = 0; h < sH; ++h)
					for (std::size_t w = 0; w < sW; ++w)
						converted_bufs[i][c*sH*sW + h*sW + w] =
						    td.data[h*sW*sC + w*sC + c] / 255.0f;

			std::vector<int64_t> shape_nchw = {N, C, H, W};
			ort_inputs.push_back(Ort::Value::CreateTensor<float>(
			    mem_info, converted_bufs[i].data(), total,
			    shape_nchw.data(), shape_nchw.size()));
		} else {
			// --- Recurrent state: NCHW float32, pass through ---
			const std::size_t total = td.data.size();
			converted_bufs[i] = td.data;   // copy (ORT needs non-const ptr)
			ort_inputs.push_back(Ort::Value::CreateTensor<float>(
			    mem_info, converted_bufs[i].data(), total,
			    td.shape.data(), td.shape.size()));
		}
	}

	// ----------------------------------------------------------------
	// Prepare name pointer arrays for ORT Run()
	// ----------------------------------------------------------------
	std::vector<const char*> in_ptrs, out_ptrs;
	in_ptrs.reserve(n_in);
	out_ptrs.reserve(n_out);
	for (auto& s : input_names_)  in_ptrs.push_back(s.c_str());
	for (auto& s : output_names_) out_ptrs.push_back(s.c_str());

	// ----------------------------------------------------------------
	// Run inference
	// ----------------------------------------------------------------
	auto ort_outputs = session_->Run(
	    Ort::RunOptions{nullptr},
	    in_ptrs.data(),  ort_inputs.data(),  n_in,
	    out_ptrs.data(), n_out);

	// ----------------------------------------------------------------
	// Collect outputs
	// ----------------------------------------------------------------
	outputs.clear();
	outputs.reserve(n_out);

	for (std::size_t j = 0; j < n_out; ++j) {
		float*       data_ptr = ort_outputs[j].GetTensorMutableData<float>();
		auto         shape    = ort_outputs[j].GetTensorTypeAndShapeInfo().GetShape();
		std::size_t  total    = 1;
		for (auto s : shape) total *= static_cast<std::size_t>(s);

		TensorData td;
		td.name  = output_names_[j];
		td.shape = shape;
		td.data.assign(data_ptr, data_ptr + total);

		// Propagate letterbox metadata from primary input (src)
		td.orig_width  = inputs[0].orig_width;
		td.orig_height = inputs[0].orig_height;
		td.pad_top     = inputs[0].pad_top;
		td.pad_bottom  = inputs[0].pad_bottom;
		td.pad_left    = inputs[0].pad_left;
		td.pad_right   = inputs[0].pad_right;

		outputs.push_back(std::move(td));
	}

	// Debug dump for primary output (index 0)
	if (isDumpEnabled() && !output_bin_path_.empty()) {
		file_utils.dumpBinary(outputs[0].data,
		    output_bin_path_ + "cpp_08_inference-Output.bin");
	}

	logger.Info("infer() complete: " + std::to_string(n_in) + " in / " +
	            std::to_string(n_out) + " out", kcurrent_module_name);
}