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

#pragma once

#include <rknn_api.h>                                                // rknn_xxx APIs
#include <string>                                                    // for std::string
#include <vector>                                                    // for std::vector
#include "pipeline/stages/inference-engine/base/inference-engine.h"  // InferenceEngine

// ---------------------------------------------------------------------------
// InferenceEngineRKNNZeroCP — RKNN zero-copy inference (multi-tensor)
//
// Supports arbitrary N-input / M-output models:
//   - MODNet: N=1, M=1
//   - RVM:    N=5, M=6
//
// Buffer allocation is done once in load() based on the model's actual
// input/output count queried from the RKNN context.
// ---------------------------------------------------------------------------

class InferenceEngineRKNNZeroCP : public InferenceEngine {
   public:
	InferenceEngineRKNNZeroCP();
	~InferenceEngineRKNNZeroCP();

	void Load(const std::string& model_path) override;

	void SetCoreMask(int mask) { core_mask_ = mask; }
	void SetPerfEnabled(bool enabled) { perf_enabled_ = enabled; }

	// Get model input dimensions from the FIRST input tensor (image/src).
	// RKNN reports NHWC layout: dims[0]=batch, dims[1]=height, dims[2]=width, dims[3]=channels
	int GetInputHeight() const override {
		return input_attrs_.empty() ? 0 : static_cast<int>(input_attrs_[0].dims[1]);
	}
	int GetInputWidth() const override {
		return input_attrs_.empty() ? 0 : static_cast<int>(input_attrs_[0].dims[2]);
	}

	// Return the shapes of recurrent state inputs (inputs 1..N).
	// For RVM: inputs[1]=r1i, inputs[2]=r2i, inputs[3]=r3i, inputs[4]=r4i.
	// Returns shapes in the model's native layout (NHWC for RKNN).
	std::vector<std::vector<int64_t>> GetRecurrentStateShapes() const override {
		std::vector<std::vector<int64_t>> shapes;
		// Skip input 0 (image/src), return shapes for inputs 1..N-1
		for (size_t i = 1; i < input_attrs_.size(); ++i) {
			std::vector<int64_t> shape;
			for (uint32_t d = 0; d < input_attrs_[i].n_dims; ++d) {
				shape.push_back(static_cast<int64_t>(input_attrs_[i].dims[d]));
			}
			shapes.push_back(std::move(shape));
		}
		return shapes;
	}

   protected:
	void DoInfer(const std::vector<TensorData>& inputs, std::vector<TensorData>& outputs) override;

   private:
	// member functions
	void ReleaseBuffers();
	void WriteInputBuffers1st(const std::vector<TensorData>& inputs);
	void ExecuteNpu2nd();
	void ReadOutputBuffers3rd(const std::vector<TensorData>& inputs,
	                          std::vector<TensorData>& outputs);
	void LogInferenceProfile4th(double cast_in_ms, double npu_run_ms,
	                            double cast_out_ms, double dump_ms);

   private:
	// member variables
	rknn_context ctx_{0};
	rknn_input_output_num io_num_{};

	// Per-tensor attributes (one per input/output)
	std::vector<rknn_tensor_attr> input_attrs_;
	std::vector<rknn_tensor_attr> output_attrs_;

	// Per-tensor zero-copy memory (one per input/output)
	std::vector<rknn_tensor_mem*> input_mems_;
	std::vector<rknn_tensor_mem*> output_mems_;

	// NPU core mask (-1 = default/CORE_ALL)
	int core_mask_ = -1;
	bool perf_enabled_ = false;  // true if COLLECT_PERF_MASK was accepted
};
