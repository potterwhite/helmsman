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
#include <memory>                                                    // for std::unique_ptr
#include <string>                                                    // for std::string
#include <vector>                                                    // for std::vector
#include "pipeline/stages/inference-engine/engine-core/inference-engine.h"  // InferenceEngine

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
	~InferenceEngineRKNNZeroCP();

	explicit InferenceEngineRKNNZeroCP(const AppConfig& config);

	// ------------------------------------------------------------------
	// getter & setter
	// Get model input dimensions from the FIRST input tensor (image/src).
	// RKNN reports NHWC layout: dims[0]=batch, dims[1]=height, dims[2]=width, dims[3]=channels
	int GetInputHeight() const override;
	int GetInputWidth() const override;

	// ------------------------------------------------------------------
	//
	void Load(const std::string& model_path) override;

	// Return the shapes of recurrent state inputs (inputs 1..N).
	// For RVM: inputs[1]=r1i, inputs[2]=r2i, inputs[3]=r3i, inputs[4]=r4i.
	// Returns shapes in the model's native layout (NHWC for RKNN).
	std::vector<std::vector<int64_t>> GetRecurrentStateShapes() const override;

	// Per-sub-step timings from the most recent Infer() call.
	std::vector<std::pair<std::string, double>> GetLastSubTimings() const override;

   protected:
	void DoInfer(const std::vector<TensorData>& inputs, std::vector<TensorData>& outputs) override;
	bool SwapRecurrentStateBuffers(std::size_t n_states, std::size_t input_offset,
	                              std::size_t output_offset) override;
	void DoReportSubStepTimers(bool timing_enabled, helmsman::utils::Logger& logger,
	                            std::string_view module) const override;

   private:
	// member functions
	void WriteInputBuffers1st(const std::vector<TensorData>& inputs);
	void ExecuteNpu2nd();
	void ReadOutputBuffers3rd(const std::vector<TensorData>& inputs,
	                          std::vector<TensorData>& outputs);

	void ReleaseBuffers();

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

	// Sub-step timing accumulators (reported via DoReportSubStepTimers)
	using sa = helmsman::utils::timing::StageAccumulator;
	sa acc_write_input_{"  1/3-write_input_buffers"};
	sa acc_execute_npu_{"  2/3-execute_npu"};
	sa acc_read_output_{"  3/3-read_output_buffers"};

	// Per-input-tensor write timing (s5_8_22_15 Exp1: granular profiling)
	// acc_write_src_  = image tensor (input[0]), typically 60~65% of write time
	// acc_write_rstate_ = recurrent states (input[1..N]), combined
	sa acc_write_src_{"    1a-write_src"};
	sa acc_write_rstate_{"    1b-write_rstate"};

	// Latest per-frame sub-step timings (returned by GetLastSubTimings)
	double last_write_input_ms_ = 0;
	double last_execute_npu_ms_ = 0;
	double last_read_output_ms_ = 0;
};
