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

// ============================================================================
// RecurrentStateManager
//
// Manages persistent recurrent state tensors across video frames.
//
// Design rationale:
//   - InferenceEngine is stateless (pure "inputs in, outputs out").
//   - Recurrent state lifecycle is a Pipeline-layer concern, NOT an
//     InferenceEngine concern.
//   - This class owns the state buffers and handles:
//       1. init()   — allocate zero-filled state buffers from StateSpec
//       2. inject() — append current states into the input vector
//       3. update() — capture new states from the output vector by name
//
// RVM tensor layout (5 inputs → 6 outputs):
//   inputs:  [src, r1i, r2i, r3i, r4i]
//   outputs: [fgr, pha, r1o, r2o, r3o, r4o]
//
// Usage:
//   state_mgr.init({{"r1i","r1o",{1,16,H,W}}, {"r2i","r2o",{1,20,H/2,W/2}}, ...});
//   for (frame : frames) {
//       auto inputs = { frontend.preprocess(frame) };
//       state_mgr.inject(inputs);        // appends r1i~r4i by input_name
//       engine.Infer(inputs, outputs);
//       state_mgr.update(outputs);       // captures r1o~r4o by output_name
//       backend.Postprocess(outputs);
//   }
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "common/types.h"

// ---------------------------------------------------------------------------
// StateSpec — Describes one recurrent state tensor's lifecycle across frames.
//
// input_name:  tensor name when injected into model inputs  (e.g. "r1i")
// output_name: tensor name when captured from model outputs (e.g. "r1o")
// shape:       initial buffer shape
// transpose_nchw_to_nhwc: if true, transpose output from NCHW→NHWC on capture
// ---------------------------------------------------------------------------
struct StateSpec {
	std::string input_name;
	std::string output_name;
	std::vector<int64_t> shape;
	bool transpose_nchw_to_nhwc = true;
};

// ---------------------------------------------------------------------------
// RecurrentStateManager
// ---------------------------------------------------------------------------
class RecurrentStateManager {
   public:
	// Initialize state buffers from explicit specifications.
	// Each StateSpec describes one recurrent state's input name, output name,
	// initial shape, and whether to transpose on capture.
	void init(const std::vector<StateSpec>& specs);

	// Append current recurrent states to `inputs`, in the order defined by specs.
	// Each pushed TensorData carries its input_name from the spec.
	void inject(std::vector<TensorData>& inputs) const;

	// Capture recurrent states from `outputs`, matched by output_name.
	// For each spec, searches `outputs` for a tensor with the matching name,
	// copies its data (with optional NCHW→NHWC transpose) into the persistent buffer.
	// Throws std::runtime_error if an expected output tensor is not found.
	void update(const std::vector<TensorData>& outputs);

	std::size_t stateCount() const;

   private:
	struct StateEntry {
		TensorData buffer;            // persistent storage (persists across frames)
		std::string input_name;       // e.g. "r1i"
		std::string output_name;      // e.g. "r1o"
		bool transpose_nchw_to_nhwc;
	};

	std::vector<StateEntry> states_;
};
