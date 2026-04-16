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
// Manages the persistent recurrent states (r1~r4) for RVM (Robust Video
// Matting) across frames.
//
// Design rationale:
//   - InferenceEngine is stateless (pure "inputs in, outputs out").
//   - Recurrent state lifecycle is a Pipeline-layer concern, NOT an
//     InferenceEngine concern.
//   - This class owns the r1~r4 buffers and handles:
//       1. init()   — allocate zero-filled state buffers from shape specs
//       2. reset()  — zero-fill all states (first frame / scene cut)
//       3. inject() — append current states into the input vector
//       4. update() — capture new states from the output vector
//
// RVM tensor layout (5 inputs → 6 outputs):
//   inputs:  [src, r1i, r2i, r3i, r4i]
//   outputs: [fgr, pha, r1o, r2o, r3o, r4o]
//
// Usage in Pipeline::run():
//   state_mgr.init(state_shapes);
//   for (frame : frames) {
//       auto inputs = { frontend.preprocess(frame) };
//       state_mgr.inject(inputs);        // appends r1i~r4i
//       engine.infer(inputs, outputs);
//       state_mgr.update(outputs);       // captures r1o~r4o
//       backend.postprocess(outputs);
//   }
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "pipeline/core/data_structure.h"

class RecurrentStateManager {
   public:
	// Initialize state buffers from shape specifications.
	// state_shapes: shapes for r1, r2, r3, r4 (4 entries).
	// state_names:  names  for r1, r2, r3, r4 (4 entries, e.g. "r1i", "r2i", ...).
	void init(const std::vector<std::vector<int64_t>>& state_shapes,
	          const std::vector<std::string>& state_names);

	// Reset all recurrent states to zero (first frame or scene cut).
	void reset();

	// Append current states (r1i~r4i) to the end of `inputs`.
	// After this call, inputs will have its original elements + 4 state tensors.
	void inject(std::vector<TensorData>& inputs) const;

	// Capture new states from outputs.
	// Expects outputs[2..5] to be r1o~r4o (indices after fgr and pha).
	// state_output_offset: index of the first state in outputs (default = 2).
	void update(const std::vector<TensorData>& outputs,
	            std::size_t state_output_offset = 2);

	bool isFirstFrame() const { return first_frame_; }
	std::size_t stateCount() const { return states_.size(); }

   private:
	std::vector<TensorData> states_;  // r1~r4 persistent buffers
	bool first_frame_ = true;
};
