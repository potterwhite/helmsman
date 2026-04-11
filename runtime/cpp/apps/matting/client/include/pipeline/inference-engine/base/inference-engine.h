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
#include <cstddef>
#include <string>
#include <vector>
#include "pipeline/core/data_structure.h"

// ---------------------------------------------------------------------------
// InferenceEngine — abstract inference interface
//
// Supports arbitrary N-input / M-output models:
//   - MODNet: N=1, M=1  (single image → single alpha matte)
//   - RVM:    N=5, M=6  (src + r1i~r4i → fgr + pha + r1o~r4o)
//   - Future: SAM2, etc. — no new subclass needed, just fill vectors
//
// Recurrent state management is the responsibility of the caller (Pipeline),
// NOT of InferenceEngine. InferenceEngine is stateless.
// ---------------------------------------------------------------------------

class InferenceEngine {
   public:
	virtual ~InferenceEngine() = default;

	virtual void load(const std::string& model_path) = 0;

	// Run inference: N inputs → M outputs.
	// Caller is responsible for pre-sizing `outputs` or leaving it empty
	// (implementations must resize outputs to match model output count).
	virtual void infer(
	    const std::vector<TensorData>& inputs,
	          std::vector<TensorData>& outputs
	) = 0;

	// Getters for model input dimensions (used by Pipeline to configure frontend).
	virtual std::size_t getInputHeight() const { return 0; }
	virtual std::size_t getInputWidth()  const { return 0; }

	// Optional: path for debug binary dumps.
	virtual void setOutputBinPath(const std::string& path) { output_bin_path_ = path; }

   protected:
	std::string output_bin_path_;
};
