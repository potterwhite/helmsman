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
#include "common/types.h"
#include "pipeline/infra/recurrent-state-manager.h"

// ---------------------------------------------------------------------------
// InferenceEngine — abstract inference interface
//
// Supports arbitrary N-input / M-output models:
//   - MODNet: N=1, M=1  (single image → single alpha matte)
//   - RVM:    N=5, M=6  (src + r1i~r4i → fgr + pha + r1o~r4o)
//   - Future: SAM2, etc. — no new subclass needed, just fill vectors
//
// Stateful inference: Infer() automatically handles recurrent state
// injection/capture and downsample_ratio injection. Subclasses implement
// DoInfer() for the pure stateless inference step.
// Stateless models (MODNet) skip InitRecurrentStates() — Infer() detects
// empty states and skips injection/capture automatically.
// ---------------------------------------------------------------------------

class InferenceEngine {
   public:
	virtual ~InferenceEngine() = default;

	virtual void Load(const std::string& model_path) = 0;

	// --- Stateful inference (public interface) ---
	// NVI(Non-Virtual Interface) :
	//		Infer() is the public entry point that handles state management and delegates to DoInfer().
	//
	// Automatically handles recurrent state injection/capture + dsr injection.
	// Delegates to DoInfer() for the actual model execution.
	void Infer(const std::vector<TensorData>& inputs,
	           std::vector<TensorData>& outputs);

	// --- Initialization (RVM-specific; MODNet and other stateless models skip this) ---
	void InitRecurrentStates();
	void SetDownsampleRatio(float dsr);

	// --- Query (unchanged) ---
	virtual int GetInputHeight() const { return 0; }
	virtual int GetInputWidth()  const { return 0; }
	virtual std::vector<std::vector<int64_t>> GetRecurrentStateShapes() const { return {}; }
	virtual bool NeedsDownsampleRatio() const { return false; }

	// Optional: path for debug binary dumps.
	virtual void SetOutputBinPath(const std::string& path) { output_bin_path_ = path; }

   protected:
	// NVI hook: subclasses implement pure stateless inference (N inputs → M outputs).
	// Google Style: DoX() is the virtual body of public X().
	virtual void DoInfer(const std::vector<TensorData>& inputs,
	                       std::vector<TensorData>& outputs) = 0;

	std::string output_bin_path_;

   private:
	RecurrentStateManager state_mgr_;
	float dsr_ = 0.25f;
};
