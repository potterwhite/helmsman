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

#include "pipeline/stages/inference-engine/base/inference-engine.h"

void InferenceEngine::InitRecurrentStates() {
	auto shapes = GetRecurrentStateShapes();
	const std::vector<std::string> in_names = {"r1i", "r2i", "r3i", "r4i"};
	const std::vector<std::string> out_names = {"r1o", "r2o", "r3o", "r4o"};

	std::vector<StateSpec> specs;
	specs.reserve(4);
	for (std::size_t i = 0; i < 4; ++i) {
		specs.push_back(StateSpec{
		    in_names[i],
		    out_names[i],
		    (shapes.size() == 4) ? shapes[i] : std::vector<int64_t>{1, 1, 1, 1},
		    true,
		});
	}
	state_mgr_.init(specs);
}

void InferenceEngine::SetDownsampleRatio(float dsr) {
	dsr_ = dsr;
}

void InferenceEngine::Infer(const std::vector<TensorData>& inputs,
                            std::vector<TensorData>& outputs) {
	// ================================================================
	// INFERENCE ENGINE SCOPE — Pre-inference: prepare model inputs
	//
	// Inject recurrent states (RVM) and downsample ratio (ONNX).
	// Stateless models (MODNet): inject() is a no-op, dsr skipped.
	// ================================================================
	std::vector<TensorData> mutable_inputs = inputs;

	// Inject recurrent states (r1i~r4i) if initialized.
	// Stateless models (e.g. MODNet) never call InitRecurrentStates(),
	// so states_ is empty and inject() is a no-op — this is by design.
	state_mgr_.inject(mutable_inputs);

	if (NeedsDownsampleRatio()) {
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {dsr_};
		mutable_inputs.push_back(std::move(dsr));
	}

	// ================================================================
	// INFERENCE ENGINE SCOPE — Model execution (subclass-specific)
	//
	// DoInfer() handles: input conversion → NPU/ORT run → output conversion.
	// After this call, `outputs` contains float32 tensors ready for
	// the MattingBackend (Postprocess = resize + composite + write).
	// ================================================================
	DoInfer(mutable_inputs, outputs);

	// ================================================================
	// INFERENCE ENGINE SCOPE — Post-inference: capture recurrent states
	// ================================================================
	if (state_mgr_.stateCount() > 0) {
		state_mgr_.update(outputs);
	}
	// --- END INFERENCE ENGINE SCOPE ---
	// Caller (RVMMode/MODNetMode) will pass `outputs` to
	// MattingBackend::Postprocess() for resize + composite + write.
}

// ---------------------------------------------------------------------------
// Virtual method default implementations
// ---------------------------------------------------------------------------

int InferenceEngine::GetInputHeight() const {
	return 0;
}
int InferenceEngine::GetInputWidth() const {
	return 0;
}

std::vector<std::vector<int64_t>> InferenceEngine::GetRecurrentStateShapes() const {
	return {};
}

bool InferenceEngine::NeedsDownsampleRatio() const {
	return false;
}

void InferenceEngine::SetOutputBinPath(const std::string& path) {
	output_bin_path_ = path;
}

void InferenceEngine::SetDumpEnabled(bool enabled) {
	dump_enabled_ = enabled;
}

void InferenceEngine::SetDiagEnabled(bool enabled) {
	diag_enabled_ = enabled;
}

bool InferenceEngine::SwapRecurrentStateBuffers(std::size_t /*n_states*/,
                                                std::size_t /*input_offset*/,
                                                std::size_t /*output_offset*/) {
	return false;
}
