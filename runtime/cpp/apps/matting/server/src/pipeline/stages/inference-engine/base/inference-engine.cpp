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
    if (shapes.size() == 4) {
        state_mgr_.init(shapes, {"r1i", "r2i", "r3i", "r4i"});
    } else {
        state_mgr_.init({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                        {"r1i", "r2i", "r3i", "r4i"});
    }
}

void InferenceEngine::SetDownsampleRatio(float dsr) {
    dsr_ = dsr;
}

void InferenceEngine::Infer(const std::vector<TensorData>& inputs,
                            std::vector<TensorData>& outputs) {
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

    DoInfer(mutable_inputs, outputs);

    // Capture recurrent states (r1o~r4o) from outputs.
    // Skip for stateless models — update() requires outputs to have
    // at least state_output_offset (2) + state_count elements.
    if (state_mgr_.stateCount() > 0) {
        state_mgr_.update(outputs);
    }
}
