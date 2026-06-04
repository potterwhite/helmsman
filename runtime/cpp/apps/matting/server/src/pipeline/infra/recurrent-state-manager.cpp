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

#include "pipeline/infra/recurrent-state-manager.h"
#include <stdexcept>

// ============================================================================
// init() — Allocate zero-filled state buffers from StateSpec definitions.
// ============================================================================
void RecurrentStateManager::init(const std::vector<StateSpec>& specs) {
	states_.clear();
	states_.reserve(specs.size());

	for (const auto& spec : specs) {
		StateEntry entry;
		entry.input_name = spec.input_name;
		entry.output_name = spec.output_name;
		entry.transpose_nchw_to_nhwc = spec.transpose_nchw_to_nhwc;

		entry.buffer.name = spec.input_name;
		entry.buffer.shape = spec.shape;

		std::size_t total = 1;
		for (auto dim : spec.shape) {
			total *= static_cast<std::size_t>(dim);
		}
		entry.buffer.data.assign(total, 0.0f);

		states_.push_back(std::move(entry));
	}
}

// ============================================================================
// inject() — Append current recurrent states to `inputs`, in spec order.
// ============================================================================
void RecurrentStateManager::inject(std::vector<TensorData>& inputs) const {
	for (const auto& entry : states_) {
		inputs.push_back(entry.buffer);
	}
}

// ============================================================================
// update() — Capture recurrent states from `outputs` by output_name.
//
// For each state, searches `outputs` for a tensor whose name matches
// the spec's output_name, then copies data (with optional NCHW→NHWC
// transpose) into the persistent buffer.
// ============================================================================
void RecurrentStateManager::update(const std::vector<TensorData>& outputs) {
	for (auto& entry : states_) {
		// Find output tensor by name
		const TensorData* src = nullptr;
		for (const auto& out : outputs) {
			if (out.name == entry.output_name) {
				src = &out;
				break;
			}
		}
		if (!src) {
			throw std::runtime_error(
			    "RecurrentStateManager::update(): output tensor '" + entry.output_name +
			    "' not found in outputs");
		}

		if (entry.transpose_nchw_to_nhwc) {
			const int64_t N = src->shape[0];
			const int64_t C = src->shape[1];
			const int64_t H = src->shape[2];
			const int64_t W = src->shape[3];

			std::vector<float> transposed(static_cast<size_t>(N * H * W * C));

			for (int64_t c = 0; c < C; ++c) {
				for (int64_t h = 0; h < H; ++h) {
					for (int64_t w = 0; w < W; ++w) {
						std::size_t src_idx = static_cast<std::size_t>((c * H * W) + (h * W) + w);
						std::size_t dst_idx = static_cast<std::size_t>((h * W * C) + (w * C) + c);
						transposed[dst_idx] = src->data[src_idx];
					}
				}
			}

			entry.buffer.data = std::move(transposed);
			entry.buffer.shape = {N, H, W, C};
		} else {
			entry.buffer.data = src->data;
			entry.buffer.shape = src->shape;
		}
	}
}

std::size_t RecurrentStateManager::stateCount() const {
	return states_.size();
}
