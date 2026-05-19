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

#include "pipeline/recurrent-state-manager.h"
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "Utils/logger/logger.h"
#include "common/common-define.h"

// ============================================================================
// init() — Allocate zero-filled state buffers
// ============================================================================
void RecurrentStateManager::init(const std::vector<std::vector<int64_t>>& state_shapes,
                                 const std::vector<std::string>& state_names) {
	if (state_shapes.size() != state_names.size()) {
		throw std::invalid_argument(
		    "RecurrentStateManager::init(): state_shapes and state_names must have equal size");
	}

	states_.clear();
	states_.reserve(state_shapes.size());

	for (std::size_t i = 0; i < state_shapes.size(); ++i) {
		TensorData td;
		td.name = state_names[i];
		td.shape = state_shapes[i];

		// Calculate total element count from shape
		std::size_t total = 1;
		for (auto dim : td.shape) {
			total *= static_cast<std::size_t>(dim);
		}
		td.data.assign(total, 0.0f);  // zero-fill

		states_.push_back(std::move(td));
	}

	first_frame_ = true;
}

// ============================================================================
// reset() — Zero-fill all state buffers (first frame / scene cut)
// ============================================================================
void RecurrentStateManager::reset() {
	for (auto& td : states_) {
		std::fill(td.data.begin(), td.data.end(), 0.0f);
	}
	first_frame_ = true;
}

// ============================================================================
// inject() — Append current r1i~r4i states to the input vector
// ============================================================================
void RecurrentStateManager::inject(std::vector<TensorData>& inputs) const {
	for (const auto& state : states_) {
		inputs.push_back(state);
	}
}

// ============================================================================
// update() — Capture r1o~r4o from outputs into persistent buffers
//
// After RVM inference, outputs layout is:
//   [0] = fgr,  [1] = pha,  [2] = r1o,  [3] = r2o,  [4] = r3o,  [5] = r4o
//
// state_output_offset defaults to 2 (skipping fgr and pha).
// ============================================================================
void RecurrentStateManager::update(const std::vector<TensorData>& outputs,
                                   std::size_t state_output_offset) {
	if (outputs.size() < state_output_offset + states_.size()) {
		throw std::runtime_error(
		    "RecurrentStateManager::update(): outputs too small. Expected at least " +
		    std::to_string(state_output_offset + states_.size()) + " elements, got " +
		    std::to_string(outputs.size()));
	}

	for (std::size_t i = 0; i < states_.size(); ++i) {
		const TensorData& src = outputs[state_output_offset + i];

		// // ---- s20 diagnostic: stride padding check ----
		// {
		// 	auto& logger = helmsman::utils::Logger::GetInstance();
		// 	const int64_t N = src.shape[0];
		// 	const int64_t C = src.shape[1];
		// 	const int64_t H = src.shape[2];
		// 	const int64_t W = src.shape[3];
		// 	const std::size_t expected_elements = static_cast<std::size_t>(N * C * H * W);
		// 	const std::size_t actual_elements = src.data.size();

		// 	std::ostringstream oss;
		// 	oss << "[s20-diag] " << src.name << "  shape=[" << N << "," << C << "," << H << "," << W
		// 	    << "]"
		// 	    << "  expected_elements=" << expected_elements
		// 	    << "  actual_elements=" << actual_elements
		// 	    << "  match=" << (expected_elements == actual_elements ? "YES" : "NO");
		// 	if (expected_elements != actual_elements) {
		// 		oss << "  ** STRIDE PADDING DETECTED **";
		// 	}
		// 	// 使用 spdlog 或 logger（参考项目现有的 log 调用方式）
		// 	logger.Info(oss.str(), kcurrent_module_name);
		// }
		// // ---- end s20 diagnostic ----

		{
			const int64_t N = src.shape[0];
			const int64_t C = src.shape[1];
			const int64_t H = src.shape[2];
			const int64_t W = src.shape[3];

			std::vector<float> transposed(static_cast<size_t>(N * H * W * C));

			for (int64_t c = 0; c < C; ++c) {
				for (int64_t h = 0; h < H; ++h) {
					for (int64_t w = 0; w < W; ++w) {
						std::size_t src_idx = static_cast<std::size_t>((c * H * W) + (h * W) + w);
						std::size_t dst_idx = static_cast<std::size_t>((h * W * C) + (w * C) + c);

						// transpose from NCHW to NHWC
						transposed[dst_idx] = src.data[src_idx];
					}  // for w
				}  // for h
			}  // for c

			states_[i].data = std::move(transposed);
			states_[i].shape = {N, H, W, C};  // update shape to NHWC
		}  // Note: The above loop is a placeholder

		/* ---------------------------------- Obsoleted
		// // Copy data and shape; keep the input-side name (r*i)
		// states_[i].data  = src.data;
		// states_[i].shape = src.shape;
		---------------------------------------------*/

		first_frame_ = false;
	}
}