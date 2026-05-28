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

#include "pipeline/stages/inference-engine/rknn/rknn-zero-copy.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include "RKNNKit/rknn-memory.h"
#include "RKNNKit/rknn-query.h"
#include "RKNNKit/utils.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/other/other-utils.h"
#include "common/common-define.h"

using helmsman::rknnkit::RKNNMemory;
using helmsman::rknnkit::RKNNQuery;

// ============================================================================
// InferenceEngineRKNNZeroCP — N-input / M-output Zero-Copy Mode
//
// Changes from the original 1-in/1-out version:
//   - input_mem_ / output_mem_ (single) → input_mems_ / output_mems_ (vector)
//   - input_attr_ / output_attr_ (single) → input_attrs_ / output_attrs_ (vector)
//   - load() queries ALL input/output attrs and allocates ALL buffers
//   - infer() iterates over all inputs/outputs
//   - MODNet (1-in/1-out) is the trivial special case of this general path
// ============================================================================

InferenceEngineRKNNZeroCP::InferenceEngineRKNNZeroCP() {
	helmsman::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNNZeroCP constructed. (Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNNZeroCP::~InferenceEngineRKNNZeroCP() {
	ReleaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
void InferenceEngineRKNNZeroCP::SetCoreMask(int mask) {
	core_mask_ = mask;
}
void InferenceEngineRKNNZeroCP::SetPerfEnabled(bool enabled) {
	perf_enabled_ = enabled;
}

int InferenceEngineRKNNZeroCP::GetInputHeight() const {
	return input_attrs_.empty() ? 0 : static_cast<int>(input_attrs_[0].dims[1]);
}

int InferenceEngineRKNNZeroCP::GetInputWidth() const {
	return input_attrs_.empty() ? 0 : static_cast<int>(input_attrs_[0].dims[2]);
}

std::vector<std::vector<int64_t>> InferenceEngineRKNNZeroCP::GetRecurrentStateShapes() const {
	// Extract recurrent state shapes from input_attrs_ (skip input[0] = image/src).
	//
	// RVM model has 5 inputs, all NHWC FP16:
	//   Input[0] "src" : [1, H, W, 3]       — image tensor (skipped)
	//   Input[1] "r1i": [1, H/4,  W/4,  16] — recurrent state 1
	//   Input[2] "r2i": [1, H/8,  W/8,  20] — recurrent state 2
	//   Input[3] "r3i": [1, H/16, W/16, 40] — recurrent state 3
	//   Input[4] "r4i": [1, H/32, W/32, 64] — recurrent state 4
	//
	// For 960x544 input: r1i=[1,136,240,16] r2i=[1,68,120,20]
	//                    r3i=[1,34,60,40]   r4i=[1,17,30,64]
	//
	// Returns shapes in model's native NHWC layout (batch, height, width, channels).
	// These shapes are used by RecurrentStateManager to allocate zero-filled
	// state buffers on the first frame.

	std::vector<std::vector<int64_t>> shapes;
	for (size_t i = 1; i < input_attrs_.size(); ++i) {
		std::vector<int64_t> shape;
		for (uint32_t d = 0; d < input_attrs_[i].n_dims; ++d) {
			int64_t dim = static_cast<int64_t>(input_attrs_[i].dims[d]);
			shape.push_back(dim);
		}
		shapes.push_back(std::move(shape));
	}

	if (diag_enabled_) {
		for (size_t i = 0; i < shapes.size(); ++i) {
			std::string dims;
			for (size_t d = 0; d < shapes[i].size(); ++d) {
				if (d > 0)
					dims += ", ";
				dims += std::to_string(shapes[i][d]);
			}
			GetLogger().Info(
			    "GetRecurrentStateShapes: shapes[" + std::to_string(i) + "] = {" + dims + "}",
			    kcurrent_module_name);
		}
	}

	return shapes;
}

// ============================================================================
// Release all zero-copy buffers
// ============================================================================
void InferenceEngineRKNNZeroCP::ReleaseBuffers() {
	RKNNMemory::FreeAll(ctx_, input_mems_);
	RKNNMemory::FreeAll(ctx_, output_mems_);
}

// ============================================================================
// Load Model and Initialize Zero-Copy Environment
//
// Phase I   - Initialize RKNN context
// Phase II  - Query ALL input/output metadata
// Phase III - Allocate zero-copy memory for EACH input and output
// Phase IV  - Bind memory to tensors
// Phase V   - Set NPU core mask
// ============================================================================
void InferenceEngineRKNNZeroCP::Load(const std::string& model_path) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// ----------------------------------------------------------------
	// Phase I - RKNN Context Initialization (file path mode)
	// ----------------------------------------------------------------
	// perf_enabled_ is set via --perf-enabled CLI flag (SetPerfEnabled()).
	// When enabled, pass RKNN_FLAG_COLLECT_PERF_MASK to rknn_init() so that
	// per-layer NPU profiling data is available via RKNN_QUERY_PERF_DETAIL.
	// Some SDK versions reject this flag in file-path mode — retry without it.
	//
	// History: §5.4 attempted RKNN_FLAG_DISABLE_FLUSH_*_MEM_CACHE together with
	// selective non-cacheable r-state buffers and A1 swap; result was 184ms →
	// 316ms infer regression (pkb §5.4). Reverted — runtime auto cache flush
	// stays on; all buffers stay cacheable.
	uint32_t init_flags = perf_enabled_ ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
	int ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
	                    init_flags, nullptr);

	if (ret < 0 && (init_flags & RKNN_FLAG_COLLECT_PERF_MASK)) {
		logger.Warning("rknn_init with COLLECT_PERF_MASK failed (ret=" + std::to_string(ret) +
		                   "), retrying without it.",
		               kcurrent_module_name);
		init_flags = 0;
		perf_enabled_ = false;
		ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
		                init_flags, nullptr);
	}
	if (ret < 0) {
		throw std::runtime_error("rknn_init failed! (ret=" + std::to_string(ret) + ")");
	}
	logger.Info("RKNN model loaded and context initialized. (COLLECT_PERF_MASK " +
	                std::string(perf_enabled_ ? "enabled" : "disabled") + ")",
	            kcurrent_module_name);

	// ----------------------------------------------------------------
	// Phase II - Query model metadata via RKNNKit
	// ----------------------------------------------------------------

	RKNNQuery::SdkVersion1st(ctx_);
	io_num_ = RKNNQuery::IoNum2nd(ctx_);
	input_attrs_ = RKNNQuery::InputAttrs3rd(ctx_, io_num_.n_input, perf_enabled_);
	output_attrs_ = RKNNQuery::OutputAttrs4th(ctx_, io_num_.n_output, perf_enabled_);

	// ----------------------------------------------------------------
	// Phase III - Allocate zero-copy buffers for all inputs and outputs
	//
	// Default: all buffers cacheable (matches legacy rknn_create_mem behaviour).
	// To experiment with non-cacheable for selected tensor indices, pass the
	// indices as the third argument to RKNNMemory::AllocAll, e.g.:
	//     RKNNMemory::AllocAll(ctx_, input_attrs_, {1, 2, 3, 4});  // RVM r1i~r4i
	// See pkb §5.4 for the (failed) A5 experiment that motivated this hook.
	// ----------------------------------------------------------------
	input_mems_ = RKNNMemory::AllocAll(ctx_, input_attrs_);
	output_mems_ = RKNNMemory::AllocAll(ctx_, output_attrs_);

	// ----------------------------------------------------------------
	// Phase IV - Bind memory to tensor descriptors
	// ----------------------------------------------------------------
	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
	}
	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
	}

	// ----------------------------------------------------------------
	// Phase V - Set NPU core mask
	// ----------------------------------------------------------------
	rknn_core_mask mask =
	    (core_mask_ < 0) ? RKNN_NPU_CORE_ALL : static_cast<rknn_core_mask>(core_mask_);
	auto retval = rknn_set_core_mask(ctx_, mask);
	if (retval == RKNN_SUCC) {
		logger.Info("Set core to " + helmsman::rknnkit::to_string(mask) + " Successfully",
		            kcurrent_module_name);
	} else {
		logger.Warning("Set core to " + helmsman::rknnkit::to_string(mask) + " Failed",
		               kcurrent_module_name);
	}

	logger.Info("Zero-copy buffers allocated and bound: " + std::to_string(io_num_.n_input) +
	                " inputs, " + std::to_string(io_num_.n_output) + " outputs.",
	            kcurrent_module_name);

	// ----------------------------------------------------------------
	// Diagnostic queries — via RKNNKit (all require Phase III buffers)
	// ----------------------------------------------------------------
	if (perf_enabled_) {
		RKNNQuery::LogMemSize7th(ctx_);
		RKNNQuery::LogCustomString8th(ctx_);
		RKNNQuery::LogNativeInputAttrs9th(ctx_, io_num_.n_input);
		RKNNQuery::LogNativeOutputAttrs10th(ctx_, io_num_.n_output);
		RKNNQuery::LogNhwcInputAttrs11th(ctx_, io_num_.n_input);
		RKNNQuery::LogNhwcOutputAttrs12th(ctx_, io_num_.n_output);
		RKNNQuery::LogInputDynamicRange14th(ctx_, io_num_.n_input);
		RKNNQuery::LogCurrentInputAttrs15th(ctx_, io_num_.n_input);
		RKNNQuery::LogCurrentOutputAttrs16th(ctx_, io_num_.n_output);
		RKNNQuery::LogCurrentNativeInputAttrs17th(ctx_, io_num_.n_input);
		RKNNQuery::LogCurrentNativeOutputAttrs18th(ctx_, io_num_.n_output);
	}
}

// ============================================================================
// Step 1 — Input Data Transfer (CPU → NPU)
//
// Convert float32 tensors into the model's native precision and
// write directly into RKNN zero-copy DMA buffers.
//
// Type conversion adapts to the model's compiled precision per tensor:
//   - INT8:  normalize [0,255]→[-1,1], quantize with scale/zp (image only)
//            For recurrent states: direct quantize (already in model range)
//   - FP16:  FLOAT32 → __fp16
//   - FP32:  direct memcpy
// ============================================================================
void InferenceEngineRKNNZeroCP::WriteInputBuffers1st(const std::vector<TensorData>& inputs) {
	if (inputs.size() != static_cast<std::size_t>(io_num_.n_input)) {
		throw std::runtime_error("Input count mismatch: model expects " +
		                         std::to_string(io_num_.n_input) + " inputs, got " +
		                         std::to_string(inputs.size()));
	}

	for (uint32_t i = 0; i < io_num_.n_input; ++i) {
		const TensorData& td = inputs[i];
		const rknn_tensor_attr& attr = input_attrs_[i];

		bool is_int8 = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8);
		bool is_fp16 = (attr.type == RKNN_TENSOR_FLOAT16);

		if (is_int8) {
			int8_t* dst = reinterpret_cast<int8_t*>(input_mems_[i]->virt_addr);
			float scale = attr.scale;
			float zp = static_cast<float>(attr.zp);

			if (i == 0) {
				// Image tensor: normalize [0,255] → [-1,1] then quantize
				for (size_t j = 0; j < td.data.size(); ++j) {
					float normalized = (td.data[j] - 127.5f) / 127.5f;
					float quantized = (normalized / scale) + zp;
					quantized = std::max(-128.0f, std::min(127.0f, quantized));
					dst[j] = static_cast<int8_t>(std::round(quantized));
				}
			} else {
				// Recurrent state: direct quantize (values already in model range)
				for (size_t j = 0; j < td.data.size(); ++j) {
					float quantized = (td.data[j] / scale) + zp;
					quantized = std::max(-128.0f, std::min(127.0f, quantized));
					dst[j] = static_cast<int8_t>(std::round(quantized));
				}
			}
		} else if (is_fp16) {
			__fp16* dst = reinterpret_cast<__fp16*>(input_mems_[i]->virt_addr);
			for (size_t j = 0; j < td.data.size(); ++j) {
				dst[j] = static_cast<__fp16>(td.data[j]);
			}
		} else {
			float* dst = reinterpret_cast<float*>(input_mems_[i]->virt_addr);
			std::memcpy(dst, td.data.data(), td.data.size() * sizeof(float));
		}
	}
}

// ============================================================================
// Step 2 — NPU Execution
//
// Dispatch the model graph to the NPU hardware and collect per-frame
// performance metrics (PERF_RUN, PERF_DETAIL).
// ============================================================================
void InferenceEngineRKNNZeroCP::ExecuteNpu2nd() {
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	// Per-frame: RKNN pure run time (us)
	RKNNQuery::PerfRun5th(ctx_);

	// Per-frame: per-layer op timing (only when COLLECT_PERF_MASK was accepted)
	if (perf_enabled_) {
		RKNNQuery::PerfDetail6th(ctx_);
	}
}

// ============================================================================
// Step 3 — Output Data Transfer (NPU → CPU)
//
// Read raw NPU output buffers and convert from model precision
// (INT8/FP16) back to float32 for the downstream MattingBackend.
// After this step, ownership of `outputs` transfers to the caller
// (InferenceEngine::Infer → MattingBackend::Postprocess).
// ============================================================================
void InferenceEngineRKNNZeroCP::ReadOutputBuffers3rd(const std::vector<TensorData>& inputs,
                                                     std::vector<TensorData>& outputs) {
	outputs.clear();
	outputs.reserve(io_num_.n_output);

	for (uint32_t i = 0; i < io_num_.n_output; ++i) {
		const rknn_tensor_attr& attr = output_attrs_[i];

		// §5.5: Backend only uses pha; fgr is never consumed.
		// Skip fgr D→H transfer to save ~8.36 MB per frame.
		if (std::string(attr.name) == "fgr") {
			static bool logged_skip = false;
			if (!logged_skip) {
				GetLogger().Info("§5.5: Skipped output '" + std::string(attr.name) + "' (" +
				                     std::to_string(attr.n_elems) + " elems)",
				                 kcurrent_module_name);
				logged_skip = true;
			}
			continue;
		}

		// Log output names on first frame for diagnostics
		static bool logged_names = false;
		if (!logged_names) {
			GetLogger().Info("§5.5: Reading output '" + std::string(attr.name) + "' (" +
			                     std::to_string(attr.n_elems) + " elems)",
			                 kcurrent_module_name);
		}

		bool is_int8_out = (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8);
		bool is_fp16_out = (attr.type == RKNN_TENSOR_FLOAT16);

		size_t element_count;
		if (is_int8_out)
			element_count = attr.size / sizeof(int8_t);
		else if (is_fp16_out)
			element_count = attr.size / sizeof(__fp16);
		else
			element_count = attr.size / sizeof(float);

		std::vector<float> out_data(element_count);

		if (is_int8_out) {
			int8_t* src = reinterpret_cast<int8_t*>(output_mems_[i]->virt_addr);
			float scale = attr.scale;
			float zp = static_cast<float>(attr.zp);
			for (size_t j = 0; j < element_count; ++j) {
				out_data[j] = (static_cast<float>(src[j]) - zp) * scale;
			}
		} else if (is_fp16_out) {
			__fp16* src = reinterpret_cast<__fp16*>(output_mems_[i]->virt_addr);
			for (size_t j = 0; j < element_count; ++j) {
				out_data[j] = static_cast<float>(src[j]);
			}
		} else {
			float* src = reinterpret_cast<float*>(output_mems_[i]->virt_addr);
			std::memcpy(out_data.data(), src, element_count * sizeof(float));
		}

		TensorData td;
		td.name = attr.name;
		td.data = std::move(out_data);

		td.shape.clear();
		for (uint32_t d = 0; d < attr.n_dims; ++d) {
			td.shape.push_back(attr.dims[d]);
		}

		// Propagate letterbox metadata from the primary input (src)
		td.orig_width = inputs[0].orig_width;
		td.orig_height = inputs[0].orig_height;
		td.pad_top = inputs[0].pad_top;
		td.pad_bottom = inputs[0].pad_bottom;
		td.pad_left = inputs[0].pad_left;
		td.pad_right = inputs[0].pad_right;

		outputs.push_back(std::move(td));

		if (!logged_names)
			logged_names = true;
	}
}

// ============================================================================
// Step 4 — Inference Profiling
//
// Log per-step timing: input conversion, NPU execution, output conversion,
// and binary dump. Also logs the primary input's precision type.
// ============================================================================
void InferenceEngineRKNNZeroCP::LogInferenceProfile4th(double cast_in_ms, double npu_run_ms,
                                                       double cast_out_ms, double dump_ms) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	bool is_int8_primary =
	    (input_attrs_[0].type == RKNN_TENSOR_INT8 || input_attrs_[0].type == RKNN_TENSOR_UINT8);
	bool is_fp16_primary = (input_attrs_[0].type == RKNN_TENSOR_FLOAT16);
	std::string precision_str = is_int8_primary ? "INT8" : (is_fp16_primary ? "FP16" : "FP32");

	logger.Info("   [Profiler] Input conversion (" + precision_str +
	                ") cost: " + std::to_string(cast_in_ms) + " ms.",
	            kcurrent_module_name);
	logger.Info("   [Profiler] Pure RKNN Run cost:   " + std::to_string(npu_run_ms) + " ms.",
	            kcurrent_module_name);
	logger.Info("   [Profiler] Output conversion cost: " + std::to_string(cast_out_ms) + " ms.",
	            kcurrent_module_name);

	if (dump_ms > 0.0) {
		logger.Info("   [Profiler] Binary Dump cost:     " + std::to_string(dump_ms) + " ms.",
		            kcurrent_module_name);
	} else {
		logger.Info("   [Profiler] Binary Dump:           disabled (no stats)",
		            kcurrent_module_name);
	}
}

// ============================================================================
// Inference — N inputs → M outputs (Zero-Copy)
//
// Orchestrates the 4-step inference pipeline:
//   1st: CPU → NPU input conversion
//   2nd: NPU execution
//   3rd: NPU → CPU output conversion
//   4th: profiling log
// ============================================================================
void InferenceEngineRKNNZeroCP::DoInfer(const std::vector<TensorData>& inputs,
                                        std::vector<TensorData>& outputs) {
	auto t0 = std::chrono::high_resolution_clock::now();

	// Step 1 - CPU → NPU: convert and write input buffers
	WriteInputBuffers1st(inputs);

	auto t1 = std::chrono::high_resolution_clock::now();

	// Step 2 - NPU execution
	ExecuteNpu2nd();

	auto t2 = std::chrono::high_resolution_clock::now();

	// Step 3 - NPU → CPU: read and convert output buffers
	ReadOutputBuffers3rd(inputs, outputs);

	auto t3 = std::chrono::high_resolution_clock::now();

	// Debug dump for pha output (name-based lookup)
	double dump_ms = 0.0;
	if (dump_enabled_ && !output_bin_path_.empty()) {
		for (const auto& td : outputs) {
			if (td.name == "pha") {
				helmsman::utils::FileUtils::GetInstance().dumpBinary(
				    td.data, output_bin_path_ + "cpp_08_inference-Output.bin");
				break;
			}
		}
		auto t4 = std::chrono::high_resolution_clock::now();
		dump_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
	}

	// Step 4 - Profiling
	LogInferenceProfile4th(std::chrono::duration<double, std::milli>(t1 - t0).count(),
	                       std::chrono::duration<double, std::milli>(t2 - t1).count(),
	                       std::chrono::duration<double, std::milli>(t3 - t2).count(), dump_ms);

	helmsman::utils::Logger::GetInstance().Info(
	    "infer() complete: " + std::to_string(io_num_.n_input) + " in / " +
	        std::to_string(io_num_.n_output) + " out",
	    kcurrent_module_name);
}

// ============================================================================
// SwapRecurrentStateBuffers — Zero-copy recurrent state optimization
//
// After NPU execution, the output DMA buffers for r1o~r4o already contain
// the new recurrent state data. Instead of copying D→H→D (read output →
// store in TensorData → write back to input DMA buffer), we swap the DMA
// buffer pointers so the output buffers become the input buffers for the
// next frame. This eliminates 3.20 MB/frame of redundant memory traffic.
//
// After the swap, rknn_set_io_mem() re-binds the buffers to the correct
// tensor descriptors so the NPU reads from the right memory on the next run.
// ============================================================================
bool InferenceEngineRKNNZeroCP::SwapRecurrentStateBuffers(std::size_t n_states,
                                                          std::size_t input_offset,
                                                          std::size_t output_offset) {
	if (input_offset + n_states > input_mems_.size() ||
	    output_offset + n_states > output_mems_.size()) {
		return false;
	}

	for (std::size_t i = 0; i < n_states; ++i) {
		std::swap(input_mems_[input_offset + i], output_mems_[output_offset + i]);
		rknn_set_io_mem(ctx_, input_mems_[input_offset + i], &input_attrs_[input_offset + i]);
		rknn_set_io_mem(ctx_, output_mems_[output_offset + i], &output_attrs_[output_offset + i]);
	}

	helmsman::utils::Logger::GetInstance().Info(
	    "SwapRecurrentStateBuffers: swapped " + std::to_string(n_states) +
	        " state DMA buffers (input[" + std::to_string(input_offset) + ".." +
	        std::to_string(input_offset + n_states - 1) + "] ↔ output[" +
	        std::to_string(output_offset) + ".." + std::to_string(output_offset + n_states - 1) +
	        "])",
	    kcurrent_module_name);

	return true;
}
