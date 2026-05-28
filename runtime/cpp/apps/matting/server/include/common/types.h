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

// =============================================================================
// types.h — Shared type definitions for the matting server.
//
// Centralises enums, structs, and type aliases used across multiple modules.
// All pipeline stages, modes, and the top-level entry point include this file
// for shared types. Keep this file free of logic — definitions only.
//
// =============================================================================

#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

enum class ModelType {
	kMODNet,
	kRVM,
};

enum class OutputMode {
	kMp4,
	kDrm,
};

// ---------------------------------------------------------------------------
// Application-level configuration
//
// Populated by main-server from CLI arguments, then passed by const reference
// through Pipeline → Mode → helper functions. This is the single source of
// truth for all runtime configuration; do not stash individual fields as
// member variables — access them via the config reference instead.
// ---------------------------------------------------------------------------

struct AppConfig {
	ModelType model_type = ModelType::kMODNet;
	OutputMode output_mode = OutputMode::kMp4;
	bool timing_enabled = true;
	bool use_hardware_decoder = false;
	bool use_prefetch_thread = true;
	bool is_video = false;
	int rknn_core_mask = -1;  // -1 = default (engine decides); 0 = CORE_ALL
	bool rknn_perf_enabled = false;  // true = collect per-layer NPU profiling data
	bool dump_enabled = false;       // true = enable binary dump for debugging
	bool diag_enabled = false;       // true = enable diagnostic logging (internal state inspection)
	std::string input_path;
	std::string model_path;
	std::string output_bin_path;
	std::string background_path;
};

// ---------------------------------------------------------------------------
// Pipeline data structures
// ---------------------------------------------------------------------------

typedef struct {
	// --- Tensor identity ---
	std::string name;             // tensor name (e.g. "src", "r1i", "pha", "r1o")
	                              // empty string is valid for legacy single-tensor paths

	// --- Tensor data ---
	std::vector<float>     data;
	std::vector<int64_t>   shape;

	// --- Letterbox metadata (filled by Preprocessor, consumed by MattingBackend) ---
	int orig_width  = 0;
	int orig_height = 0;
	int pad_top     = 0;
	int pad_bottom  = 0;
	int pad_left    = 0;
	int pad_right   = 0;
} TensorData;
