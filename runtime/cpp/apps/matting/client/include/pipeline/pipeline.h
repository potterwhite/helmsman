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

#include "CVKit/base/base.h"
#include "Runtime/onnx/onnx.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"
#include "pipeline/core/recurrent-state-manager.h"

// ---------------------------------------------------------------------------
// ModelType — explicit model type selection
//
// The caller (main-client) chooses which model type to run.
// InferenceEngine itself is model-agnostic (N→M tensor).
// Pipeline uses this enum to decide:
//   - Whether to initialize RecurrentStateManager
//   - How many benchmark iterations to run
//   - Which backend post-processing path to take
// ---------------------------------------------------------------------------
enum class ModelType {
	kMODNet,  // Single-frame matting (1 input → 1 output)
	kRVM,     // Video matting with recurrent state (5 inputs → 6 outputs)
};

class Pipeline {
   public:
	static Pipeline& GetInstance();
	void init(const std::string& image_path, const std::string& model_path,
	          const std::string& output_bin_path, const std::string& background_path = "",
	          ModelType model_type = ModelType::kMODNet);

	int run();

   private:
	Pipeline();
	~Pipeline();

	void verify_parameters_necessary();

	// MODNet path: single-frame inference with benchmarking
	int runMODNet();

	// RVM path: multi-frame inference with recurrent state management
	int runRVM();

   private:
	std::string image_path_;
	std::string model_path_;
	std::string output_bin_path_;
	std::string background_path_;
	ModelType   model_type_ = ModelType::kMODNet;

	RecurrentStateManager state_mgr_;
};
