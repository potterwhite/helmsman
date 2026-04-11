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

#include "pipeline/pipeline.h"
#include <chrono>
#include "common-define.h"
#include "pipeline/backend/backend.h"
#include "pipeline/backend/post-processor/guided-filter-post-processor.h"
#include "pipeline/frontend/frontend.h"

#ifdef ENABLE_RKNN_BACKEND
#include "pipeline/inference-engine/rknn/rknn-non-zero-copy.h"
#include "pipeline/inference-engine/rknn/rknn-zero-copy.h"
#else
#include "pipeline/inference-engine/onnx/onnx.h"
#endif

Pipeline& Pipeline::GetInstance() {
	static Pipeline instance;
	return instance;
}

Pipeline::Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline object constructed.",
	                                                      kcurrent_module_name);
}

Pipeline::~Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline cleaned up.",
	                                                      kcurrent_module_name);
}

void Pipeline::init(const std::string& image_path, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
	this->image_path_      = image_path;
	this->model_path_      = model_path;
	this->output_bin_path_ = output_bin_path;
	this->background_path_ = background_path;
	this->model_type_      = model_type;
}

void Pipeline::verify_parameters_necessary() {
	if (image_path_.empty()) {
		throw std::invalid_argument("Image path is empty.");
	}
	if (model_path_.empty()) {
		throw std::invalid_argument("Model path is empty.");
	}
	if (output_bin_path_.empty()) {
		throw std::invalid_argument("Output binary path is empty.");
	}
}

int Pipeline::run() {
	verify_parameters_necessary();

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	switch (model_type_) {
		case ModelType::kMODNet:
			logger.Info("Pipeline: running MODNet path (single-frame)", kcurrent_module_name);
			return runMODNet();
		case ModelType::kRVM:
			logger.Info("Pipeline: running RVM path (recurrent multi-frame)", kcurrent_module_name);
			return runRVM();
		default:
			throw std::runtime_error("Unknown model type");
	}
}

// ============================================================================
// MODNet path — single-frame, 10x benchmark loop
//
// Fully backward-compatible with the original pipeline.
// inputs = { src }  (1 tensor)
// outputs = { pha } (1 tensor)
// ============================================================================
int Pipeline::runMODNet() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// 1. Load model to get input dimensions
	engine.setOutputBinPath(output_bin_path_);
	engine.load(model_path_);

#ifdef ENABLE_RKNN_BACKEND
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width  = engine.getInputWidth();
#else
	size_t model_input_height = 512;
	size_t model_input_width  = 512;
#endif

	// 2. Frontend: preprocess image
	frontend.setOutputBinPath(output_bin_path_);
	TensorData src = frontend.preprocess(image_path_, model_input_width, model_input_height);

	// 3. Inference: wrap single tensor into vector, benchmark 10 iterations
	std::vector<TensorData> inputs  = { src };
	std::vector<TensorData> outputs;

	for (int i = 0; i < 10; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		engine.infer(inputs, outputs);
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = end - start;
		logger.Info(
		    "[Performance Benchmark " + std::to_string(i + 1) +
		        "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) + " ms.",
		    kcurrent_module_name);
	}

	// 4. Backend: postprocess
	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));
	backend.postprocess(outputs);

	return 0;
}

// ============================================================================
// RVM path — multi-frame with recurrent state management
//
// inputs  = { src, r1i, r2i, r3i, r4i }  (5 tensors)
// outputs = { fgr, pha, r1o, r2o, r3o, r4o }  (6 tensors)
//
// Currently runs N frames using the same image (single-image loop test).
// Will be extended to video frame input in Phase-5.
// ============================================================================
int Pipeline::runRVM() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// 1. Load model
	engine.setOutputBinPath(output_bin_path_);
	engine.load(model_path_);

#ifdef ENABLE_RKNN_BACKEND
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width  = engine.getInputWidth();
#else
	// RVM ONNX models have different default sizes; for now use 288x512
	size_t model_input_height = 288;
	size_t model_input_width  = 512;
#endif

	// 2. Initialize recurrent state manager with RVM state shapes.
	//    RVM MobileNetV3 @ 288x512 state shapes (from Phase-2 verification):
	//      r1: [1, 16, 144, 256]  (1/2 resolution)
	//      r2: [1, 20,  72, 128]  (1/4 resolution)
	//      r3: [1, 40,  36,  64]  (1/8 resolution)
	//      r4: [1, 64,  18,  32]  (1/16 resolution)
	//
	//    Note: These shapes are specific to 288x512 input.
	//    TODO: Query from model or make configurable.
	state_mgr_.init(
	    {
	        {1, 16, static_cast<int64_t>(model_input_height / 2),
	                static_cast<int64_t>(model_input_width / 2)},
	        {1, 20, static_cast<int64_t>(model_input_height / 4),
	                static_cast<int64_t>(model_input_width / 4)},
	        {1, 40, static_cast<int64_t>(model_input_height / 8),
	                static_cast<int64_t>(model_input_width / 8)},
	        {1, 64, static_cast<int64_t>(model_input_height / 16),
	                static_cast<int64_t>(model_input_width / 16)},
	    },
	    {"r1i", "r2i", "r3i", "r4i"}
	);

	// 3. Frontend: preprocess image (same image for all frames in test mode)
	frontend.setOutputBinPath(output_bin_path_);
	TensorData src = frontend.preprocess(image_path_, model_input_width, model_input_height);

	// 4. Multi-frame inference loop (5 frames using same image for integration test)
	constexpr int kNumTestFrames = 5;

	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));

	for (int frame = 0; frame < kNumTestFrames; ++frame) {
		logger.Info("=== RVM Frame " + std::to_string(frame + 1) + "/" +
		            std::to_string(kNumTestFrames) + " ===", kcurrent_module_name);

		// Build input vector: src + r1i~r4i
		std::vector<TensorData> inputs = { src };
		state_mgr_.inject(inputs);  // appends r1i, r2i, r3i, r4i

		// Run inference
		std::vector<TensorData> outputs;
		auto start = std::chrono::high_resolution_clock::now();
		engine.infer(inputs, outputs);
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = end - start;
		logger.Info(
		    "[RVM Frame " + std::to_string(frame + 1) +
		        "] infer() cost: " + std::to_string(dur.count()) + " ms.",
		    kcurrent_module_name);

		// Update recurrent states: r1o~r4o → r1i~r4i for next frame
		state_mgr_.update(outputs);  // outputs[2..5] = r1o~r4o

		// Post-process (only last frame saves result, or all frames for debugging)
		if (frame == kNumTestFrames - 1) {
			backend.postprocess(outputs);
		}
	}

	return 0;
}
