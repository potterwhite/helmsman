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

void Pipeline::init(const std::string& image_path, const std::string& onnx_path,
                    const std::string& output_bin_path, const std::string& background_path) {
	this->image_path_ = image_path;
	this->onnx_path_ = onnx_path;
	this->output_bin_path_ = output_bin_path;
	this->background_path_ = background_path;
}

void Pipeline::verify_parameters_necessary() {
	if (image_path_.empty()) {
		throw std::invalid_argument("Image path is empty.");
	}
	if (onnx_path_.empty()) {
		throw std::invalid_argument("ONNX model path is empty.");
	}
	if (output_bin_path_.empty()) {
		throw std::invalid_argument("Output binary path is empty.");
	}
}

int Pipeline::run() {
	verify_parameters_necessary();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	// InferenceEngineRKNN engine;
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// --------
	// 1st. Load model first to get input dimensions
	engine.setOutputBinPath(output_bin_path_);
	engine.load(onnx_path_);

#ifdef ENABLE_RKNN_BACKEND
	// Get model input size (assumes square input: height == width)
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width = engine.getInputWidth();
#else
	size_t model_input_height = 512;  // Default input height for ONNX model
	size_t model_input_width = 512;   // Default input width for ONNX model
#endif

	// --------
	// 2nd. Frontend: preprocess with model's input size
	frontend.setOutputBinPath(output_bin_path_);
	auto input = frontend.preprocess(image_path_, model_input_width, model_input_height);

	// --------
	// 3rd. Inference Engine: run inference multiple times for benchmarking
	TensorData output_tensor;
	for (int i = 0; i < 10; ++i) {
		auto start_time = std::chrono::high_resolution_clock::now();
		output_tensor = engine.infer(input);
		auto end_time = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> infer_duration = end_time - start_time;
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "🚀 [Performance Benchmark " + std::to_string(i + 1) +
		        "] Inference Engine [infer()] cost: " + std::to_string(infer_duration.count()) +
		        " ms.",
		    kcurrent_module_name);
	}

	// --------
	// 3rd. Backend: postprocess
	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(4, 1e-4, 0.4f));
	auto result = backend.postprocess(output_tensor);

	return 0;
}
