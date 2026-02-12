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
#include "pipeline/backend/backend.h"
#include "pipeline/frontend/frontend.h"
#include "pipeline/inference-engine/onnx/onnx.h"
#include "pipeline/inference-engine/rknn/rknn.h"

Pipeline& Pipeline::GetInstance() {
	static Pipeline instance;

	return instance;
}

Pipeline::Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline object constructed.");
}

Pipeline::~Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline cleaned up.");
}

void Pipeline::init(const std::string& image_path, const std::string& onnx_path,
                    const std::string& output_bin_path) {
	this->image_path_ = image_path;
	this->onnx_path_ = onnx_path;
	this->output_bin_path_ = output_bin_path;
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
	// InferenceEngineONNX engine;
	InferenceEngineRKNN engine;
	MattingBackend backend;

	// --------
	// 1st. Frontend: preprocess
	frontend.setOutputBinPath(output_bin_path_);
	auto input = frontend.preprocess(image_path_);

	// --------
	// 2nd. Inference Engine: load model and infer
	engine.setOutputBinPath(output_bin_path_);
	engine.load(onnx_path_);
	auto output_tensor = engine.infer(input);

	// --------
	// 3rd. Backend: postprocess
	backend.setOutputPath(output_bin_path_);
	auto result = backend.postprocess(output_tensor);

	return 0;
}

// ----

// int Pipeline::main_pipeline() {

// 	verify_parameters_necessary();
// 	auto& logger_ = arcforge::embedded::utils::Logger::GetInstance();
// 	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();
// 	// auto& math_utils_ = arcforge::utils::MathUtils::GetInstance();
// 	auto& runtime_ = arcforge::runtime::RuntimeONNX::GetInstance();

// 	try {
// 		// ----------------
// 		// Processing -- 1. create ONNX Runtime environment and session
// 		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "v1.3.3-inference-execution");

// 		Ort::Session session(env, onnx_path_.c_str(), runtime_.init_session_option());

// 		Ort::AllocatorWithDefaultOptions allocator;

// 		const char* input_name = session.GetInputName(0, allocator);
// 		const char* output_name = session.GetOutputName(0, allocator);

// 		std::vector<int64_t> input_shape =
// 		    session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

// 		logger_.Info("Input Name: " + std::string(input_name));
// 		logger_.Info("Output Name: " + std::string(output_name));

// 		logger_.Info("Input Shape: ");
// 		for (auto s : input_shape) {
// 			logger_.Info(std::to_string(s) + " ");
// 		}
// 		logger_.Info("\n");

// 		// ----------------
// 		// Processing -- 2. obtain std::vector<float> input tensor from preprocessing pipeline
// 		TensorData tensor_data;
// 		std::vector<float> input_tensor_values;
// 		tensor_data = processing_pipeline(image_path_, output_bin_path_, tensor_data);
// 		input_tensor_values = tensor_data.data;
// 		logger_.Info("Loaded input tensor size: " + std::to_string(input_tensor_values.size()));

// 		//---------------
// 		// Processing -- 3. process input tensor shape
// 		int64_t N = 1;
// 		int64_t C = 3;
// 		int64_t H = tensor_data.height;
// 		int64_t W = tensor_data.width;

// 		std::vector<int64_t> real_input_shape = {N, C, H, W};

// 		size_t input_tensor_size = static_cast<size_t>(N * C * H * W);

// 		logger_.Info("Input tensor size: " + std::to_string(input_tensor_size));
// 		logger_.Info("Input tensor actual size: " + std::to_string(input_tensor_values.size()));
// 		if (input_tensor_size != input_tensor_values.size()) {
// 			logger_.Error("❌ Size mismatch! expected " + std::to_string(input_tensor_size) +
// 			              " got " + std::to_string(input_tensor_values.size()));
// 			return 1;
// 		}

// 		//----------------
// 		// Processing -- 4. create input tensor object and run inference
// 		Ort::MemoryInfo memory_info =
// 		    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

// 		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
// 		    memory_info, input_tensor_values.data(), input_tensor_size, real_input_shape.data(),
// 		    real_input_shape.size());

// 		auto output_tensors =
// 		    session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

// 		// ----------------
// 		// Processing -- Echo output tensor
// 		float* output_data = output_tensors[0].GetTensorMutableData<float>();

// 		auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

// 		logger_.Info("Output Shape: ");
// 		for (auto s : output_shape) {
// 			logger_.Info(std::to_string(s) + " ");
// 		}
// 		logger_.Info("\n");

// 		size_t output_tensor_size = 1;
// 		for (auto s : output_shape) {
// 			output_tensor_size *= static_cast<size_t>(s);
// 		}

// 		// --------------------
// 		// Processing -- Dump output tensor to binary file
// 		std::vector<float> output_vector(output_data, output_data + output_tensor_size);

// 		file_utils_.dumpBinary(output_vector, output_bin_path_ + "cpp_08_inference-Output.bin");

// 		logger_.Info("✅ Inference done. Output dumped.");

// 	} catch (const Ort::Exception& e) {
// 		std::cerr << "Error: " << e.what() << std::endl;
// 		return 1;
// 	}

// 	return 0;
// }
