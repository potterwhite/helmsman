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

#include "pipeline/inference-engine/onnx/onnx.h"

// InferenceEngineONNX& InferenceEngineONNX::GetInstance() {
// 	static InferenceEngineONNX instance;

// 	return instance;
// }

InferenceEngineONNX::InferenceEngineONNX()
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx-inference-engine") {
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineONNX object constructed.");
}

InferenceEngineONNX::~InferenceEngineONNX() {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineONNX cleaned up.");
}

void InferenceEngineONNX::setOutputBinPath(const std::string& path) {
	output_bin_path_ = path;
}

void InferenceEngineONNX::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& runtime = arcforge::runtime::RuntimeONNX::GetInstance();

	// ----------------
	// Processing -- 1. create ONNX Runtime environment and session
	//

	session_ =
	    std::make_unique<Ort::Session>(env_, model_path.c_str(), runtime.init_session_option());

	Ort::AllocatorWithDefaultOptions allocator;

	// these two API maybe deprecated, need to check later
	// char* input_name = (session_->GetInputNameAllocated(0, allocator)).get();
	// char* output_name = (session_->GetOutputNameAllocated(0, allocator)).get();
	input_name_ = (session_->GetInputNameAllocated(0, allocator)).get();
	output_name_ = (session_->GetOutputNameAllocated(0, allocator)).get();

	// input_name_ = std::string(input_name);
	// output_name_ = std::string(output_name);

	std::vector<int64_t> input_shape =
	    session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	logger.Info("Input Name: " + input_name_);
	logger.Info("Output Name: " + output_name_);

	logger.Info("Input Shape: ");
	for (auto s : input_shape) {
		logger.Info(std::to_string(s) + " ");
	}
	logger.Info("\n");
}

TensorData InferenceEngineONNX::infer(const TensorData& input) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	//---------------
	// Processing -- 3. process input tensor shape
	// int64_t N = 1;
	// int64_t C = 3;
	// int64_t H = input.height;
	// int64_t W = input.width;

	// std::vector<int64_t> real_input_shape = {N, C, H, W};
	// size_t input_tensor_size = static_cast<size_t>(N * C * H * W);
	std::vector<int64_t> real_input_shape = input.shape;

	size_t input_tensor_size = 1;
	for (auto s : real_input_shape)
		input_tensor_size *= static_cast<size_t>(s);

	logger.Info("Input tensor size: " + std::to_string(input_tensor_size));
	logger.Info("Input tensor actual size: " + std::to_string(input.data.size()));
	if (input_tensor_size != input.data.size()) {
		logger.Error("❌ Size mismatch! expected " + std::to_string(input_tensor_size) + " got " +
		             std::to_string(input.data.size()));

		throw std::runtime_error("Input size mismatch");
	}

	//----------------
	// Processing -- 4. create input tensor object and run inference
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// make a copy of input data to ensure the data pointer validity during inference
	std::vector<float> input_copy = input.data;
	Ort::Value input_tensor =
	    Ort::Value::CreateTensor<float>(memory_info, input_copy.data(), input_tensor_size,
	                                    real_input_shape.data(), real_input_shape.size());

	const char* input_name = input_name_.c_str();
	const char* output_name = output_name_.c_str();
	auto output_tensors =
	    session_->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

	// ----------------
	// Processing -- Echo output tensor
	float* output_data = output_tensors[0].GetTensorMutableData<float>();

	auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

	logger.Info("Output Shape: ");
	for (auto s : output_shape) {
		logger.Info(std::to_string(s) + " ");
	}
	logger.Info("\n");

	size_t output_tensor_size = 1;
	for (auto s : output_shape) {
		output_tensor_size *= static_cast<size_t>(s);
	}

	// --------------------
	// Processing -- Dump output tensor to binary file
	std::vector<float> output_vector(output_data, output_data + output_tensor_size);

	file_utils_.dumpBinary(output_vector, output_bin_path_ + "cpp_08_inference-Output.bin");

	// --------------------
	// Assemble output TensorData

	TensorData output;
	output.data.assign(output_data, output_data + output_tensor_size);
	output.shape = output_shape;
	// output.height = output_shape[2];
	// output.width = output_shape[3];

	return output;
}