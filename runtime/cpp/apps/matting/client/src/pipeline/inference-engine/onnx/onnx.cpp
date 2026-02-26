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
#include "common-define.h"

// InferenceEngineONNX& InferenceEngineONNX::GetInstance() {
// 	static InferenceEngineONNX instance;

// 	return instance;
// }

InferenceEngineONNX::InferenceEngineONNX()
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx-inference-engine") {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineONNX object constructed.",
	                                                      kcurrent_module_name);
}

InferenceEngineONNX::~InferenceEngineONNX() {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineONNX cleaned up.");
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

	logger.Info("Input Name: " + input_name_, kcurrent_module_name);
	logger.Info("Output Name: " + output_name_, kcurrent_module_name);

	logger.Info("Input Shape: ");
	for (auto s : input_shape) {
		logger.Info(std::to_string(s) + " ", kcurrent_module_name);
	}
	logger.Info("\n");
}

TensorData InferenceEngineONNX::infer(const TensorData& input) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// ------------------------------------------------------------------------
	// 架构适配：ONNX 引擎方言翻译 (NHWC -> NCHW & Normalization)
	// Frontend 传来的是: NHWC (例如 1, 512, 896, 3), 且数据为 0~255
	// ONNX 模型需要的是: NCHW (例如 1, 3, 512, 896), 且数据通常需要归一化
	// ------------------------------------------------------------------------
	int64_t N = input.shape[0];  // 1
	int64_t H = input.shape[1];  // 512
	int64_t W = input.shape[2];  // 896
	int64_t C = input.shape[3];  // 3

	// 构造 ONNX 期望的 NCHW shape (ONNX API 强制要求 int64_t)
	std::vector<int64_t> onnx_input_shape = {N, C, H, W};

	// 转换为 size_t 以满足 C++ 严格的符号检查标准，用于内存分配和遍历
	size_t size_N = static_cast<size_t>(N);
	size_t size_H = static_cast<size_t>(H);
	size_t size_W = static_cast<size_t>(W);
	size_t size_C = static_cast<size_t>(C);

	size_t input_tensor_size = size_N * size_C * size_H * size_W;

	if (input_tensor_size != input.data.size()) {
		logger.Error("❌ Size mismatch!", kcurrent_module_name);
		throw std::runtime_error("Input size mismatch");
	}

	// 申请用于存放 NCHW 格式的内存
	std::vector<float> nchw_data(input_tensor_size);

	// 执行 NHWC 到 NCHW 的内存重排，并恢复被 Frontend 取消的归一化
	// 使用 size_t 遍历，彻底杜绝 sign-conversion 编译报错
	for (size_t c = 0; c < size_C; ++c) {
		for (size_t h = 0; h < size_H; ++h) {
			for (size_t w = 0; w < size_W; ++w) {
				size_t nhwc_idx = h * size_W * size_C + w * size_C + c;
				size_t nchw_idx = c * size_H * size_W + h * size_W + w;

				// 核心：把 0~255 的数据，除以 255.0f 归一化到 0~1
				// 并放置到 NCHW 正确的内存偏移位置上
				nchw_data[nchw_idx] = input.data[nhwc_idx] / 255.0f;
			}
		}
	}

	logger.Info("Data translated from NHWC(0~255) to NCHW(0~1) for ONNX.", kcurrent_module_name);

	//----------------
	// Processing -- 4. create input tensor object and run inference
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// 使用刚刚重排并归一化好的 nchw_data 喂给 ORT
	Ort::Value input_tensor =
	    Ort::Value::CreateTensor<float>(memory_info, nchw_data.data(), input_tensor_size,
	                                    onnx_input_shape.data(), onnx_input_shape.size());

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
		logger.Info(std::to_string(s) + " ", kcurrent_module_name);
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