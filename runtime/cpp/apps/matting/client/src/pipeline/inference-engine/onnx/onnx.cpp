
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
	input_name_ = static_cast<std::string>(session_->GetInputName(0, allocator));
	output_name_ = static_cast<std::string>(session_->GetOutputName(0, allocator));

	std::vector<int64_t> input_shape =
	    session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	logger.Info("Input Name: " + input_name_);
	logger.Info("Output Name: " + output_name_);

	logger.Info("Input Shape: ");
	for (auto s : input_shape) {
		logger.Info(std::to_string(s) + " ");
	}
	logger.Info("\n");

	allocator.Free(input_name_.data());
	allocator.Free(output_name_.data());
}

TensorData InferenceEngineONNX::infer(const TensorData& input) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	//---------------
	// Processing -- 3. process input tensor shape
	int64_t N = 1;
	int64_t C = 3;
	int64_t H = input.height;
	int64_t W = input.width;

	std::vector<int64_t> real_input_shape = {N, C, H, W};

	size_t input_tensor_size = static_cast<size_t>(N * C * H * W);

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
	// Assemble output TensorData

	TensorData output;
	output.data.assign(output_data, output_data + output_tensor_size);
	output.height = output_shape[2];
	output.width = output_shape[3];

	return output;
}