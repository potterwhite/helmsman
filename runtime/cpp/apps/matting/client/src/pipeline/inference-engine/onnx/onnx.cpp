// ============================================================================
// ONNX Inference Engine
//
// Processing Flow:
//
//   Step 1  - Create ONNX Runtime session
//   Step 2  - Translate data layout (NHWC → NCHW)
//   Step 3  - Restore normalization (0~255 → 0~1)
//   Step 4  - Create ONNX input tensor
//   Step 5  - Run inference
//   Step 6  - Extract output tensor
//   Step 7  - Assemble TensorData structure
//
// Important:
//   • Frontend provides image in NHWC format (0~255 range)
//   • ONNX model expects NCHW format (0~1 normalized)
//   • This engine performs layout translation + normalization recovery
// ============================================================================

#include "pipeline/inference-engine/onnx/onnx.h"
#include "common-define.h"

// InferenceEngineONNX& InferenceEngineONNX::GetInstance() {
// 	static InferenceEngineONNX instance;
// 	return instance;
// }

InferenceEngineONNX::InferenceEngineONNX()
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx-inference-engine") {

	arcforge::embedded::utils::Logger::GetInstance()
	    .Info("InferenceEngineONNX object constructed. (CPU Mode)",
	          kcurrent_module_name);
}

InferenceEngineONNX::~InferenceEngineONNX() {

	arcforge::embedded::utils::Logger::GetInstance()
	    .Info("InferenceEngineONNX cleaned up.");
}

// ============================================================================
// Step 1 - Load ONNX Model and Create Session
// ============================================================================
void InferenceEngineONNX::load(const std::string& model_path) {

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& runtime = arcforge::runtime::RuntimeONNX::GetInstance();

	// Create ONNX Runtime session
	session_ =
	    std::make_unique<Ort::Session>(
	        env_,
	        model_path.c_str(),
	        runtime.init_session_option());

	Ort::AllocatorWithDefaultOptions allocator;

	// Retrieve input and output tensor names
	// (These APIs may be deprecated in future ONNX versions)
	input_name_ =
	    (session_->GetInputNameAllocated(0, allocator)).get();

	output_name_ =
	    (session_->GetOutputNameAllocated(0, allocator)).get();

	// Query input tensor shape
	std::vector<int64_t> input_shape =
	    session_->GetInputTypeInfo(0)
	        .GetTensorTypeAndShapeInfo()
	        .GetShape();

	logger.Info("Input Name: " + input_name_, kcurrent_module_name);
	logger.Info("Output Name: " + output_name_, kcurrent_module_name);

	logger.Info("Input Shape: ");
	for (auto s : input_shape) {
		logger.Info(std::to_string(s) + " ", kcurrent_module_name);
	}
	logger.Info("\n");
}

// ============================================================================
// Step 2~7 - Run Inference
// ============================================================================
TensorData InferenceEngineONNX::infer(const TensorData& input) {

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// ------------------------------------------------------------------------
	// Step 2 - Architecture Adaptation (NHWC → NCHW)
	//
	// Frontend provides:
	//   Layout: NHWC
	//   Example shape: [1, 512, 896, 3]
	//   Data range: 0~255
	//
	// ONNX model expects:
	//   Layout: NCHW
	//   Example shape: [1, 3, 512, 896]
	//   Data range: 0~1 (normalized)
	// ------------------------------------------------------------------------

	int64_t N = input.shape[0];  // Batch size
	int64_t H = input.shape[1];  // Height
	int64_t W = input.shape[2];  // Width
	int64_t C = input.shape[3];  // Channels

	// ONNX requires int64_t for shape definition
	std::vector<int64_t> onnx_input_shape = {N, C, H, W};

	// Convert to size_t for safe memory operations
	size_t size_N = static_cast<size_t>(N);
	size_t size_H = static_cast<size_t>(H);
	size_t size_W = static_cast<size_t>(W);
	size_t size_C = static_cast<size_t>(C);

	size_t input_tensor_size =
	    size_N * size_C * size_H * size_W;

	// Validate total element count
	if (input_tensor_size != input.data.size()) {
		logger.Error("Size mismatch!", kcurrent_module_name);
		throw std::runtime_error("Input size mismatch");
	}

	// Allocate memory for NCHW data
	std::vector<float> nchw_data(input_tensor_size);

	// ------------------------------------------------------------------------
	// Step 3 - Memory Reordering + Normalization Recovery
	//
	// Convert:
	//   NHWC memory layout → NCHW memory layout
	//
	// Restore normalization:
	//   Divide pixel values by 255.0f
	//   Convert 0~255 → 0~1
	// ------------------------------------------------------------------------

	for (size_t c = 0; c < size_C; ++c) {
		for (size_t h = 0; h < size_H; ++h) {
			for (size_t w = 0; w < size_W; ++w) {

				size_t nhwc_idx =
				    h * size_W * size_C +
				    w * size_C +
				    c;

				size_t nchw_idx =
				    c * size_H * size_W +
				    h * size_W +
				    w;

				nchw_data[nchw_idx] =
				    input.data[nhwc_idx] / 255.0f;
			}
		}
	}

	logger.Info("Data translated from NHWC(0~255) to NCHW(0~1) for ONNX.",
	            kcurrent_module_name);

	// ------------------------------------------------------------------------
	// Step 4 - Create ONNX Input Tensor
	// ------------------------------------------------------------------------

	Ort::MemoryInfo memory_info =
	    Ort::MemoryInfo::CreateCpu(
	        OrtArenaAllocator,
	        OrtMemTypeDefault);

	Ort::Value input_tensor =
	    Ort::Value::CreateTensor<float>(
	        memory_info,
	        nchw_data.data(),
	        input_tensor_size,
	        onnx_input_shape.data(),
	        onnx_input_shape.size());

	// ------------------------------------------------------------------------
	// Step 5 - Execute Inference
	// ------------------------------------------------------------------------

	const char* input_name = input_name_.c_str();
	const char* output_name = output_name_.c_str();

	auto output_tensors =
	    session_->Run(Ort::RunOptions{nullptr},
	                  &input_name,
	                  &input_tensor,
	                  1,
	                  &output_name,
	                  1);

	// ------------------------------------------------------------------------
	// Step 6 - Extract Output Tensor
	// ------------------------------------------------------------------------

	float* output_data =
	    output_tensors[0].GetTensorMutableData<float>();

	auto output_shape =
	    output_tensors[0]
	        .GetTensorTypeAndShapeInfo()
	        .GetShape();

	logger.Info("Output Shape: ");
	for (auto s : output_shape) {
		logger.Info(std::to_string(s) + " ",
		            kcurrent_module_name);
	}
	logger.Info("\n");

	size_t output_tensor_size = 1;
	for (auto s : output_shape) {
		output_tensor_size *= static_cast<size_t>(s);
	}

	// ------------------------------------------------------------------------
	// Step 7 - Dump Output and Construct TensorData
	// ------------------------------------------------------------------------

	std::vector<float> output_vector(
	    output_data,
	    output_data + output_tensor_size);

	file_utils_.dumpBinary(
	    output_vector,
	    output_bin_path_ + "cpp_08_inference-Output.bin");

	TensorData output;
	output.data.assign(
	    output_data,
	    output_data + output_tensor_size);

	output.shape = output_shape;

	return output;
}