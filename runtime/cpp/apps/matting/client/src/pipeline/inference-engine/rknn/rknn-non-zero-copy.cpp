#include "pipeline/inference-engine/rknn/rknn-non-zero-copy.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "Runtime/rknn.h/rknn.h"
#include "common-define.h"

// ============================================================================
// Constructor
// Purpose:
//   Create RKNN inference engine object (Non-Zero-Copy mode)
// ============================================================================
InferenceEngineRKNN::InferenceEngineRKNN() {
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNN object constructed. (Non-Zero-Copy Mode)", kcurrent_module_name);
}

// ============================================================================
// Destructor
// Purpose:
//   Automatically release resources when object is destroyed
// ============================================================================
InferenceEngineRKNN::~InferenceEngineRKNN() {
	release();
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineRKNN cleaned up.",
	                                                      kcurrent_module_name);
}

// ============================================================================
// Step 0: Release RKNN context
// Purpose:
//   Destroy RKNN runtime context if it exists
// ============================================================================
void InferenceEngineRKNN::release() {
	if (ctx_ > 0) {
		rknn_destroy(ctx_);
		ctx_ = 0;
	}
}

// ============================================================================
// Load Model Procedure
// This function performs 3 major steps:
//
//   Step 1 - Read RKNN model file into memory
//   Step 2 - Initialize RKNN runtime context
//   Step 3 - Query model input/output information
// ============================================================================
void InferenceEngineRKNN::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// ------------------------------------------------------------------------
	// Step 1: Read RKNN model file into memory buffer
	//
	// 1.1 Open model file (binary mode)
	// 1.2 Get file size
	// 1.3 Allocate memory buffer
	// 1.4 Read entire model into memory
	// ------------------------------------------------------------------------
	std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
	if (!ifs.is_open()) {
		logger.Error("❌ Failed to open RKNN model file: " + model_path, kcurrent_module_name);
		throw std::runtime_error("Failed to open RKNN model");
	}

	ifs.seekg(0, std::ios::end);
	size_t model_size = static_cast<size_t>(ifs.tellg());
	ifs.seekg(0, std::ios::beg);

	std::vector<unsigned char> model_data(model_size);
	ifs.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(model_size));
	ifs.close();

	logger.Info("RKNN model loaded into memory, size: " + std::to_string(model_size) + " bytes.",
	            kcurrent_module_name);

	// ------------------------------------------------------------------------
	// Step 2: Initialize RKNN runtime context
	//
	// 2.1 Pass model buffer into rknn_init()
	// 2.2 Create runtime context (ctx_)
	// ------------------------------------------------------------------------
	int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_init fail! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_init failed");
	}

	// ------------------------------------------------------------------------
	// Step 3: Query model metadata (I/O information)
	//
	// 3.1 Query number of inputs and outputs
	// 3.2 Query input tensor attributes
	// 3.3 Query output tensor attributes
	// ------------------------------------------------------------------------
	ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
	if (ret < 0) {
		throw std::runtime_error("rknn_query IN_OUT_NUM failed");
	}
	logger.Info("Model Input Num: " + std::to_string(io_num_.n_input) +
	                ", Output Num: " + std::to_string(io_num_.n_output),
	            kcurrent_module_name);

	// ----- Step 3.2: Query each input tensor attribute -----
	input_attrs_.clear();
	for (uint32_t i = 0; i < io_num_.n_input; i++) {
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
		input_attrs_.push_back(attr);

		logger.Info("Input attr: " + arcforge::runtime::to_string(attr));
	}

	// ----- Step 3.3: Query each output tensor attribute -----
	output_attrs_.clear();
	for (uint32_t i = 0; i < io_num_.n_output; i++) {
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
		output_attrs_.push_back(attr);

		logger.Info("Output attr: " + arcforge::runtime::to_string(attr));
	}
}

// ============================================================================
// Inference Procedure
// This function performs 5 major steps:
//
//   Step 1 - Prepare and set input tensor
//   Step 2 - Run inference
//   Step 3 - Retrieve output tensor
//   Step 4 - Convert output into TensorData structure
//   Step 5 - Release output memory
// ============================================================================
TensorData InferenceEngineRKNN::infer(const TensorData& input) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// ------------------------------------------------------------------------
	// Pre-check: Ensure model is loaded
	// ------------------------------------------------------------------------
	if (ctx_ == 0) {
		throw std::runtime_error("RKNN Context is null, please load model first.");
	}

	// ------------------------------------------------------------------------
	// Step 1: Prepare and set input tensor
	//
	// 1.1 Allocate rknn_input structure
	// 1.2 Configure tensor index
	// 1.3 Set input data type to FLOAT32
	// 1.4 Set data layout format (NCHW or NHWC)
	// 1.5 Provide buffer pointer and size
	// 1.6 Disable pass-through (Non-Zero-Copy mode)
	// ------------------------------------------------------------------------
	std::vector<rknn_input> inputs(io_num_.n_input);
	memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));

	inputs[0].index = 0;

	// Tell RKNN that we provide FLOAT32 input.
	// RKNN driver will internally convert to model-required format (INT8/FP16).
	inputs[0].type = RKNN_TENSOR_FLOAT32;

	// Must match model's expected layout format.
	inputs[0].fmt = input_attrs_[0].fmt;

	inputs[0].size = static_cast<uint32_t>(input.data.size() * sizeof(float));
	inputs[0].buf = const_cast<void*>(reinterpret_cast<const void*>(input.data.data()));

	// Non-Zero-Copy mode: let RKNN handle data conversion internally.
	inputs[0].pass_through = 0;

	int ret = rknn_inputs_set(ctx_, io_num_.n_input, inputs.data());
	if (ret < 0) {
		logger.Error("❌ rknn_inputs_set failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_inputs_set failed");
	}

	// ------------------------------------------------------------------------
	// Step 2: Execute inference
	// ------------------------------------------------------------------------
	ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_run failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_run failed");
	}

	// ------------------------------------------------------------------------
	// Step 3: Retrieve output tensor
	//
	// 3.1 Configure output request
	// 3.2 Ask RKNN to convert output to FLOAT32
	// 3.3 Let RKNN allocate memory (Non-Zero-Copy mode)
	// ------------------------------------------------------------------------
	std::vector<rknn_output> outputs(io_num_.n_output);
	memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));

	outputs[0].want_float = 1;   // Request dequantized FLOAT32 output
	outputs[0].is_prealloc = 0;  // Let RKNN allocate output buffer

	ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_outputs_get failed! ret=" + std::to_string(ret),
		             kcurrent_module_name);
		throw std::runtime_error("rknn_outputs_get failed");
	}

	// ------------------------------------------------------------------------
	// Step 4: Convert RKNN output into TensorData structure
	//
	// 4.1 Copy output data into std::vector<float>
	// 4.2 Copy output shape information
	// ------------------------------------------------------------------------
	TensorData output_tensor;
	float* out_data_ptr = reinterpret_cast<float*>(outputs[0].buf);
	size_t out_elements_count = outputs[0].size / sizeof(float);

	output_tensor.data.assign(out_data_ptr, out_data_ptr + out_elements_count);

	logger.Info("Output Shape: ", kcurrent_module_name);
	for (uint32_t i = 0; i < output_attrs_[0].n_dims; i++) {
		output_tensor.shape.push_back(output_attrs_[0].dims[i]);
		logger.Info(std::to_string(output_attrs_[0].dims[i]) + " ", kcurrent_module_name);
	}
	logger.Info("\n", kcurrent_module_name);

	// ------------------------------------------------------------------------
	// Step 4.3: Dump output tensor to binary file (for debugging/verification)
	// ------------------------------------------------------------------------
	file_utils_.dumpBinary(output_tensor.data, "cpp_08_inference-Output-RKNN.bin");

	// ------------------------------------------------------------------------
	// Step 5: Release output buffer allocated by RKNN
	// IMPORTANT: Must call this to avoid memory leak
	// ------------------------------------------------------------------------
	rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

	return output_tensor;
}
