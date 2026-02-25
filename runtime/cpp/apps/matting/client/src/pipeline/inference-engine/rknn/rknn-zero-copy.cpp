#include "pipeline/inference-engine/rknn/rknn-zero-copy.h"
#include <cstring>
#include <fstream>
#include "Runtime/rknn.h/rknn.h"
#include "common-define.h"

// ============================================================================
// InferenceEngineRKNNZeroCP
//
// This class implements RKNN inference using Zero-Copy mode.
//
// Zero-Copy Design Philosophy:
//   • Allocate NPU-visible buffers once
//   • Bind buffers directly to input/output tensors
//   • Avoid rknn_inputs_set / rknn_outputs_get
//   • CPU writes directly into NPU memory
//   • CPU reads directly from NPU output buffer
//
// High-Level Lifecycle:
//
//   Step 1 - Initialize RKNN context
//   Step 2 - Query input/output attributes
//   Step 3 - Allocate zero-copy buffers
//   Step 4 - Bind buffers to tensors
//   Step 5 - Run inference
//   Step 6 - Read output directly from mapped memory
// ============================================================================

InferenceEngineRKNNZeroCP::InferenceEngineRKNNZeroCP() {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineRKNNZeroCP constructed.",
	                                                      kcurrent_module_name);
}

InferenceEngineRKNNZeroCP::~InferenceEngineRKNNZeroCP() {
	// Ensure buffers are released before destroying context
	releaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

// ============================================================================
// Load Model and Initialize Zero-Copy Environment
//
// Major Phases:
//
//   Phase I   - Initialize RKNN context
//   Phase II  - Query input/output metadata
//   Phase III - Allocate zero-copy memory
//   Phase IV  - Bind memory to tensors
// ============================================================================
void InferenceEngineRKNNZeroCP::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// ------------------------------------------------------------------------
	// Phase I - RKNN Context Initialization
	// ------------------------------------------------------------------------

	// // I. Read RKNN model from binary buffer (disabled version)
	// std::ifstream file(model_path, std::ios::binary | std::ios::ate);
	// size_t model_size = static_cast<size_t>(file.tellg());
	// file.seekg(0);
	// std::vector<uint8_t> model_data(model_size);
	// file.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(model_size));
	// file.close();

	/* II. RKNN context initialization explanation:
	 *
	 * rknn_context *context:
	 *     Pointer to RKNN runtime context.
	 *
	 * void *model:
	 *     Either:
	 *       - Pointer to binary RKNN model data
	 *       - OR file path string to RKNN model
	 *
	 * uint32_t size:
	 *     > 0 : model is binary data
	 *     = 0 : model is file path
	 *
	 * uint32_t flag:
	 *     Initialization flags (default behavior = 0)
	 *
	 * rknn_init_extend:
	 *     Optional extended initialization config.
	 *     Pass NULL when not needed.
	 */

	// ---------------
	// Option 1: Binary buffer mode (disabled here)
	// int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);

	// ---------------
	// Option 2: File path mode (used here)
	int ret = rknn_init(&ctx_,
	                    const_cast<void*>(static_cast<const void*>(model_path.c_str())),
	                    0,
	                    0,
	                    nullptr);

	if (ret < 0) {
		throw std::runtime_error("rknn_init failed!");
	} else {
		logger.Info("RKNN model loaded and context initialized successfully.",
		            kcurrent_module_name);
	}

	// ------------------------------------------------------------------------
	// Phase II - Query Model Input/Output Information
	// ------------------------------------------------------------------------

	// III. Query number of input/output tensors
	rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

	logger.Info("Input num: " + std::to_string(io_num_.n_input), kcurrent_module_name);
	logger.Info("Output num: " + std::to_string(io_num_.n_output), kcurrent_module_name);

	// IV. Query input and output tensor attributes
	memset(&input_attr_, 0, sizeof(input_attr_));
	input_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));

	memset(&output_attr_, 0, sizeof(output_attr_));
	output_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));

	input_size_ = input_attr_.size;
	output_size_ = output_attr_.size;

	logger.Info("Input size bytes: " + std::to_string(input_size_), kcurrent_module_name);
	logger.Info("Output size bytes: " + std::to_string(output_size_), kcurrent_module_name);

	logger.Info("Input attr: " + arcforge::runtime::to_string(input_attr_));
	logger.Info("Output attr: " + arcforge::runtime::to_string(output_attr_));

	// ------------------------------------------------------------------------
	// Phase III - Create Zero-Copy Buffers
	// ------------------------------------------------------------------------
	// rknn_create_mem allocates NPU-accessible memory.
	// This memory is shared between CPU and NPU.
	// No additional copy is needed during inference.
	// ------------------------------------------------------------------------
	input_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(input_size_));
	output_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(output_size_));

	if (!input_mem_ || !output_mem_) {
		throw std::runtime_error("rknn_create_mem failed!");
	}

	// ------------------------------------------------------------------------
	// Phase IV - Bind Memory to Tensor Descriptors
	// ------------------------------------------------------------------------
	// rknn_set_io_mem connects allocated memory with model input/output.
	// After binding:
	//   • Writing into input_mem_->virt_addr feeds the model
	//   • Reading from output_mem_->virt_addr retrieves results
	// ------------------------------------------------------------------------
	rknn_set_io_mem(ctx_, input_mem_, &input_attr_);
	rknn_set_io_mem(ctx_, output_mem_, &output_attr_);

	logger.Info("Zero-copy buffers allocated and bound.", kcurrent_module_name);
}

// ============================================================================
// Inference (Zero-Copy Mode)
//
// Execution Steps:
//
//   Step 1 - Validate input size
//   Step 2 - Copy CPU tensor into zero-copy input buffer
//   Step 3 - Run inference
//   Step 4 - Read output directly from mapped memory
//   Step 5 - Assemble TensorData
// ============================================================================
TensorData InferenceEngineRKNNZeroCP::infer(const TensorData& input) {

	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// ------------------------------------------------------------------------
	// Step 1 - Validate input size
	// ------------------------------------------------------------------------
	if (input.data.size() * sizeof(float) != input_size_) {
		throw std::runtime_error("Input size mismatch with RKNN model.");
	}

	// ------------------------------------------------------------------------
	// Step 2 - Write data into zero-copy input buffer
	//
	// No rknn_inputs_set is used.
	// We directly memcpy into memory shared with NPU.
	// ------------------------------------------------------------------------
	memcpy(input_mem_->virt_addr, input.data.data(), input_size_);

	// ------------------------------------------------------------------------
	// Step 3 - Execute inference on NPU
	// ------------------------------------------------------------------------
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	// ------------------------------------------------------------------------
	// Step 4 - Read output directly from zero-copy buffer
	//
	// No rknn_outputs_get is needed.
	// The output is already in output_mem_->virt_addr.
	// ------------------------------------------------------------------------
	float* output_data = reinterpret_cast<float*>(output_mem_->virt_addr);
	size_t output_element_count = output_size_ / sizeof(float);

	// Dump output for debugging consistency
	std::vector<float> output_vector(output_data, output_data + output_element_count);
	file_utils_.dumpBinary(output_vector, output_bin_path_ + "cpp_08_inference-Output.bin");

	// ------------------------------------------------------------------------
	// Step 5 - Construct TensorData structure
	// ------------------------------------------------------------------------
	TensorData output;
	output.data = output_vector;

	// Assemble output shape from tensor attribute metadata
	output.shape.clear();
	for (uint32_t i = 0; i < output_attr_.n_dims; i++) {
		output.shape.push_back(output_attr_.dims[i]);
	}

	return output;
}

// ============================================================================
// Release Zero-Copy Buffers
//
// Must be called before destroying context to avoid memory leak.
// ============================================================================
void InferenceEngineRKNNZeroCP::releaseBuffers() {
	if (input_mem_) {
		rknn_destroy_mem(ctx_, input_mem_);
		input_mem_ = nullptr;
	}

	if (output_mem_) {
		rknn_destroy_mem(ctx_, output_mem_);
		output_mem_ = nullptr;
	}
}
