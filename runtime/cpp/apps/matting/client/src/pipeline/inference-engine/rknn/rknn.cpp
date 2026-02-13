#include "pipeline/inference-engine/rknn/rknn.h"
#include <cstring>
#include <fstream>
#include "Runtime/rknn.h/rknn.h"
#include "common-define.h"

InferenceEngineRKNN::InferenceEngineRKNN() {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineRKNN constructed.",
	                                                      kcurrent_module_name);
}

InferenceEngineRKNN::~InferenceEngineRKNN() {
	releaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

void InferenceEngineRKNN::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// // I. 读取 rknn 文件
	// std::ifstream file(model_path, std::ios::binary | std::ios::ate);
	// // std::streamsize model_size = file.tellg();
	// size_t model_size = static_cast<size_t>(file.tellg());
	// file.seekg(0);

	// std::vector<uint8_t> model_data(model_size);
	// file.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(model_size));
	// file.close();

	/* II. rknn context initialization
	 * rknn_context *context:
	 *		pointer of rknn_context。
	 * void *model:
	 * 		RKNN模型的二进制数据或者RKNN模型路径。
	 * 		当参数size大于0时,model表示二进制数据;
	 * 		当参数size等于0时,model表示RKNN模型路径。
	 * uint32_t size:
	 * 		当model是二进制数据,表示模型大小,当model是路径,则设置为0。
	 * uint32_t flag:
	 * 		初始化标志,默认初始化行为需要设置为0。
	 * rknn_init_extend:
	 * 		特定初始化时的扩展信息。没有使用,传入NULL即可。如果需要共享模型weight内存,则需要传入另个模型rknn_context指针。
	*/
	// ---------------
	// 1. binary mode
	// int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);
	// ---------------
	// 2. file path mode
	int ret = rknn_init(&ctx_, const_cast<void*>(static_cast<const void*>(model_path.c_str())), 0,
	                    0, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_init failed!");
	} else {
		logger.Info("RKNN model loaded and context initialized successfully.",
		            kcurrent_module_name);
	}

	// III. 查询 IO 数量
	rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

	logger.Info("Input num: " + std::to_string(io_num_.n_input), kcurrent_module_name);
	logger.Info("Output num: " + std::to_string(io_num_.n_output), kcurrent_module_name);

	// IV. 查询 input & output attr
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

	// ================================
	// V. ZERO-COPY BUFFER 创建
	// ================================
	input_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(input_size_));
	output_mem_ = rknn_create_mem(ctx_, static_cast<uint32_t>(output_size_));

	if (!input_mem_ || !output_mem_) {
		throw std::runtime_error("rknn_create_mem failed!");
	}

	// 绑定 buffer 到 tensor
	rknn_set_io_mem(ctx_, input_mem_, &input_attr_);
	rknn_set_io_mem(ctx_, output_mem_, &output_attr_);

	logger.Info("Zero-copy buffers allocated and bound.", kcurrent_module_name);
}

TensorData InferenceEngineRKNN::infer(const TensorData& input) {

	// auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	// 1️⃣ 检查 size
	if (input.data.size() * sizeof(float) != input_size_) {
		throw std::runtime_error("Input size mismatch with RKNN model.");
	}

	// 2️⃣ 🔥 zero-copy 写入 input buffer
	memcpy(input_mem_->virt_addr, input.data.data(), input_size_);

	// 3️⃣ Run
	int ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_run failed!");
	}

	// 4️⃣ 🔥 直接从 output buffer 读（没有 rknn_outputs_get）
	float* output_data = reinterpret_cast<float*>(output_mem_->virt_addr);

	size_t output_element_count = output_size_ / sizeof(float);

	// Dump
	std::vector<float> output_vector(output_data, output_data + output_element_count);
	file_utils_.dumpBinary(output_vector, output_bin_path_ + "cpp_08_inference-Output.bin");

	TensorData output;
	output.data = output_vector;

	// shape 组装
	output.shape.clear();
	for (uint32_t i = 0; i < output_attr_.n_dims; i++) {
		output.shape.push_back(output_attr_.dims[i]);
	}

	return output;
}

void InferenceEngineRKNN::releaseBuffers() {
	if (input_mem_) {
		rknn_destroy_mem(ctx_, input_mem_);
		input_mem_ = nullptr;
	}

	if (output_mem_) {
		rknn_destroy_mem(ctx_, output_mem_);
		output_mem_ = nullptr;
	}
}
