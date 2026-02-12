#include "pipeline/inference-engine/rknn/rknn.h"
#include <cstring>
#include <fstream>

InferenceEngineRKNN::InferenceEngineRKNN() {
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineRKNN constructed.");
}

InferenceEngineRKNN::~InferenceEngineRKNN() {
	releaseBuffers();
	if (ctx_) {
		rknn_destroy(ctx_);
	}
}

void InferenceEngineRKNN::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// 1️⃣ 读取 rknn 文件
	std::ifstream file(model_path, std::ios::binary | std::ios::ate);
	size_t model_size = file.tellg();
	file.seekg(0);

	std::vector<uint8_t> model_data(model_size);
	file.read(reinterpret_cast<char*>(model_data.data()), model_size);
	file.close();

	// 2️⃣ 初始化 rknn context
	int ret = rknn_init(&ctx_, model_data.data(), model_size, 0, nullptr);
	if (ret < 0) {
		throw std::runtime_error("rknn_init failed!");
	}

	// 3️⃣ 查询 IO 数量
	rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

	logger.Info("Input num: " + std::to_string(io_num_.n_input));
	logger.Info("Output num: " + std::to_string(io_num_.n_output));

	// 4️⃣ 查询 input attr
	memset(&input_attr_, 0, sizeof(input_attr_));
	input_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));

	// 5️⃣ 查询 output attr
	memset(&output_attr_, 0, sizeof(output_attr_));
	output_attr_.index = 0;
	rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_));

	input_size_ = input_attr_.size;
	output_size_ = output_attr_.size;

	logger.Info("Input size bytes: " + std::to_string(input_size_));
	logger.Info("Output size bytes: " + std::to_string(output_size_));

	// ================================
	// 🔥🔥🔥 ZERO-COPY BUFFER 创建
	// ================================

	input_mem_ = rknn_create_mem(ctx_, input_size_);
	output_mem_ = rknn_create_mem(ctx_, output_size_);

	if (!input_mem_ || !output_mem_) {
		throw std::runtime_error("rknn_create_mem failed!");
	}

	// 绑定 buffer 到 tensor
	rknn_set_io_mem(ctx_, input_mem_, &input_attr_);
	rknn_set_io_mem(ctx_, output_mem_, &output_attr_);

	logger.Info("Zero-copy buffers allocated and bound.");
}

TensorData InferenceEngineRKNN::infer(const TensorData& input) {

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
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
