
#include "pipeline/inference-engine/rknn/rknn-non-zero-copy.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

InferenceEngineRKNN::InferenceEngineRKNN() {
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "InferenceEngineRKNN object constructed. (Non-Zero-Copy Mode)", kcurrent_module_name);
}

InferenceEngineRKNN::~InferenceEngineRKNN() {
	release();
	arcforge::embedded::utils::Logger::GetInstance().Info("InferenceEngineRKNN cleaned up.",
	                                                      kcurrent_module_name);
}

void InferenceEngineRKNN::release() {
	if (ctx_ > 0) {
		rknn_destroy(ctx_);
		ctx_ = 0;
	}
}

void InferenceEngineRKNN::load(const std::string& model_path) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// ----------------
	// Processing -- 1. 读取 RKNN 模型文件到内存
	//
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

	// ----------------
	// Processing -- 2. 初始化 RKNN Context
	//
	int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_init fail! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_init failed");
	}

	// ----------------
	// Processing -- 3. 获取模型的输入输出数量和属性
	//
	ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
	if (ret < 0) {
		throw std::runtime_error("rknn_query IN_OUT_NUM failed");
	}
	logger.Info("Model Input Num: " + std::to_string(io_num_.n_input) +
	                ", Output Num: " + std::to_string(io_num_.n_output),
	            kcurrent_module_name);

	// 获取输入属性
	input_attrs_.clear();
	for (uint32_t i = 0; i < io_num_.n_input; i++) {
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
		input_attrs_.push_back(attr);

		logger.Info("Input " + std::to_string(i) + " name: " + std::string(attr.name),
		            kcurrent_module_name);
	}

	// 获取输出属性
	output_attrs_.clear();
	for (uint32_t i = 0; i < io_num_.n_output; i++) {
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
		output_attrs_.push_back(attr);

		logger.Info("Output " + std::to_string(i) + " name: " + std::string(attr.name),
		            kcurrent_module_name);
	}
}

TensorData InferenceEngineRKNN::infer(const TensorData& input) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();

	if (ctx_ == 0) {
		throw std::runtime_error("RKNN Context is null, please load model first.");
	}

	// ----------------
	// Processing -- 1. 设置输入 (rknn_inputs_set)
	//
	std::vector<rknn_input> inputs(io_num_.n_input);
	memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));

	// 这里假设我们只处理单输入，对应你 ONNX 的逻辑
	inputs[0].index = 0;

	// 【最核心的配置】：无论模型底层是 INT8 还是 FP16，我们都告诉 RKNN 我们传的是 FLOAT32
	// NPU 驱动会自动在底层将我们的 FLOAT32 数据量化/转化为模型需要的格式
	inputs[0].type = RKNN_TENSOR_FLOAT32;

	// fmt 必须和模型导出时一致，这里直接读取模型属性里的 fmt (通常是 NCHW 或 NHWC)
	inputs[0].fmt = input_attrs_[0].fmt;

	inputs[0].size = static_cast<uint32_t>(input.data.size() * sizeof(float));
	inputs[0].buf = const_cast<void*>(reinterpret_cast<const void*>(input.data.data()));

	// 非 Zero-Copy 模式关键：pass_through 设为 0，让 NPU 帮我们做数据转换处理
	inputs[0].pass_through = 0;

	int ret = rknn_inputs_set(ctx_, io_num_.n_input, inputs.data());
	if (ret < 0) {
		logger.Error("❌ rknn_inputs_set failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_inputs_set failed");
	}

	// ----------------
	// Processing -- 2. 执行推理 (rknn_run)
	//
	ret = rknn_run(ctx_, nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_run failed! ret=" + std::to_string(ret), kcurrent_module_name);
		throw std::runtime_error("rknn_run failed");
	}

	// ----------------
	// Processing -- 3. 获取输出 (rknn_outputs_get)
	//
	std::vector<rknn_output> outputs(io_num_.n_output);
	memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));

	// 假设只处理单输出
	outputs[0].want_float = 1;   // 【关键点】：要求 NPU 将输出的反量化直接转成 FLOAT32 给我们
	outputs[0].is_prealloc = 0;  // 非 Zero-Copy 模式，让 RKNN 驱动帮我们分配内存

	ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
	if (ret < 0) {
		logger.Error("❌ rknn_outputs_get failed! ret=" + std::to_string(ret),
		             kcurrent_module_name);
		throw std::runtime_error("rknn_outputs_get failed");
	}

	// ----------------
	// Processing -- 4. 组装输出 TensorData
	//
	TensorData output_tensor;
	float* out_data_ptr = reinterpret_cast<float*>(outputs[0].buf);
	size_t out_elements_count = outputs[0].size / sizeof(float);

	output_tensor.data.assign(out_data_ptr, out_data_ptr + out_elements_count);

	// 复制输出 Shape
	logger.Info("Output Shape: ", kcurrent_module_name);
	for (uint32_t i = 0; i < output_attrs_[0].n_dims; i++) {
		output_tensor.shape.push_back(output_attrs_[0].dims[i]);
		logger.Info(std::to_string(output_attrs_[0].dims[i]) + " ", kcurrent_module_name);
	}
	logger.Info("\n", kcurrent_module_name);

	// --------------------
	// Processing -- Dump output tensor to binary file (与你的 ONNX 逻辑保持一致)
	// 注：这里借用了你在 ONNX 代码中的输出路径变量 output_bin_path_ (如果你的基类有定义的话)
	// 如果没有，可以换成直接硬编码的临时测试路径
	file_utils_.dumpBinary(output_tensor.data, "cpp_08_inference-Output-RKNN.bin");

	// ----------------
	// Processing -- 5. 释放由 rknn_outputs_get 分配的输出内存 (必做，否则内存泄漏)
	//
	rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

	return output_tensor;
}