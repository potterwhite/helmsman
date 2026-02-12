#pragma once

#ifdef ENABLE_RKNN_BACKEND
#include <rknn_api.h>
#endif

#include <string>
#include <vector>

#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "pipeline/inference-engine/base/inference-engine.h"

class InferenceEngineRKNN : public InferenceEngine {
   public:
	InferenceEngineRKNN();
	~InferenceEngineRKNN();

	void load(const std::string& model_path) override;
	TensorData infer(const TensorData& input) override;

   private:
	// member functions
	void releaseBuffers();

   private:
	// member variables
	rknn_context ctx_{0};

	rknn_input_output_num io_num_;
	rknn_tensor_attr input_attr_;
	rknn_tensor_attr output_attr_;

	rknn_tensor_mem* input_mem_{nullptr};
	rknn_tensor_mem* output_mem_{nullptr};

	size_t input_size_{0};
	size_t output_size_{0};
};
