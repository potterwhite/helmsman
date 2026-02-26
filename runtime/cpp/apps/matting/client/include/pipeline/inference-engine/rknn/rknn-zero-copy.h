#pragma once

#include <rknn_api.h>                                         // rknn_xxx APIs
#include <string>                                             // for std::string
#include "common-define.h"                                    // TensorData
#include "pipeline/inference-engine/base/inference-engine.h"  // InferenceEngine

class InferenceEngineRKNNZeroCP : public InferenceEngine {
   public:
	InferenceEngineRKNNZeroCP();
	~InferenceEngineRKNNZeroCP();

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
