#pragma once
#include <string>
#include "pipeline/core/data_structure.h"

class InferenceEngine {
   public:
	virtual ~InferenceEngine() = default;

	virtual void load(const std::string& model_path) = 0;
	virtual TensorData infer(const TensorData& input) = 0;
};
