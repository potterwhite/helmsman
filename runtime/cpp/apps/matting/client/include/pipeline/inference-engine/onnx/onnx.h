#pragma once

#include "Runtime/onnx/onnx.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"
#include "pipeline/core/data_structure.h"
#include "pipeline/inference-engine/base/inference-engine.h"

class InferenceEngineONNX : public InferenceEngine {

   public:
	InferenceEngineONNX();
	~InferenceEngineONNX();
	// static InferenceEngineONNX& GetInstance();

	// getter and setter
	void setOutputBinPath(const std::string& path);

	void load(const std::string& model_path) override;
	TensorData infer(const TensorData& input) override;

   private:
	// member functions

   private:
	// member variables
	Ort::Env env_;
	std::unique_ptr<Ort::Session> session_;

	std::string input_name_;
	std::string output_name_;

	std::string output_bin_path_;
};
