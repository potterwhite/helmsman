#pragma once

#include "CVKit/base/base.h"
#include "Runtime/onnx/onnx.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"

typedef struct {
	std::vector<float> data;
	int64_t height;
	int64_t width;
} TensorData;

// constexpr std::string_view kcurrent_app_name = "Pipeline";

class Pipeline {
   public:
	static Pipeline& GetInstance();
	void init(const std::string& image_path, const std::string& onnx_path,
	          const std::string& output_bin_path);
	int main_pipeline();

   private:
   	Pipeline();
	~Pipeline();

	void verify_parameters_necessary();
	TensorData& processing_pipeline(const std::string& imagePath, const std::string& outputBinPath,
	                                TensorData& tensor_data);

   private:
	std::string image_path_;
	std::string onnx_path_;
	std::string output_bin_path_;
};