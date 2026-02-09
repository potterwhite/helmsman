#pragma once

#include "CVKit/base/base.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"
#include "pipeline/core/data_structure.h"

class ImageFrontend {
   public:
	ImageFrontend();
	~ImageFrontend();
	// static ImageFrontend& GetInstance();
	TensorData preprocess(const std::string& image_path);

	// getter and setter
	void setOutputBinPath(const std::string& path);

   private:
	// member functions
   private:
	// member variables
	std::string outputBinPath_ = "";
};
