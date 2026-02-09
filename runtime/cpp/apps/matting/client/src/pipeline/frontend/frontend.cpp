
#include "pipeline/frontend/frontend.h"

// ImageFrontend& ImageFrontend::GetInstance() {
// 	static ImageFrontend instance;

// 	return instance;
// }

ImageFrontend::ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend object constructed.");
}

ImageFrontend::~ImageFrontend() {
	arcforge::embedded::utils::Logger::GetInstance().Info("ImageFrontend cleaned up.");
}

void ImageFrontend::setOutputBinPath(const std::string& path) {
	outputBinPath_ = path;
}

TensorData ImageFrontend::preprocess(const std::string& image_path) {

	auto& logger_ = arcforge::embedded::utils::Logger::GetInstance();
	auto& file_utils_ = arcforge::utils::FileUtils::GetInstance();
	auto& math_utils_ = arcforge::utils::MathUtils::GetInstance();
	// auto& runtime_ = arcforge::runtime::RuntimeONNX::GetInstance();
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();

	TensorData tensor_data;

	cv::Mat img = cvkit_obj->loadImage(image_path);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_01_loadimage.bin");

	img = cvkit_obj->bgrToRgb(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_02_bgrToRgb.bin");

	img = cvkit_obj->ensure3Channel(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_03_ensure3Channel.bin");
	/*
	     * NOTE:
	     * DO NOT use cv::normalize / convertTo here.
	     * This must be bitwise identical to NumPy preprocessing.
	     *
	     * Date: Feb03.2026
	     * Author: PotterWhite
	     *
	     * img = cvkit_obj->normalizeToMinusOneToOne(img);
	     */
	img = cvkit_obj->normalize_exact_numpy(img);
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_04_normalized.bin");

	// 1. resize to fit model input size
	constexpr int ref_size = 512;
	auto scale_factor = math_utils_.getScaleFactor(img.rows, img.cols, ref_size);
	std::cout << std::setprecision(17) << "x_scale_factor=" << scale_factor.first
	          << ", y_scale_factor=" << scale_factor.second << std::endl;
	cv::resize(img,                  // src
	           img,                  // dst（可以原地）
	           cv::Size(),           // dsize 为空
	           scale_factor.first,   // fx
	           scale_factor.second,  // fy
	           cv::INTER_AREA        // interpolation
	);
	logger_.Info("Resized Width=" + std::to_string(img.cols) +
	             ", Resized Height=" + std::to_string(img.rows));
	cvkit_obj->dumpBinary(img, outputBinPath_ + "/cpp_05_resized.bin");

	// 2. convert to NCHW
	// std::vector<float> result = hwcToNchw(img);
	tensor_data.data = cvkit_obj->hwcToNchw(img, 3);
	//the number of 06 & 07 is according to python debug file naming
	file_utils_.dumpBinary(tensor_data.data, outputBinPath_ + "/cpp_06-07_hwcToNchw.bin");
	// NCHW
	tensor_data.shape = {1, 3, static_cast<int64_t>(img.rows), static_cast<int64_t>(img.cols)};
	// tensor_data.height = static_cast<int64_t>(img.rows);
	// tensor_data.width = static_cast<int64_t>(img.cols);

	return tensor_data;
}