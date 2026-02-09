// Copyright (c) 2025 PotterWhite
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "CVKit/base/base.h"
#include "Runtime/onnx/onnx.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <csignal>  // For signal handling
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>  // For std::ostringstream
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "main-client.h"

using namespace arcforge::embedded;

const std::string_view kcurrent_app_name = "matting-client";

const std::string ksocket_path = "/tmp/soCket.paTh";

// --- kill signal capture ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);

auto& logger = arcforge::embedded::utils::Logger::GetInstance();
auto& file_utils = arcforge::utils::FileUtils::GetInstance();
auto& math_utils = arcforge::utils::MathUtils::GetInstance();
auto& runtime = arcforge::runtime::RuntimeONNX::GetInstance();

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(), kcurrent_app_name);
}

bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

TensorData& processing_pipeline(const std::string& imagePath, const std::string& outputBinPath,
                                TensorData& tensor_data) {
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();

	cv::Mat img = cvkit_obj->loadImage(imagePath);
	cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_01_loadimage.bin");

	img = cvkit_obj->bgrToRgb(img);
	cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_02_bgrToRgb.bin");

	img = cvkit_obj->ensure3Channel(img);
	cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_03_ensure3Channel.bin");
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
	cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_04_normalized.bin");

	// 1. resize to fit model input size
	constexpr int ref_size = 512;
	auto scale_factor = math_utils.getScaleFactor(img.rows, img.cols, ref_size);
	std::cout << std::setprecision(17) << "x_scale_factor=" << scale_factor.first
	          << ", y_scale_factor=" << scale_factor.second << std::endl;
	cv::resize(img,                  // src
	           img,                  // dst（可以原地）
	           cv::Size(),           // dsize 为空
	           scale_factor.first,   // fx
	           scale_factor.second,  // fy
	           cv::INTER_AREA        // interpolation
	);
	logger.Info("Resized Width=" + std::to_string(img.cols) +
	                ", Resized Height=" + std::to_string(img.rows),
	            kcurrent_app_name);
	cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_05_resized.bin");

	// 2. convert to NCHW
	// std::vector<float> result = hwcToNchw(img);
	tensor_data.data = cvkit_obj->hwcToNchw(img, 3);
	//the number of 06 & 07 is according to python debug file naming
	file_utils.dumpBinary(tensor_data.data, outputBinPath + "/cpp_06-07_hwcToNchw.bin");
	tensor_data.height = static_cast<int64_t>(img.rows);
	tensor_data.width = static_cast<int64_t>(img.cols);

	return tensor_data;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
	// auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	// logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);

	//*****************************************************
	// logger level configuration
	// obtain logger unique instance

	// 1. Configure logger level
	if (isRelease() == true) {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kinfo);
	} else {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);
	}

	// 2. Configure output targets (Sinks)
	logger.ClearSinks();
	// logger.AddSink(
	//     std::make_shared<arcforge::embedded::utils::FileSink>("/root/my_app_client.log"));
	logger.AddSink(std::make_shared<arcforge::embedded::utils::ConsoleSink>());

	//----------------------------------

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

#if 1
	// v1.3.3 inference execution
	if (argc != 4) {
		std::cerr << "Usage: infer <image_path> <onnx_path> <output_bin>\n";
		return 1;
	}

	const std::string image_path = argv[1];
	const std::string onnx_path = argv[2];
	// const std::string input_bin_path = argv[3];
	const std::string output_bin_path = argv[3];

	try {
		// ----------------
		// Processing -- 1. create ONNX Runtime environment and session
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "v1.3.3-inference-execution");

		Ort::Session session(env, onnx_path.c_str(), runtime.init_session_option());

		Ort::AllocatorWithDefaultOptions allocator;

		const char* input_name = session.GetInputName(0, allocator);
		const char* output_name = session.GetOutputName(0, allocator);

		std::vector<int64_t> input_shape =
		    session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

		logger.Info("Input Name: " + std::string(input_name));
		logger.Info("Output Name: " + std::string(output_name));

		logger.Info("Input Shape: ");
		for (auto s : input_shape) {
			logger.Info(std::to_string(s) + " ");
		}
		logger.Info("\n");

		// ----------------
		// Processing -- 2. obtain std::vector<float> input tensor from preprocessing pipeline
		TensorData tensor_data;
		std::vector<float> input_tensor_values;
		tensor_data = processing_pipeline(image_path, output_bin_path, tensor_data);
		input_tensor_values = tensor_data.data;
		logger.Info("Loaded input tensor size: " + std::to_string(input_tensor_values.size()));

		//---------------
		// Processing -- 3. process input tensor shape
		int64_t N = 1;
		int64_t C = 3;
		int64_t H = tensor_data.height;
		int64_t W = tensor_data.width;

		std::vector<int64_t> real_input_shape = {N, C, H, W};

		size_t input_tensor_size = static_cast<size_t>(N * C * H * W);

		logger.Info("Input tensor size: " + std::to_string(input_tensor_size));
		logger.Info("Input tensor actual size: " + std::to_string(input_tensor_values.size()));
		if (input_tensor_size != input_tensor_values.size()) {
			logger.Error("❌ Size mismatch! expected " + std::to_string(input_tensor_size) +
			             " got " + std::to_string(input_tensor_values.size()));
			return 1;
		}

		//----------------
		// Processing -- 4. create input tensor object and run inference
		Ort::MemoryInfo memory_info =
		    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		    memory_info, input_tensor_values.data(), input_tensor_size, real_input_shape.data(),
		    real_input_shape.size());

		auto output_tensors =
		    session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

		// ----------------
		// Processing -- Echo output tensor
		float* output_data = output_tensors[0].GetTensorMutableData<float>();

		auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

		logger.Info("Output Shape: ");
		for (auto s : output_shape) {
			logger.Info(std::to_string(s) + " ");
		}
		logger.Info("\n");

		size_t output_tensor_size = 1;
		for (auto s : output_shape) {
			output_tensor_size *= static_cast<size_t>(s);
		}

		// --------------------
		// Processing -- Dump output tensor to binary file
		std::vector<float> output_vector(output_data, output_data + output_tensor_size);

		file_utils.dumpBinary(output_vector,
		                      output_bin_path + "cpp_08_inference-Output.bin");

		logger.Info("✅ Inference done. Output dumped.");

	} catch (const Ort::Exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
#endif

#if 0
	// v1.3.2 hwcToNchw
	if (argc != 3) {
		std::cerr << "Usage: preprocess_normalize <input_image> <output_bin>\n";
		return 1;
	}

	const std::string imagePath = argv[1];
	const std::string outputBinPath = argv[2];
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();
	try {
		cv::Mat img = cvkit_obj->loadImage(imagePath);
		img = cvkit_obj->bgrToRgb(img);
		img = cvkit_obj->ensure3Channel(img);
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
		cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_00_04_normalized.bin");

		// 1. resize to fit model input size
		constexpr int ref_size = 512;
		auto scale_factor = math_utils.getScaleFactor(img.rows, img.cols, ref_size);
		std::cout << std::setprecision(17) << "x_scale_factor=" << scale_factor.first
		          << ", y_scale_factor=" << scale_factor.second << std::endl;
		cv::resize(img,                  // src
		           img,                  // dst（可以原地）
		           cv::Size(),           // dsize 为空
		           scale_factor.first,   // fx
		           scale_factor.second,  // fy
		           cv::INTER_AREA        // interpolation
		);
		logger.Info("Resized Width=" + std::to_string(img.cols) +
		                ", Resized Height=" + std::to_string(img.rows),
		            kcurrent_app_name);

		// 2. convert to NCHW
		// std::vector<float> result = hwcToNchw(img);
		std::vector<float> result = cvkit_obj->hwcToNchw(img, 3);

		// 3. dump binary
		file_utils.dumpBinary(result, outputBinPath + "/cpp_01_nchw_input.bin");

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
#endif

#if 0
	// v1.3.1 Load and infer ONNX model
	if (argc != 2) {
		std::cerr << "Usage: load_onnx <onnx_path> \n";
		return 1;
	}
	const std::string onnx_path = argv[1];

	try {
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-demo");

		// 3. create session
		Ort::Session session(env, onnx_path.c_str(), runtime.init_session_option());

		runtime.show_input(session);

		runtime.show_output(session);

	} catch (const Ort::Exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
#endif

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}