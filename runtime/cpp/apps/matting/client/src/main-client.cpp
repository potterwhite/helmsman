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
#include "Utils/logger/logger.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"

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

using namespace arcforge::embedded;

const std::string_view kcurrent_app_name = "matting-client";

const std::string ksocket_path = "/tmp/soCket.paTh";

// --- kill signal capture ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);

auto& logger = arcforge::embedded::utils::Logger::GetInstance();
auto& file_utils = arcforge::utils::FileUtils::GetInstance();

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

Ort::SessionOptions init_session_option(void) {
	Ort::SessionOptions opt;
	opt.SetIntraOpNumThreads(1);
	opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

	return opt;
}

void show_input(const Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions allocator;

	// 4. Echo input message
	size_t num_inputs = session.GetInputCount();
	// std::count << "Number of inputs: " << num_inputs << std::end;
	logger.Info("Number of inputs: " + std::to_string(num_inputs));

	for (size_t i = 0; i < num_inputs; i++) {
		char* input_name = session.GetInputName(i, allocator);
		auto input_type_info = session.GetInputTypeInfo(i);
		auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_shape = tensor_info.GetShape();

		std::cout << "Input " << i << " name: " << input_name << "\n";
		std::cout << "Input shape: [ ";
		for (auto dim : input_shape) {
			std::cout << dim << " ";
		}
		std::cout << "]\n";

		allocator.Free(input_name);
	}
}

void show_output(const Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions allocator;

	// 5. echo output message
	size_t num_outputs = session.GetOutputCount();
	std::cout << "Number of outputs: " << num_outputs << std::endl;

	for (size_t i = 0; i < num_outputs; i++) {
		char* output_name = session.GetOutputName(i, allocator);
		auto output_type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_shape = tensor_info.GetShape();

		std::cout << "Output " << i << " name: " << output_name << "\n";
		std::cout << "Output shape: [ ";
		for (auto dim : output_shape) {
			std::cout << dim << " ";
		}
		std::cout << "]\n";

		allocator.Free(output_name);
	}
}

std::vector<float> hwcToNchw(const cv::Mat& origin_img) {
	// 基本检查
	CV_Assert(origin_img.type() == CV_32FC3);
	CV_Assert(origin_img.isContinuous());

	const size_t H = static_cast<size_t>(origin_img.rows);
	const size_t W = static_cast<size_t>(origin_img.cols);
	const size_t C = 3;

	// 输出：1 * 3 * H * W
	std::vector<float> nchw;
	nchw.resize(C * H * W);

	// OpenCV HWC: [H][W][C]
	// NCHW: [C][H][W]
	for (size_t h = 0; h < H; ++h) {
		for (size_t w = 0; w < W; ++w) {
			const cv::Vec3f& pixel =
			    origin_img.at<cv::Vec3f>(static_cast<int>(h), static_cast<int>(w));

			// R
			nchw[0 * H * W + h * W + w] = pixel[0];
			// G
			nchw[1 * H * W + h * W + w] = pixel[1];
			// B
			nchw[2 * H * W + h * W + w] = pixel[2];
		}
	}

	logger.Info("Width=" + std::to_string(W) + ", Height=" + std::to_string(H), kcurrent_app_name);

	return nchw;
}

std::pair<double, double> getScaleFactor(int im_h, int im_w, int ref_size) {
	int im_rh;
	int im_rw;

	if (std::max(im_h, im_w) < ref_size || std::min(im_h, im_w) > ref_size) {
		if (im_w >= im_h) {
			im_rh = ref_size;
			im_rw = static_cast<int>(static_cast<double>(im_w) / im_h * ref_size);
		} else {
			im_rw = ref_size;
			im_rh = static_cast<int>(static_cast<double>(im_h) / im_w * ref_size);
		}
	} else {
		im_rh = im_h;
		im_rw = im_w;
	}

	// 对齐到 32 倍数
	im_rw = im_rw - (im_rw % 32);
	im_rh = im_rh - (im_rh % 32);

	double x_scale_factor = static_cast<double>(im_rw) / im_w;

	double y_scale_factor = static_cast<double>(im_rh) / im_h;

	return {x_scale_factor, y_scale_factor};
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
	// signal(SIGTERM, SignalHan/development/docker_volumes/src/ai/image-matting/helmsman.git/runtime/cpp/build/native-release/debug/dler);

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
		auto scale_factor = getScaleFactor(img.rows, img.cols, ref_size);
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
		std::vector<float> result = hwcToNchw(img);

		// 3. dump binary
		file_utils.dumpBinary(result, outputBinPath + "/cpp_01_nchw_input.bin");

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	// try {
	// 	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-demo");

	// 	// 3. create session
	// 	Ort::Session session(env, onnx_path.c_str(), init_session_option());

	// 	show_input(session);

	// 	show_output(session);

	// } catch (const Ort::Exception& e) {
	// 	std::cerr << "Error: " << e.what() << std::endl;
	// 	return 1;
	// }

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}