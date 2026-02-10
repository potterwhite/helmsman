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

#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "CVKit/base/base.h"

#include <chrono>
#include <csignal>  // For signal handling
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>  // For std::ostringstream
#include <string>
#include <thread>
#include <vector>

using namespace arcforge::embedded;

const std::string_view kcurrent_app_name = "matting-client";

const std::string ksocket_path = "/tmp/soCket.paTh";

// --- kill signal capture ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);

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

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
	// auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	// logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);

	//*****************************************************
	// logger level configuration
	// obtain logger unique instance
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// 1. Configure logger level
	if (isRelease() == true) {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kinfo);
	} else {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);
	}

	// 2. Configure output targets (Sinks)
	logger.ClearSinks();
	logger.AddSink(
	    std::make_shared<arcforge::embedded::utils::FileSink>("/root/my_app_client.log"));
	logger.AddSink(std::make_shared<arcforge::embedded::utils::ConsoleSink>());

	//----------------------------------

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

	if (argc != 3) {
		std::cerr << "Usage: preprocess_normalize <input_image> <output_bin>\n";
		return 1;
	}

	const std::string imagePath = argv[1];
	const std::string outputBinPath = argv[2];
	auto cvkit_obj = std::make_unique<arcforge::cvkit::Base>();
	try {
		cv::Mat img = cvkit_obj->loadImage(imagePath);
		cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_00_01_imread.bin");

		img = cvkit_obj->bgrToRgb(img);
		cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_00_02_bgrToRgb.bin");

		img = cvkit_obj->ensure3Channel(img);
		cvkit_obj->dumpBinary(img, outputBinPath + "/cpp_00_03_rgb3.bin");
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

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}