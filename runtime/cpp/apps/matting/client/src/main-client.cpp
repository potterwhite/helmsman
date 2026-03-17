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
#include "common-define.h"
#include "pipeline/pipeline.h"

using namespace arcforge::embedded;

// const std::string_view kcurrent_module_name = "matting-client";

const std::string ksocket_path = "/tmp/soCket.paTh";

// --- kill signal capture ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);

auto& logger = arcforge::embedded::utils::Logger::GetInstance();
auto& file_utils = arcforge::utils::FileUtils::GetInstance();
auto& math_utils = arcforge::utils::MathUtils::GetInstance();
auto& runtime = arcforge::runtime::RuntimeONNX::GetInstance();
auto& pipeline = Pipeline::GetInstance();

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(), kcurrent_module_name);
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
	if (argc != 4 && argc != 5) {
		std::cerr << "Usage: infer <image_path> <onnx_path> <output_bin> [background_path]\n";
		return 1;
	}

	const std::string image_path = argv[1];
	const std::string onnx_path = argv[2];
	// const std::string input_bin_path = argv[3];
	const std::string output_bin_path = argv[3];
	const std::string background_path = (argc == 5) ? argv[4] : "";
	pipeline.init(image_path, onnx_path, output_bin_path, background_path);

	pipeline.run();

#endif

	std::cout << "hello " << kcurrent_module_name << std::endl;

	return 0;
}