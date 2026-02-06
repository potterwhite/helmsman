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
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"

#include <onnxruntime_cxx_api.h>
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
	// logger.AddSink(
	//     std::make_shared<arcforge::embedded::utils::FileSink>("/root/my_app_client.log"));
	logger.AddSink(std::make_shared<arcforge::embedded::utils::ConsoleSink>());

	//----------------------------------

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

	if (argc != 2) {
		std::cerr << "Usage: load_onnx <onnx_path> \n";
		return 1;
	}

	const std::string onnx_path = argv[1];

	try {
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-demo");

		// 2. session
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

		// 3. create session
		Ort::Session session(env, onnx_path.c_str(), session_options);

		Ort::AllocatorWithDefaultOptions allocator;

		// 4. Echo input message
		size_t num_inputs = session.GetInputCount();
		// std::count << "Number of inputs: " << num_inputs << std::end;
		logger.Info("Number of inputs:" + num_inputs);

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

	} catch (const Ort::Exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}