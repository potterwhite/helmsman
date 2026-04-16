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
#include <cstring>  // For strcmp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>  // For std::ostringstream
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "common/common-define.h"
#include "input/mp4-input-source.h"
#include "pipeline/pipeline.h"

using namespace arcforge::embedded;

const std::string ksocket_path = "/tmp/soCket.paTh";

// --- kill signal capture ---
std::atomic<bool> g_stop_signal_received(false);

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

// Detect video files by extension
static bool isVideoFile(const std::string& path) {
	auto dot = path.rfind('.');
	if (dot == std::string::npos) return false;
	std::string ext = path.substr(dot);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm");
}

// ============================================================================
// CLI Usage:
//
//   Helmsman_Matting_Server <input> <model> <output_dir> [background] [--rvm]
//
// Positional arguments:
//   input        Path to input image OR video file (.mp4/.avi/.mkv/.mov/.webm)
//   model        Path to ONNX or RKNN model file
//   output_dir   Directory for output files and debug dumps
//   background   (optional) Path to background image for compositing
//
// Flags:
//   --rvm        Use RVM (Robust Video Matting) mode with recurrent states.
//                Default is MODNet (single-frame matting).
//                Auto-selected when input is a video file.
//
// Examples:
//   # MODNet single image (default):
//   Helmsman_Matting_Server photo.png modnet.onnx ./output/
//
//   # RVM single image test:
//   Helmsman_Matting_Server photo.png rvm.rknn ./output/ --rvm
//
//   # RVM video (auto-detected):
//   Helmsman_Matting_Server video.mp4 rvm.rknn ./output/
//
//   # RVM video with background compositing:
//   Helmsman_Matting_Server video.mp4 rvm.rknn ./output/ bg.jpg
// ============================================================================
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {

	// -----------------------------------------------
	// 1. Configure logger
	// -----------------------------------------------
	if (isRelease()) {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kinfo);
	} else {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);
	}

	logger.ClearSinks();
	logger.AddSink(std::make_shared<arcforge::embedded::utils::ConsoleSink>());

	// Setup signal handler
	signal(SIGINT, SignalHandler);

	// -----------------------------------------------
	// 2. Parse arguments
	// -----------------------------------------------
	// Scan for --rvm flag anywhere in argv, then collect positional args.
	ModelType model_type = ModelType::kMODNet;
	std::vector<std::string> positional_args;

	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--rvm") == 0) {
			model_type = ModelType::kRVM;
		} else if (std::strcmp(argv[i], "--modnet") == 0) {
			model_type = ModelType::kMODNet;  // explicit default
		} else {
			positional_args.push_back(argv[i]);
		}
	}

	if (positional_args.size() < 3 || positional_args.size() > 4) {
		std::cerr << "Usage: " << argv[0]
		          << " <image_path> <model_path> <output_dir> [background_path] [--rvm]\n"
		          << "\n"
		          << "Flags:\n"
		          << "  --rvm      Use RVM (Robust Video Matting) with recurrent states\n"
		          << "  --modnet   Use MODNet single-frame matting (default)\n";
		return 1;
	}

	const std::string input_path      = positional_args[0];
	const std::string model_path      = positional_args[1];
	const std::string output_bin_path = positional_args[2];
	const std::string background_path = (positional_args.size() == 4) ? positional_args[3] : "";

	// Auto-detect video input → force RVM mode
	const bool is_video = isVideoFile(input_path);
	if (is_video && model_type == ModelType::kMODNet) {
		logger.Info("Video input detected, auto-switching to RVM mode.", kcurrent_module_name);
		model_type = ModelType::kRVM;
	}

	// -----------------------------------------------
	// 3. Log configuration
	// -----------------------------------------------
	std::string mode_str = (model_type == ModelType::kRVM) ? "RVM" : "MODNet";
	logger.Info("Model type: " + mode_str, kcurrent_module_name);
	logger.Info("Input:      " + input_path + (is_video ? " (video)" : " (image)"), kcurrent_module_name);
	logger.Info("Model:      " + model_path, kcurrent_module_name);
	logger.Info("Output:     " + output_bin_path, kcurrent_module_name);
	if (!background_path.empty()) {
		logger.Info("Background: " + background_path, kcurrent_module_name);
	}

	// -----------------------------------------------
	// 4. Run pipeline
	// -----------------------------------------------
	if (is_video) {
		// Video mode: create Mp4InputSource, pass ownership to Pipeline
		auto source = std::make_unique<Mp4InputSource>();
		if (!source->open(input_path)) {
			logger.Warning("Failed to open video: " + input_path, kcurrent_module_name);
			return 1;
		}
		logger.Info("Video source: " + std::to_string(source->width()) + "x" +
		            std::to_string(source->height()) + " @ " +
		            std::to_string(source->fps()) + " fps", kcurrent_module_name);

		pipeline.init(std::move(source), model_path, output_bin_path,
		              background_path, model_type);
	} else {
		// Single image mode (existing path)
		pipeline.init(input_path, model_path, output_bin_path,
		              background_path, model_type);
	}

	pipeline.run();

	std::cout << "hello " << kcurrent_module_name << std::endl;

	return 0;
}
