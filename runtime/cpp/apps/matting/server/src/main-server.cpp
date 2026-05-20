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
#include <optional>
#include <sstream>  // For std::ostringstream
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "common/common-define.h"
#include "pipeline/pipeline.h"

using namespace helmsman;

// ============================================================================
// Global singletons
// ============================================================================

const std::string ksocket_path = "/tmp/soCket.paTh";
std::atomic<bool> g_stop_signal_received(false);

auto& logger = helmsman::utils::Logger::GetInstance();
auto& file_utils = helmsman::utils::FileUtils::GetInstance();
auto& math_utils = helmsman::utils::MathUtils::GetInstance();
auto& runtime = helmsman::runtime::RuntimeONNX::GetInstance();
auto& pipeline = Pipeline::GetInstance();

// ============================================================================
// Internal helpers
// ============================================================================

static void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	helmsman::utils::Logger::GetInstance().Warning(oss.str(), kcurrent_module_name);
}

[[maybe_unused]] static bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

[[maybe_unused]] static bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

static bool isVideoFile(const std::string& path) {
	auto dot = path.rfind('.');
	if (dot == std::string::npos)
		return false;
	std::string ext = path.substr(dot);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm");
}

// ============================================================================
// Server configuration & initialization
// ============================================================================

// Configure logger, parse CLI arguments, and log the resulting configuration.
// Returns nullopt when arguments are invalid (usage printed to stderr).
static std::optional<PipelineConfig> initServer(int argc, char* argv[]) {
	// --- Logger ---
	if (isRelease()) {
		logger.setLevel(helmsman::utils::LoggerLevel::kinfo);
	} else {
		logger.setLevel(helmsman::utils::LoggerLevel::kdebug);
	}

	logger.ClearSinks();
	logger.AddSink(std::make_shared<helmsman::utils::ConsoleSink>());

	signal(SIGINT, SignalHandler);

	if (isDumpEnabled()) {
		logger.Info(
		    "[DEBUG] Binary dump ENABLED (HELMSMAN_DUMP is set). "
		    "Unset to disable for production runs.",
		    kcurrent_module_name);
	}

	// --- Parse arguments ---
	PipelineConfig cfg;
	std::vector<std::string> positional_args;

	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--rvm") == 0) {
			cfg.model_type = ModelType::kRVM;
		} else if (std::strcmp(argv[i], "--modnet") == 0) {
			cfg.model_type = ModelType::kMODNet;
		} else if (std::strcmp(argv[i], "--output=mp4") == 0) {
			cfg.output_mode = OutputMode::kMp4;
		} else if (std::strcmp(argv[i], "--output=drm") == 0) {
			cfg.output_mode = OutputMode::kDrm;
		} else if (std::strcmp(argv[i], "--timing=off") == 0) {
			cfg.timing_enabled = false;
		} else if (std::strcmp(argv[i], "--timing=on") == 0) {
			cfg.timing_enabled = true;
		} else if (std::strcmp(argv[i], "--hwdecoder") == 0) {
			cfg.use_hardware_decoder = true;
		} else {
			positional_args.push_back(argv[i]);
		}
	}

	if (positional_args.size() < 3 || positional_args.size() > 4) {
		std::cerr << "Usage: " << argv[0]
		          << " <image_path> <model_path> <output_dir> [background_path] [--rvm] "
		             "[--output=mp4|drm] [--timing=off] [--hwdecoder]\n"
		          << "\n"
		          << "Flags:\n"
		          << "  --rvm          Use RVM (Robust Video Matting) with recurrent states\n"
		          << "  --modnet       Use MODNet single-frame matting (default)\n"
		          << "  --output=mp4   Write composited video to mp4 file (default)\n"
		          << "  --output=drm   Display on DRM/KMS panel (embedded only)\n"
		          << "  --timing=off   Disable pipeline timing statistics (default: on)\n"
		          << "  --timing=on    Enable pipeline timing statistics (default)\n"
		          << "  --hwdecoder    Use hardware decode path (requires FFmpeg + MPPKit)\n";
		return std::nullopt;
	}

	cfg.input_path = positional_args[0];
	cfg.model_path = positional_args[1];
	cfg.output_bin_path = positional_args[2];
	cfg.background_path = (positional_args.size() == 4) ? positional_args[3] : "";

	cfg.is_video = isVideoFile(cfg.input_path);

	// --- Log configuration ---
	std::string mode_str = (cfg.model_type == ModelType::kRVM) ? "RVM" : "MODNet";
	std::string output_str = (cfg.output_mode == OutputMode::kDrm) ? "DRM" : "MP4";
	std::string decode_str =
	    cfg.use_hardware_decoder ? "hardware decoder" : "software decoder (OpenCV)";
	logger.Info("Model type: " + mode_str, kcurrent_module_name);
	logger.Info("Output mode: " + output_str, kcurrent_module_name);
	logger.Info("Decode path: " + decode_str, kcurrent_module_name);
	logger.Info("Input:      " + cfg.input_path + (cfg.is_video ? " (video)" : " (image)"),
	            kcurrent_module_name);
	logger.Info("Model:      " + cfg.model_path, kcurrent_module_name);
	logger.Info("Output:     " + cfg.output_bin_path, kcurrent_module_name);
	if (!cfg.background_path.empty()) {
		logger.Info("Background: " + cfg.background_path, kcurrent_module_name);
	}

	return cfg;
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
// Environment variables:
//   HELMSMAN_DUMP=1   Enable debug binary dumps (off by default)
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
int main(int argc, char* argv[]) {
	auto config = initServer(argc, argv);
	if (!config)
		return 1;

	pipeline.init(*config);
	pipeline.run();

	return 0;
}
