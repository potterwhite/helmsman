// Copyright (c) 2026 PotterWhite
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

#include "pipeline/pipeline.h"
#include <atomic>
#include <chrono>
#include <future>
#include <opencv2/videoio.hpp>
#include "common/common-define.h"
#include "pipeline/backend/backend.h"
#include "pipeline/backend/post-processor/guided-filter-post-processor.h"
#include "pipeline/frontend/frontend.h"

#ifdef ENABLE_RKNN_BACKEND
#include "pipeline/inference-engine/rknn/rknn-non-zero-copy.h"
#include "pipeline/inference-engine/rknn/rknn-zero-copy.h"
#else
#include "pipeline/inference-engine/onnx/onnx.h"
#endif

// Reference the global stop signal from main-server.cpp (set by SIGINT handler)
extern std::atomic<bool> g_stop_signal_received;

Pipeline& Pipeline::GetInstance() {
	static Pipeline instance;
	return instance;
}

Pipeline::Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline object constructed.",
	                                                      kcurrent_module_name);
}

Pipeline::~Pipeline() {
	arcforge::embedded::utils::Logger::GetInstance().Info("Pipeline cleaned up.",
	                                                      kcurrent_module_name);
}

void Pipeline::init(std::unique_ptr<InputSource> input_source, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
	this->input_source_ = std::move(input_source);
	this->model_path_ = model_path;
	this->output_bin_path_ = output_bin_path;
	this->background_path_ = background_path;
	this->model_type_ = model_type;
}

void Pipeline::init(const std::string& input_image_path, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
	this->input_image_path_ = input_image_path;
	this->model_path_ = model_path;
	this->output_bin_path_ = output_bin_path;
	this->background_path_ = background_path;
	this->model_type_ = model_type;
}

void Pipeline::verify_parameters_necessary() {
	if (input_source_ == nullptr && input_image_path_.empty()) {
		throw std::invalid_argument("No input source: neither video nor image provided.");
	}
	if (model_path_.empty()) {
		throw std::invalid_argument("Model path is empty.");
	}
	if (output_bin_path_.empty()) {
		throw std::invalid_argument("Output binary path is empty.");
	}
}

int Pipeline::run() {
	verify_parameters_necessary();

	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	switch (model_type_) {
		case ModelType::kMODNet:
			logger.Info("Pipeline: running MODNet path (single-frame)", kcurrent_module_name);
			return runMODNet();
		case ModelType::kRVM:
			logger.Info("Pipeline: running RVM path (recurrent multi-frame)", kcurrent_module_name);
			return runRVM();
		default:
			throw std::runtime_error("Unknown model type");
	}
}

// ============================================================================
// MODNet path — single-frame, 10x benchmark loop
//
// Fully backward-compatible with the original pipeline.
// inputs = { src }  (1 tensor)
// outputs = { pha } (1 tensor)
// ============================================================================
int Pipeline::runMODNet() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// 1. Load model to get input dimensions
	engine.setOutputBinPath(output_bin_path_);
	engine.load(model_path_);

#ifdef ENABLE_RKNN_BACKEND
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width = engine.getInputWidth();
#else
	size_t model_input_height = 512;
	size_t model_input_width = 512;
#endif

	// 2. Frontend: preprocess image
	frontend.setOutputBinPath(output_bin_path_);
	TensorData src = frontend.preprocess(input_image_path_, model_input_width, model_input_height);

	// 3. Inference: wrap single tensor into vector, benchmark 10 iterations
	std::vector<TensorData> inputs = {src};
	std::vector<TensorData> outputs;

	for (int i = 0; i < 10; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		engine.infer(inputs, outputs);
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = end - start;
		logger.Info("[Performance Benchmark " + std::to_string(i + 1) +
		                "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) +
		                " ms.",
		            kcurrent_module_name);
	}

	// 4. Backend: postprocess
	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(input_image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));
	backend.postprocess(outputs);

	return 0;
}

// ============================================================================
// RVM path — multi-frame with recurrent state management
//
// ONNX path: inputs  = { src, r1i, r2i, r3i, r4i, downsample_ratio } (6 tensors)
// RKNN path: inputs  = { src, r1i, r2i, r3i, r4i }                   (5 tensors)
//            (downsample_ratio is folded into a constant by ArcFoundry)
// outputs = { fgr, pha, r1o, r2o, r3o, r4o }                         (6 tensors)
//
// Currently runs N frames using the same image (single-image loop test).
// Will be extended to video frame input in Phase-5.
// ============================================================================
int Pipeline::runRVM_CV_SinglePicture() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// 1. Load model
	engine.setOutputBinPath(output_bin_path_);
	engine.load(model_path_);

#ifdef ENABLE_RKNN_BACKEND
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width = engine.getInputWidth();
#else
	// RVM ONNX models have dynamic shapes; use 288x512 as standard
	size_t model_input_height = 288;
	size_t model_input_width = 512;
#endif

	// 2. Initialize recurrent state manager with RVM state shapes.
	//    RVM MobileNetV3 with downsample_ratio=0.25 @ 288x512:
	//    The internal processing resolution = input * ds_ratio = 72x128.
	//    Recurrent state shapes are derived from the INTERNAL resolution:
	//      r1: [1, 16, 36, 64]   (internal_H/2, internal_W/2)
	//      r2: [1, 20, 18, 32]   (internal_H/4, internal_W/4)
	//      r3: [1, 40,  9, 16]   (internal_H/8, internal_W/8)
	//      r4: [1, 64,  5,  8]   (internal_H/16, internal_W/16)
	//
	//    These are empirically verified from Phase-2 RKNN and ONNX inference.
	//    First-frame uses [1,1,1,1] zero states (dynamic shape in ONNX);
	//    for C++ we pre-allocate the correct shape with zeros.
	constexpr float kDownsampleRatio = 0.25f;
	const int64_t internal_h =
	    static_cast<int64_t>(static_cast<float>(model_input_height) * kDownsampleRatio);
	const int64_t internal_w =
	    static_cast<int64_t>(static_cast<float>(model_input_width) * kDownsampleRatio);

	// Helper lambda for ceil division (RVM decoder uses ceil rounding)
	auto ceil_div = [](int64_t a, int64_t b) -> int64_t {
		return (a + b - 1) / b;
	};

	state_mgr_.init(
	    {
	        {1, 16, ceil_div(internal_h, 2), ceil_div(internal_w, 2)},    // r1: [1, 16, 36, 64]
	        {1, 20, ceil_div(internal_h, 4), ceil_div(internal_w, 4)},    // r2: [1, 20, 18, 32]
	        {1, 40, ceil_div(internal_h, 8), ceil_div(internal_w, 8)},    // r3: [1, 40,  9, 16]
	        {1, 64, ceil_div(internal_h, 16), ceil_div(internal_w, 16)},  // r4: [1, 64,  5,  8]
	    },
	    {"r1i", "r2i", "r3i", "r4i"});

	// 3. Frontend: preprocess image (same image for all frames in test mode)
	frontend.setOutputBinPath(output_bin_path_);
	TensorData src = frontend.preprocess(input_image_path_, model_input_width, model_input_height);

	// 4. Multi-frame inference loop (5 frames using same image for integration test)
	constexpr int kNumTestFrames = 5;

	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(input_image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));

	for (int frame = 0; frame < kNumTestFrames; ++frame) {
		logger.Info("=== RVM Frame " + std::to_string(frame + 1) + "/" +
		                std::to_string(kNumTestFrames) + " ===",
		            kcurrent_module_name);

		// Build input vector: src + r1i~r4i
		std::vector<TensorData> inputs = {src};
		state_mgr_.inject(inputs);  // appends r1i, r2i, r3i, r4i

#ifndef ENABLE_RKNN_BACKEND
		// ONNX path: append downsample_ratio as the 6th input tensor.
		// (RKNN path: downsample_ratio is folded into a constant by ArcFoundry)
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {kDownsampleRatio};
		inputs.push_back(std::move(dsr));
#endif

		// Run inference
		std::vector<TensorData> outputs;
		auto start = std::chrono::high_resolution_clock::now();
		engine.infer(inputs, outputs);
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = end - start;
		logger.Info("[RVM Frame " + std::to_string(frame + 1) +
		                "] infer() cost: " + std::to_string(dur.count()) + " ms.",
		            kcurrent_module_name);

		// Update recurrent states: r1o~r4o → r1i~r4i for next frame
		state_mgr_.update(outputs);  // outputs[2..5] = r1o~r4o

		// Post-process (only last frame saves result, or all frames for debugging)
		if (frame == kNumTestFrames - 1) {
			backend.postprocess(outputs);
		}
	}

	return 0;
}

// ============================================================================
// RVM path — multi-frame with recurrent state management
//
// ONNX path: inputs  = { src, r1i, r2i, r3i, r4i, downsample_ratio } (6 tensors)
// RKNN path: inputs  = { src, r1i, r2i, r3i, r4i }                   (5 tensors)
//            (downsample_ratio is folded into a constant by ArcFoundry)
// outputs = { fgr, pha, r1o, r2o, r3o, r4o }                         (6 tensors)
//
// Currently runs N frames using the same image (single-image loop test).
// Will be extended to video frame input in Phase-5.
// ============================================================================
int Pipeline::runRVM() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	ImageFrontend frontend;
#ifdef ENABLE_RKNN_BACKEND
	InferenceEngineRKNNZeroCP engine;
#else
	InferenceEngineONNX engine;
#endif
	MattingBackend backend;

	// 1. Load model
	engine.setOutputBinPath(output_bin_path_);
	engine.load(model_path_);

#ifdef ENABLE_RKNN_BACKEND
	size_t model_input_height = engine.getInputHeight();
	size_t model_input_width = engine.getInputWidth();
#else
	// RVM ONNX models have dynamic shapes; use 288x512 as standard
	size_t model_input_height = 288;
	size_t model_input_width = 512;
#endif

	// 2. Initialize recurrent state manager with RVM state shapes.
	//    RVM MobileNetV3 with downsample_ratio=0.25 @ 288x512:
	//    The internal processing resolution = input * ds_ratio = 72x128.
	//    Recurrent state shapes are derived from the INTERNAL resolution:
	//      r1: [1, 16, 36, 64]   (internal_H/2, internal_W/2)
	//      r2: [1, 20, 18, 32]   (internal_H/4, internal_W/4)
	//      r3: [1, 40,  9, 16]   (internal_H/8, internal_W/8)
	//      r4: [1, 64,  5,  8]   (internal_H/16, internal_W/16)
	//
	//    These are empirically verified from Phase-2 RKNN and ONNX inference.
	//    First-frame uses [1,1,1,1] zero states (dynamic shape in ONNX);
	//    for C++ we pre-allocate the correct shape with zeros.
	constexpr float kDownsampleRatio = 0.25f;
	const int64_t internal_h =
	    static_cast<int64_t>(static_cast<float>(model_input_height) * kDownsampleRatio);
	const int64_t internal_w =
	    static_cast<int64_t>(static_cast<float>(model_input_width) * kDownsampleRatio);

	// Helper lambda for ceil division (RVM decoder uses ceil rounding)
	auto ceil_div = [](int64_t a, int64_t b) -> int64_t {
		return (a + b - 1) / b;
	};

	state_mgr_.init(
	    {
	        {1, 16, ceil_div(internal_h, 2), ceil_div(internal_w, 2)},    // r1: [1, 16, 36, 64]
	        {1, 20, ceil_div(internal_h, 4), ceil_div(internal_w, 4)},    // r2: [1, 20, 18, 32]
	        {1, 40, ceil_div(internal_h, 8), ceil_div(internal_w, 8)},    // r3: [1, 40,  9, 16]
	        {1, 64, ceil_div(internal_h, 16), ceil_div(internal_w, 16)},  // r4: [1, 64,  5,  8]
	    },
	    {"r1i", "r2i", "r3i", "r4i"});

	// 3. Frontend: preprocess image (same image for all frames in test mode)
	frontend.setOutputBinPath(output_bin_path_);

	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	// Video mode: skip GuidedFilter post-processor.
	// GF requires a high-res guide image (foreground_image_path_), which doesn't exist
	// in video mode — each frame IS the foreground. GF is designed for single-image
	// refinement and would hurt throughput for Phase-5.3 (30fps target).

	// --- Block 5.1.5: Video composition output ---
	// Set up VideoWriter to produce a composited MP4 output.
	// Alpha compositing: composite = fg * alpha + bg * (1 - alpha)
	// Default background: solid green (chroma key style).
	const int src_width = input_source_->width();
	const int src_height = input_source_->height();
	const double src_fps = input_source_->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;

	const std::string output_video_path = output_bin_path_ + "/output_composited.mp4";
	cv::VideoWriter video_writer;
	video_writer.open(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), output_fps,
	                  cv::Size(src_width, src_height));

	if (!video_writer.isOpened()) {
		logger.Warning("Failed to open VideoWriter: " + output_video_path +
		                   ". Composited video will NOT be saved.",
		               kcurrent_module_name);
	} else {
		logger.Info("VideoWriter opened: " + output_video_path + " (" + std::to_string(src_width) +
		                "x" + std::to_string(src_height) + " @ " + std::to_string(output_fps) +
		                " fps)",
		            kcurrent_module_name);
	}

	// Load or create background image for compositing
	cv::Mat bg_bgr;
	if (!background_path_.empty()) {
		bg_bgr = cv::imread(background_path_, cv::IMREAD_COLOR);
		if (!bg_bgr.empty()) {
			cv::resize(bg_bgr, bg_bgr, cv::Size(src_width, src_height));
		}
	}
	if (bg_bgr.empty()) {
		// Default: solid blue background (easier to verify than green on green-bg videos)
		bg_bgr = cv::Mat(src_height, src_width, CV_8UC3, cv::Scalar(255, 100, 0));
	}

	cv::Mat frame;
	int frame_count = 0;

	while (input_source_->read(frame)) {
		// Check for Ctrl+C / SIGINT
		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing video at frame " +
			                std::to_string(frame_count) + ".",
			            kcurrent_module_name);
			break;
		}

		logger.Info("=== RVM Frame " + std::to_string(frame_count + 1) + " ===",
		            kcurrent_module_name);

		// 1. Frontend: cv::Mat → TensorData
		TensorData src = frontend.preprocess(frame, model_input_width, model_input_height);

		// 2. Build input vector: src + r1i~r4i
		std::vector<TensorData> inputs = {src};
		state_mgr_.inject(inputs);  // appends r1i, r2i, r3i, r4i

#ifndef ENABLE_RKNN_BACKEND
		// ONNX path: append downsample_ratio as the 6th input tensor.
		// (RKNN path: downsample_ratio is folded into a constant by ArcFoundry)
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {kDownsampleRatio};
		inputs.push_back(std::move(dsr));
#endif

		// 3. Run inference
		std::vector<TensorData> outputs;
		auto start = std::chrono::high_resolution_clock::now();
		engine.infer(inputs, outputs);
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = end - start;
		logger.Info("[RVM Frame " + std::to_string(frame_count + 1) +
		                "] infer() cost: " + std::to_string(dur.count()) + " ms.",
		            kcurrent_module_name);

		// 4. Update recurrent states: r1o~r4o → r1i~r4i for next frame
		state_mgr_.update(outputs);  // outputs[2..5] = r1o~r4o

		// 5. Post-process: get alpha matte (CV_8UC1, original resolution)
		cv::Mat alpha_8u = backend.postprocess(outputs);

		// 6. Alpha compositing: composite = fg * alpha + (1 - alpha) * bg
		if (video_writer.isOpened() && !alpha_8u.empty()) {
			// Ensure alpha is single channel
			cv::Mat alpha_1ch;
			if (alpha_8u.channels() == 1) {
				alpha_1ch = alpha_8u;
			} else {
				cv::cvtColor(alpha_8u, alpha_1ch, cv::COLOR_BGR2GRAY);
			}

			// Resize alpha to match original frame if needed
			if (alpha_1ch.size() != frame.size()) {
				cv::resize(alpha_1ch, alpha_1ch, frame.size(), 0, 0, cv::INTER_LINEAR);
			}

			// Convert to float for blending
			cv::Mat alpha_f32;
			alpha_1ch.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);

			// Broadcast alpha to 3 channels
			cv::Mat alpha_3ch;
			cv::cvtColor(alpha_f32, alpha_3ch, cv::COLOR_GRAY2BGR);

			// Blend: composite = alpha * fg + (1 - alpha) * bg
			cv::Mat fg_f32, bg_f32, composed_f32, composed_8u;
			frame.convertTo(fg_f32, CV_32FC3, 1.0 / 255.0);
			bg_bgr.convertTo(bg_f32, CV_32FC3, 1.0 / 255.0);

			composed_f32 =
			    alpha_3ch.mul(fg_f32) + (cv::Scalar(1.0, 1.0, 1.0) - alpha_3ch).mul(bg_f32);
			composed_f32.convertTo(composed_8u, CV_8UC3, 255.0);

			video_writer.write(composed_8u);
		}

		frame_count++;
	}

	// Release VideoWriter (flush and finalize MP4)
	if (video_writer.isOpened()) {
		video_writer.release();
		logger.Info("Video compositing complete: " + std::to_string(frame_count) +
		                " frames written to " + output_video_path,
		            kcurrent_module_name);
	}

	logger.Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count),
	            kcurrent_module_name);

	return 0;
}
