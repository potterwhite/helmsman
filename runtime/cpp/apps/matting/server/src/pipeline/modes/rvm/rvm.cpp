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

#include "pipeline/modes/rvm/rvm.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include "Utils/timing/timer.h"
#include "common/common-define.h"
#include "pipeline/single-slot-channel.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using arcforge::utils::timing::ManualTimer;
using arcforge::utils::timing::ScopedTimer;
using arcforge::utils::timing::StageAccumulator;

extern std::atomic<bool> g_stop_signal_received;

constexpr float kDownsampleRatio = 0.25f;
inline constexpr std::string_view kRvmModuleName = "RVMMode";

void RVMMode::initRecurrentStates(size_t model_input_height, size_t model_input_width) {
	const int64_t internal_h =
	    static_cast<int64_t>(static_cast<float>(model_input_height) * kDownsampleRatio);
	const int64_t internal_w =
	    static_cast<int64_t>(static_cast<float>(model_input_width) * kDownsampleRatio);

	auto ceil_div = [](int64_t a, int64_t b) -> int64_t {
		return (a + b - 1) / b;
	};

	state_mgr_.init(
	    {
	        {1, 16, ceil_div(internal_h, 2), ceil_div(internal_w, 2)},
	        {1, 20, ceil_div(internal_h, 4), ceil_div(internal_w, 4)},
	        {1, 40, ceil_div(internal_h, 8), ceil_div(internal_w, 8)},
	        {1, 64, ceil_div(internal_h, 16), ceil_div(internal_w, 16)},
	    },
	    {"r1i", "r2i", "r3i", "r4i"});
}

bool RVMMode::openVideoWriter(cv::VideoWriter& writer, const std::string& path, int width,
                              int height, double fps) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
	if (!writer.isOpened()) {
		logger.Warning(
		    "Failed to open VideoWriter: " + path + ". Composited video will NOT be saved.",
		    kRvmModuleName);
		return false;
	}
	logger.Info("VideoWriter opened: " + path + " (" + std::to_string(width) + "x" +
	                std::to_string(height) + " @ " + std::to_string(fps) + " fps)",
	            kRvmModuleName);
	return true;
}

cv::Mat RVMMode::loadOrCreateBackground(int width, int height) {
	cv::Mat bg;
	if (!background_path_.empty()) {
		bg = cv::imread(background_path_, cv::IMREAD_COLOR);
		if (!bg.empty()) {
			cv::resize(bg, bg, cv::Size(width, height));
			return bg;
		}
	}
	return cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 100, 0));
}

cv::Mat RVMMode::inferOneFrame(InferenceEngine* engine, const TensorData& src,
                               const cv::Mat& guide_bgr) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	std::vector<TensorData> inputs = {src};
	state_mgr_.inject(inputs);

#if defined(INFERENCE_BACKEND_ONNX)
	TensorData dsr;
	dsr.name = "downsample_ratio";
	dsr.shape = {1};
	dsr.data = {kDownsampleRatio};
	inputs.push_back(std::move(dsr));
#endif

	std::vector<TensorData> outputs;
	auto t0 = std::chrono::high_resolution_clock::now();
	engine->infer(inputs, outputs);
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> dur = t1 - t0;
	logger.Info("infer() cost: " + std::to_string(dur.count()) + " ms.", kRvmModuleName);

	state_mgr_.update(outputs);
	return backend_.postprocess(outputs, guide_bgr);
}

void RVMMode::runPrefetchWorker(size_t model_w, size_t model_h,
                                SingleSlotChannel<cv::Mat>& raw_ch,
                                SingleSlotChannel<TensorData>& tensor_ch,
                                StageAccumulator& preprocess_acc) {
	while (true) {
		// Block until a raw frame is available, or the channel is closed (EOF).
		auto frame_opt = raw_ch.pop();
		if (!frame_opt)
			break;  // raw_ch was closed by the main thread — no more frames

		ManualTimer t;
		t.start();
		// BGR frame → letterbox-resized float32 tensor ready for inference.
		// prefetch_frontend_ is a separate ImageFrontend instance so this thread
		// never races with the main thread's frontend_ usage.
		auto tensor = prefetch_frontend_.preprocess(*frame_opt, model_w, model_h);
		preprocess_acc.record(t.stop());

		tensor_ch.push(std::move(tensor));
	}
	// Signal EOF downstream: tensor_ch.pop() in the main loop returns nullopt.
	tensor_ch.close();
}


void RVMMode::compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
                                const cv::Mat& alpha_8u, const cv::Mat& bg_bgr) {
	if (!writer.isOpened() || alpha_8u.empty()) {
		return;
	}

	cv::Mat alpha_1ch;
	if (alpha_8u.channels() == 1) {
		alpha_1ch = alpha_8u;
	} else {
		cv::cvtColor(alpha_8u, alpha_1ch, cv::COLOR_BGR2GRAY);
	}

	if (alpha_1ch.size() != frame.size()) {
		cv::resize(alpha_1ch, alpha_1ch, frame.size(), 0, 0, cv::INTER_LINEAR);
	}

	cv::Mat alpha_f32;
	alpha_1ch.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);

	cv::Mat alpha_3ch;
	cv::cvtColor(alpha_f32, alpha_3ch, cv::COLOR_GRAY2BGR);

	cv::Mat fg_f32, bg_f32, composed_f32, composed_8u;
	frame.convertTo(fg_f32, CV_32FC3, 1.0 / 255.0);
	bg_bgr.convertTo(bg_f32, CV_32FC3, 1.0 / 255.0);

	composed_f32 = alpha_3ch.mul(fg_f32) + (cv::Scalar(1.0, 1.0, 1.0) - alpha_3ch).mul(bg_f32);
	composed_f32.convertTo(composed_8u, CV_8UC3, 255.0);

	writer.write(composed_8u);
}

RvmRunSetup RVMMode::prepareRun(InferenceEngine* engine, const std::string& model_path,
                                const std::string& output_bin_path,
                                const std::string& background_path, bool timing_enabled) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	{
		ScopedTimer t("runRVM: model load", timing_enabled, logger, kRvmModuleName);
		engine->setOutputBinPath(output_bin_path);
		engine->load(model_path);
	}

	RvmRunSetup setup;
	setup.model_input_height = engine->getInputHeight() > 0 ? engine->getInputHeight() : 288;
	setup.model_input_width  = engine->getInputWidth()  > 0 ? engine->getInputWidth()  : 512;

	initRecurrentStates(setup.model_input_height, setup.model_input_width);

	frontend_.setOutputBinPath(output_bin_path);
	prefetch_frontend_.setOutputBinPath(output_bin_path);

	backend_.setOutputPath(output_bin_path);
	backend_.setBackgroundPath(background_path);
	// Attach Guided Filter post-processor.
	// Tuning: radius=8 (wide snap range for 1080p upscaled from 512px),
	//         epsilon=1e-4, threshold=0.4 (harden soft alpha before GF),
	//         erode_iters=2 (compensate AI mask's ~3-5px outward bias),
	//         src_blur_ksize=5 (smooth binary edge before GF snaps).
	backend_.setPostProcessor(
	    std::make_shared<GuidedFilterPostProcessor>(8, 1e-4, 0.4f, 2, 5));

	return setup;
}

int RVMMode::run(InferenceEngine* engine, std::unique_ptr<InputSource> input_source,
                 const std::string& model_path, const std::string& output_bin_path,
                 const std::string& background_path, bool timing_enabled) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// Stash paths as member variables so helper methods (e.g. future per-frame
	// post-processing) can reach them without extra parameters.
	output_bin_path_ = output_bin_path;
	background_path_ = background_path;

	ScopedTimer run_rvm_timer("runRVM total", timing_enabled, logger, kRvmModuleName);

	// -------------------------------------------------------------------------
	// 1st — Setup phase
	//    - load the model into the inference engine
	//    - query the model's expected input resolution (fall back to 288×512 if
	//      the engine returns 0, which can happen with dynamic-shape ONNX models)
	//    - initialise the four RNN hidden-state tensors (r1i–r4i) to zero
	//    - wire output / background paths into frontend and backend members
	// -------------------------------------------------------------------------
	const RvmRunSetup setup = prepareRun(engine, model_path, output_bin_path, background_path, timing_enabled);
	const size_t model_input_height = setup.model_input_height;
	const size_t model_input_width  = setup.model_input_width;

	// -------------------------------------------------------------------------
	// 2nd — Video I/O setup
	//    - read source dimensions and fps from the InputSource abstraction
	//      (works for both Mp4InputSource and any future camera/IPC source)
	//    - open the VideoWriter; if it fails, compositeAndWrite() is a no-op
	//    - load or synthesise a solid-colour background for alpha compositing
	// -------------------------------------------------------------------------
	const int src_width    = input_source->width();
	const int src_height   = input_source->height();
	const double src_fps   = input_source->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;
	const std::string output_video_path = output_bin_path + "/output_composited.mp4";

	cv::VideoWriter video_writer;
	openVideoWriter(video_writer, output_video_path, src_width, src_height, output_fps);

	cv::Mat bg_bgr = loadOrCreateBackground(src_width, src_height);

	// -------------------------------------------------------------------------
	// 3rd — Dual-buffer prefetch worker (producer thread)
	//
	//    Two single-slot channels decouple three stages of work:
	//      raw_ch    : decoded BGR frames   (written by main, read by worker)
	//      tensor_ch : preprocessed tensors (written by worker, read by main)
	//
	//    The worker loop:
	//      a. block on raw_ch.pop() — waits until main pushes the next raw frame
	//      b. call prefetch_frontend_.preprocess() — resize + normalise on a
	//         separate CPU thread, so the GPU/NPU is never idle waiting for data
	//      c. push the result onto tensor_ch for the main thread to consume
	//      d. close tensor_ch when raw_ch is closed (signals EOF to main)
	//
	//    Timing is accumulated in preprocess_acc and reported after the loop.
	// -------------------------------------------------------------------------
	SingleSlotChannel<cv::Mat>    raw_ch;
	SingleSlotChannel<TensorData> tensor_ch;

	StageAccumulator preprocess_acc("worker::preprocess");
	StageAccumulator infer_acc("main::infer+composite");

	std::thread prefetch_worker(&RVMMode::runPrefetchWorker, this,
	                             model_input_width, model_input_height,
	                             std::ref(raw_ch), std::ref(tensor_ch),
	                             std::ref(preprocess_acc));

	// Seed the pipeline with the very first frame before entering the main loop.
	cv::Mat current_frame;
	if (!input_source->read(current_frame)) {
		logger.Info("No frames to process.", kRvmModuleName);
		raw_ch.close();         // unblock worker so it can exit cleanly
		prefetch_worker.join();
		return 0;
	}
	raw_ch.push(current_frame);  // worker starts preprocessing frame 0 immediately

	int frame_count = 0;

	// -------------------------------------------------------------------------
	// 4th — Main inference loop  (consumer side of the dual-buffer pipeline)
	//
	//    Each iteration:
	//      a. check for SIGINT — if received, drain and stop gracefully
	//      b. pop the preprocessed tensor for the current frame (may block
	//         briefly if the worker is still preprocessing)
	//      c. read the *next* raw frame from the source and push it to raw_ch
	//         so the worker can preprocess it while we run inference on the
	//         current tensor (this is the overlap that hides decode latency)
	//      d. run inference → postprocess → composite → write to video
	//      e. advance current_frame and frame_count; stop when no next frame
	// -------------------------------------------------------------------------
	while (true) {
		// Graceful SIGINT handling: close the feed channel so the worker thread
		// can exit, then break out — the main thread will join and flush below.
		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing video at frame " +
			                std::to_string(frame_count) + ".",
			            kRvmModuleName);
			raw_ch.close();
			break;
		}

		// Block until the worker has finished preprocessing the current frame.
		auto tensor_opt = tensor_ch.pop();
		if (!tensor_opt)
			break;  // channel closed (worker exited after last frame)

		logger.Info("=== RVM Frame " + std::to_string(frame_count + 1) + " ===", kRvmModuleName);

		// Read the frame that comes *after* the one we are about to infer,
		// and hand it to the worker immediately so preprocessing overlaps with
		// inference below (the dual-buffer trick).
		cv::Mat next_frame;
		bool has_next = input_source->read(next_frame);
		if (has_next) {
			raw_ch.push(next_frame);
		} else {
			raw_ch.close();  // no more frames — worker will drain and close tensor_ch
		}

		// Run inference on the current tensor, postprocess the alpha matte,
		// and composite + write the output frame to the video file.
		ManualTimer infer_t;
		infer_t.start();
		cv::Mat alpha_8u = inferOneFrame(engine, *tensor_opt, current_frame);
		compositeAndWrite(video_writer, current_frame, alpha_8u, bg_bgr);
		infer_acc.record(infer_t.stop());

		current_frame = std::move(next_frame);  // advance sliding window
		frame_count++;

		if (!has_next)
			break;
	}

	// Wait for the worker thread to finish before destroying the channels.
	prefetch_worker.join();

	preprocess_acc.report(timing_enabled, logger, kRvmModuleName);
	infer_acc.report(timing_enabled, logger, kRvmModuleName);

	if (video_writer.isOpened()) {
		video_writer.release();
		logger.Info("Video compositing complete: " + std::to_string(frame_count) +
		                " frames written to " + output_video_path,
		            kRvmModuleName);
	}
	logger.Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count),
	            kRvmModuleName);
	return 0;
}

