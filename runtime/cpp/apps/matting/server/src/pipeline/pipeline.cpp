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
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "common/common-define.h"
#include "pipeline/backend/backend.h"
#include "pipeline/backend/post-processor/guided-filter-post-processor.h"
#include "pipeline/frontend/frontend.h"
#include "Utils/timing/timer.h"

using arcforge::utils::timing::ManualTimer;
using arcforge::utils::timing::ScopedTimer;
using arcforge::utils::timing::StageAccumulator;

// ---------------------------------------------------------------------------
// Backend-specific includes — ONLY here, nowhere else in the pipeline.
// Controlled by INFERENCE_BACKEND_* macros set by CMake (libs/runtime).
// ---------------------------------------------------------------------------
#if defined(INFERENCE_BACKEND_RKNN_ZEROCOPY)
#include "pipeline/inference-engine/rknn/rknn-zero-copy.h"
#elif defined(INFERENCE_BACKEND_RKNN_NON_ZEROCOPY)
#include "pipeline/inference-engine/rknn/rknn-non-zero-copy.h"
#else
#include "pipeline/inference-engine/onnx/onnx.h"
#endif

// Reference the global stop signal from main-server.cpp (set by SIGINT handler)
extern std::atomic<bool> g_stop_signal_received;

constexpr float kDownsampleRatio = 0.25f;

// ============================================================================
// SingleSlotChannel<T>
//
// A minimal blocking single-slot channel for producer/consumer synchronisation.
// Capacity = 1: producer blocks until consumer has picked up the previous item.
//
// Usage in prefetch pipeline:
//   Main thread  → push(raw_frame)    // blocks if slot full (consumer busy)
//   Worker thread→ pop()              // blocks until slot filled; nullopt = EOF
//   Worker thread→ push(tensor)
//   Main thread  → pop()              // blocks until tensor ready
//
// Design constraints:
//   - Header-only, no external dependencies
//   - Movable T only (cv::Mat / TensorData both qualify)
//   - close() signals EOF: pop() returns std::nullopt
// ============================================================================
template <typename T>
class SingleSlotChannel {
   public:
    // Push an item. Blocks if the slot is already occupied.
    // Returns false if the channel has been closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_empty_.wait(lk, [this] { return !has_item_ || closed_; });
        if (closed_) return false;
        item_     = std::move(item);
        has_item_ = true;
        cv_full_.notify_one();
        return true;
    }

    // Pop an item. Blocks until one is available.
    // Returns std::nullopt if the channel is closed and empty.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_full_.wait(lk, [this] { return has_item_ || closed_; });
        if (!has_item_) return std::nullopt;
        T out      = std::move(*item_);
        item_      = std::nullopt;
        has_item_  = false;
        cv_empty_.notify_one();
        return out;
    }

    // Signal EOF. Any blocked pop() will return nullopt after the last item.
    void close() {
        std::unique_lock<std::mutex> lk(mtx_);
        closed_ = true;
        cv_full_.notify_all();
        cv_empty_.notify_all();
    }

   private:
    std::mutex              mtx_;
    std::condition_variable cv_empty_;   // signalled when slot becomes empty
    std::condition_variable cv_full_;    // signalled when slot becomes full
    std::optional<T>        item_;
    bool                    has_item_ = false;
    bool                    closed_   = false;
};

// ============================================================================
// Factory — the single location that knows which concrete engine to build.
// All callers use InferenceEngine* and never mention the concrete type.
// ============================================================================
std::unique_ptr<InferenceEngine> Pipeline::make_engine() {
#if defined(INFERENCE_BACKEND_RKNN_ZEROCOPY)
 	return std::make_unique<InferenceEngineRKNNZeroCP>();
#elif defined(INFERENCE_BACKEND_RKNN_NON_ZEROCOPY)
 	return std::make_unique<InferenceEngineRKNN>();
#else
 	return std::make_unique<InferenceEngineONNX>();
#endif
}

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
	this->input_source_    = std::move(input_source);
	this->model_path_      = model_path;
	this->output_bin_path_ = output_bin_path;
	this->background_path_ = background_path;
	this->model_type_      = model_type;
	this->engine_          = make_engine();
}

void Pipeline::init(const std::string& input_image_path, const std::string& model_path,
                    const std::string& output_bin_path, const std::string& background_path,
                    ModelType model_type) {
	this->input_image_path_ = input_image_path;
	this->model_path_       = model_path;
	this->output_bin_path_  = output_bin_path;
	this->background_path_  = background_path;
	this->model_type_       = model_type;
	this->engine_           = make_engine();
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

	// Top-level timer: covers the entire pipeline from model load to final output.
	ScopedTimer run_timer("Pipeline::run() total", timing_enabled_, logger, kcurrent_module_name);

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
// inputs = { src }  (1 tensor)
// outputs = { pha } (1 tensor)
// ============================================================================
int Pipeline::runMODNet() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	MattingBackend backend;

	// 1. Load model
	{
		ScopedTimer t("runMODNet: model load", timing_enabled_, logger, kcurrent_module_name);
		engine_->setOutputBinPath(output_bin_path_);
		engine_->load(model_path_);
	}

	const size_t model_input_height = engine_->getInputHeight() > 0 ? engine_->getInputHeight() : 512;
	const size_t model_input_width  = engine_->getInputWidth()  > 0 ? engine_->getInputWidth()  : 512;

	// 2. Frontend: preprocess image
	TensorData src;
	{
		ScopedTimer t("runMODNet: preprocess", timing_enabled_, logger, kcurrent_module_name);
		frontend_.setOutputBinPath(output_bin_path_);
		src = frontend_.preprocess(input_image_path_, model_input_width, model_input_height);
	}

	// 3. Inference: benchmark 10 iterations
	std::vector<TensorData> inputs = {src};
	std::vector<TensorData> outputs;

	{
		ScopedTimer bench_timer("runMODNet: benchmark 10x total", timing_enabled_, logger, kcurrent_module_name);
		for (int i = 0; i < 10; ++i) {
			auto start = std::chrono::high_resolution_clock::now();
			engine_->infer(inputs, outputs);
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> dur = end - start;
			logger.Info("[Performance Benchmark " + std::to_string(i + 1) +
			                "] Inference Engine [infer()] cost: " + std::to_string(dur.count()) + " ms.",
			            kcurrent_module_name);
		}
	}

	// 4. Backend: postprocess
	{
		ScopedTimer t("runMODNet: postprocess", timing_enabled_, logger, kcurrent_module_name);
		backend.setOutputPath(output_bin_path_);
		backend.setBackgroundPath(background_path_);
		backend.setForegroundImagePath(input_image_path_);
		backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));
		backend.postprocess(outputs);
	}

	return 0;
}

// ============================================================================
// RVM helpers
// ============================================================================

// --- initRVMRecurrentStates -------------------------------------------------
// Computes internal resolution and initialises the recurrent state manager.
//
// RVM MobileNetV3 with downsample_ratio=0.25:
//   internal = input * 0.25   (e.g. 288x512 → 72x128)
//   r1: [1,16, H/2,  W/2 ]
//   r2: [1,20, H/4,  W/4 ]
//   r3: [1,40, H/8,  W/8 ]
//   r4: [1,64, H/16, W/16]
// ----------------------------------------------------------------------------
void Pipeline::initRVMRecurrentStates(size_t model_input_height, size_t model_input_width) {
	// constexpr float kDownsampleRatio = 0.25f;
	const int64_t internal_h =
	    static_cast<int64_t>(static_cast<float>(model_input_height) * kDownsampleRatio);
	const int64_t internal_w =
	    static_cast<int64_t>(static_cast<float>(model_input_width) * kDownsampleRatio);

	auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

	state_mgr_.init(
	    {
	        {1, 16, ceil_div(internal_h, 2),  ceil_div(internal_w, 2)},   // r1: [1,16,36,64]
	        {1, 20, ceil_div(internal_h, 4),  ceil_div(internal_w, 4)},   // r2: [1,20,18,32]
	        {1, 40, ceil_div(internal_h, 8),  ceil_div(internal_w, 8)},   // r3: [1,40, 9,16]
	        {1, 64, ceil_div(internal_h, 16), ceil_div(internal_w, 16)},  // r4: [1,64, 5, 8]
	    },
	    {"r1i", "r2i", "r3i", "r4i"});
}

// --- openVideoWriter --------------------------------------------------------
bool Pipeline::openVideoWriter(cv::VideoWriter& writer, const std::string& path,
                               int width, int height, double fps) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
	if (!writer.isOpened()) {
		logger.Warning("Failed to open VideoWriter: " + path + ". Composited video will NOT be saved.",
		               kcurrent_module_name);
		return false;
	}
	logger.Info("VideoWriter opened: " + path + " (" + std::to_string(width) + "x" +
	                std::to_string(height) + " @ " + std::to_string(fps) + " fps)",
	            kcurrent_module_name);
	return true;
}

// --- loadOrCreateBackground -------------------------------------------------
cv::Mat Pipeline::loadOrCreateBackground(int width, int height) {
	cv::Mat bg;
	if (!background_path_.empty()) {
		bg = cv::imread(background_path_, cv::IMREAD_COLOR);
		if (!bg.empty()) {
			cv::resize(bg, bg, cv::Size(width, height));
			return bg;
		}
	}
	// Default: solid blue (easier to verify on green-background test videos)
	return cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 100, 0));
}

// --- inferOneFrame ----------------------------------------------------------
// Build inputs (src + recurrent states [+ downsample_ratio for ONNX]),
// run inference, update recurrent states, return alpha matte from backend.
cv::Mat Pipeline::inferOneFrame(const TensorData& src) {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// constexpr float kDownsampleRatio = 0.25f;

	std::vector<TensorData> inputs = {src};
	state_mgr_.inject(inputs);  // appends r1i, r2i, r3i, r4i

#if defined(INFERENCE_BACKEND_ONNX)
	// ONNX path: downsample_ratio is a runtime input tensor.
	// RKNN path: it is folded into a constant by ArcFoundry at conversion time.
	TensorData dsr;
	dsr.name  = "downsample_ratio";
	dsr.shape = {1};
	dsr.data  = {kDownsampleRatio};
	inputs.push_back(std::move(dsr));
#endif

	std::vector<TensorData> outputs;
	auto t0 = std::chrono::high_resolution_clock::now();
	engine_->infer(inputs, outputs);
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> dur = t1 - t0;
	logger.Info("infer() cost: " + std::to_string(dur.count()) + " ms.", kcurrent_module_name);

	state_mgr_.update(outputs);  // outputs[2..5] = r1o~r4o → r1i~r4i for next frame

	// backend_.postprocess returns alpha_8u (CV_8UC1, original resolution)
	return backend_.postprocess(outputs);
}

// --- compositeAndWrite ------------------------------------------------------
// Alpha-blend frame onto background and push to VideoWriter.
void Pipeline::compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
                                 const cv::Mat& alpha_8u, const cv::Mat& bg_bgr) {
	if (!writer.isOpened() || alpha_8u.empty()) {
		return;
	}

	// Ensure single-channel alpha
	cv::Mat alpha_1ch;
	if (alpha_8u.channels() == 1) {
		alpha_1ch = alpha_8u;
	} else {
		cv::cvtColor(alpha_8u, alpha_1ch, cv::COLOR_BGR2GRAY);
	}

	// Resize alpha to match source frame if needed
	if (alpha_1ch.size() != frame.size()) {
		cv::resize(alpha_1ch, alpha_1ch, frame.size(), 0, 0, cv::INTER_LINEAR);
	}

	// Float blending: composite = alpha * fg + (1 - alpha) * bg
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

// ============================================================================
// RVM path — multi-frame with recurrent state management + dual-buffer prefetch
//
// ONNX:  inputs = { src, r1i, r2i, r3i, r4i, downsample_ratio }  (6 tensors)
// RKNN:  inputs = { src, r1i, r2i, r3i, r4i }                    (5 tensors)
// outputs = { fgr, pha, r1o, r2o, r3o, r4o }                     (6 tensors)
// ============================================================================
int Pipeline::runRVM() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// Full-pipeline wall-clock timer (covers load → last frame → file release).
	ScopedTimer run_rvm_timer("runRVM total", timing_enabled_, logger, kcurrent_module_name);

	// 1. Load model and query input dimensions
	{
		ScopedTimer t("runRVM: model load", timing_enabled_, logger, kcurrent_module_name);
		engine_->setOutputBinPath(output_bin_path_);
		engine_->load(model_path_);
	}

	const size_t model_input_height = engine_->getInputHeight() > 0 ? engine_->getInputHeight() : 288;
	const size_t model_input_width  = engine_->getInputWidth()  > 0 ? engine_->getInputWidth()  : 512;

	// 2. Recurrent state manager
	initRVMRecurrentStates(model_input_height, model_input_width);

	// 3. Frontend setup
	frontend_.setOutputBinPath(output_bin_path_);

	// 4. Backend setup (no GuidedFilter in video mode — would hurt throughput)
	backend_.setOutputPath(output_bin_path_);
	backend_.setBackgroundPath(background_path_);

	// 5. Video output
	const int    src_width  = input_source_->width();
	const int    src_height = input_source_->height();
	const double src_fps    = input_source_->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;
	const std::string output_video_path = output_bin_path_ + "/output_composited.mp4";

	cv::VideoWriter video_writer;
	openVideoWriter(video_writer, output_video_path, src_width, src_height, output_fps);

	cv::Mat bg_bgr = loadOrCreateBackground(src_width, src_height);

	// -----------------------------------------------------------------------
	// 6. Persistent prefetch worker
	//
	// Two channels connect main thread ↔ worker:
	//   raw_ch:    main → worker   (cv::Mat frames, one at a time)
	//   tensor_ch: worker → main   (preprocessed TensorData)
	//
	// Main thread reads the next raw frame, pushes it to raw_ch, then
	// calls infer() on the *current* tensor while the worker preprocesses
	// the next frame concurrently.  No per-frame thread creation overhead.
	//
	// Timing:
	//   preprocess_acc  — records each frame's preprocess cost (worker thread)
	//   infer_acc       — records each frame's infer+composite cost (main thread)
	// -----------------------------------------------------------------------
	SingleSlotChannel<cv::Mat>    raw_ch;
	SingleSlotChannel<TensorData> tensor_ch;

	// Accumulators are declared here so both threads can write to them,
	// and the main thread can call report() after join().
	StageAccumulator preprocess_acc("worker::preprocess");
	StageAccumulator infer_acc("main::infer+composite");

	std::thread prefetch_worker([this, model_input_width, model_input_height,
	                             &raw_ch, &tensor_ch, &preprocess_acc]() {
		while (true) {
			auto frame_opt = raw_ch.pop();
			if (!frame_opt) break;                          // EOF sentinel
			ManualTimer t;
			t.start();
			auto tensor = prefetch_frontend_.preprocess(*frame_opt, model_input_width, model_input_height);
			preprocess_acc.record(t.stop());
			tensor_ch.push(std::move(tensor));
		}
		tensor_ch.close();
	});

	// Bootstrap: push first frame so worker starts immediately
	cv::Mat current_frame;
	if (!input_source_->read(current_frame)) {
		logger.Info("No frames to process.", kcurrent_module_name);
		raw_ch.close();
		prefetch_worker.join();
		return 0;
	}
	raw_ch.push(current_frame);

	int frame_count = 0;

	// 7. Main loop
	while (true) {
		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing video at frame " +
			                std::to_string(frame_count) + ".",
			            kcurrent_module_name);
			raw_ch.close();
			break;
		}

		// STEP A: collect the preprocessed tensor for the current frame
		auto tensor_opt = tensor_ch.pop();
		if (!tensor_opt) break;                             // worker finished

		logger.Info("=== RVM Frame " + std::to_string(frame_count + 1) + " ===",
		            kcurrent_module_name);

		// STEP B: read next raw frame and kick off prefetch in worker
		//         (worker preprocesses next frame concurrently with infer below)
		cv::Mat next_frame;
		bool    has_next = input_source_->read(next_frame);
		if (has_next) {
			raw_ch.push(next_frame);
		} else {
			raw_ch.close();
		}

		// STEP C: infer current tensor + composite (overlaps with worker's preprocess)
		ManualTimer infer_t;
		infer_t.start();
		cv::Mat alpha_8u = inferOneFrame(*tensor_opt);
		compositeAndWrite(video_writer, current_frame, alpha_8u, bg_bgr);
		infer_acc.record(infer_t.stop());

		current_frame = std::move(next_frame);              // advance frame pointer
		frame_count++;

		if (!has_next) break;
	}

	prefetch_worker.join();

	// 8. Per-stage timing summary (printed after all threads are done)
	preprocess_acc.report(timing_enabled_, logger, kcurrent_module_name);
	infer_acc.report(timing_enabled_, logger, kcurrent_module_name);

	// 9. Finalise
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

// ============================================================================
// RVM path — single-picture loop test (integration test / development only)
// ============================================================================
int Pipeline::runRVM_CV_SinglePicture() {
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	MattingBackend backend;

	engine_->setOutputBinPath(output_bin_path_);
	engine_->load(model_path_);

	const size_t model_input_height = engine_->getInputHeight() > 0 ? engine_->getInputHeight() : 288;
	const size_t model_input_width  = engine_->getInputWidth()  > 0 ? engine_->getInputWidth()  : 512;

	initRVMRecurrentStates(model_input_height, model_input_width);

	frontend_.setOutputBinPath(output_bin_path_);
	TensorData src = frontend_.preprocess(input_image_path_, model_input_width, model_input_height);

	backend.setOutputPath(output_bin_path_);
	backend.setBackgroundPath(background_path_);
	backend.setForegroundImagePath(input_image_path_);
	backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));

	constexpr int kNumTestFrames = 5;
	for (int frame = 0; frame < kNumTestFrames; ++frame) {
		logger.Info("=== RVM Frame " + std::to_string(frame + 1) + "/" +
		                std::to_string(kNumTestFrames) + " ===",
		            kcurrent_module_name);

		// constexpr float kDownsampleRatio = 0.25f;
		std::vector<TensorData> inputs   = {src};
		state_mgr_.inject(inputs);

#if defined(INFERENCE_BACKEND_ONNX)
		TensorData dsr;
		dsr.name  = "downsample_ratio";
		dsr.shape = {1};
		dsr.data  = {kDownsampleRatio};
		inputs.push_back(std::move(dsr));
#endif

		std::vector<TensorData> outputs;
		auto t0 = std::chrono::high_resolution_clock::now();
		engine_->infer(inputs, outputs);
		auto t1 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> dur = t1 - t0;
		logger.Info("[RVM Frame " + std::to_string(frame + 1) +
		                "] infer() cost: " + std::to_string(dur.count()) + " ms.",
		            kcurrent_module_name);

		state_mgr_.update(outputs);

		if (frame == kNumTestFrames - 1) {
			backend.postprocess(outputs);
		}
	}

	return 0;
}
