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
#include <cstring>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include "RGAKit/rga_operation.h"
#include "Utils/timing/timer.h"
#include "common/common-define.h"
#include "pipeline/single-slot-channel.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;

using helmsman::utils::timing::ManualTimer;
using helmsman::utils::timing::ScopedTimer;
using helmsman::utils::timing::StageAccumulator;

extern std::atomic<bool> g_stop_signal_received;

inline constexpr std::string_view kRvmModuleName = "RVMMode";

void RVMMode::initRecurrentStates(InferenceEngine* engine) {
	// Try to get recurrent state shapes from the engine (RKNN reports actual shapes).
	auto shapes = engine->getRecurrentStateShapes();
	if (shapes.size() == 4) {
		auto& logger = helmsman::utils::Logger::GetInstance();
		logger.Info("Using model-reported recurrent state shapes", kRvmModuleName);
		state_mgr_.init(shapes, {"r1i", "r2i", "r3i", "r4i"});
	} else {
		// Fallback: use [1,1,1,1] (ONNX broadcasts, RKNN first frame uses zeros).
		state_mgr_.init({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
		                {"r1i", "r2i", "r3i", "r4i"});
	}
}

bool RVMMode::openVideoWriter(cv::VideoWriter& writer, const std::string& path, int width,
                              int height, double fps) {
	auto& logger = helmsman::utils::Logger::GetInstance();
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
	// Default fallback background: BGR(155,255,120) = RGB(120,255,155), matches e.py baseline
	return cv::Mat(height, width, CV_8UC3, cv::Scalar(155, 255, 120));
}

cv::Mat RVMMode::inferOneFrame(InferenceEngine* engine, const TensorData& src,
                               const cv::Mat& guide_bgr) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	std::vector<TensorData> inputs = {src};
	state_mgr_.inject(inputs);

	if (engine->needsDownsampleRatio()) {
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {dsr_};
		inputs.push_back(std::move(dsr));
	}

	std::vector<TensorData> outputs;
	auto t0 = std::chrono::high_resolution_clock::now();
	engine->infer(inputs, outputs);
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> dur = t1 - t0;
	logger.Info("infer() cost: " + std::to_string(dur.count()) + " ms.", kRvmModuleName);

	state_mgr_.update(outputs);
	return backend_.postprocess(outputs, guide_bgr);
}

void RVMMode::runPrefetchWorker(size_t model_w, size_t model_h, SingleSlotChannel<cv::Mat>& raw_ch,
                                SingleSlotChannel<TensorData>& tensor_ch,
                                StageAccumulator& acc_1st_worker_preprocess) {
	while (true) {
		// Block until a raw frame is available, or the channel is closed (EOF).
		auto frame_opt = raw_ch.pop();
		if (!frame_opt)
			break;  // raw_ch was closed by the main thread — no more frames

		ManualTimer t;
		t.start();
		// BGR frame → letterbox-resized float32 tensor ready for inference.
		// frontend_->preprocess() is thread-safe (Preprocessor is stateless).
		auto tensor = frontend_->preprocess(*frame_opt, model_w, model_h);
		acc_1st_worker_preprocess.record(t.stop());

		tensor_ch.push(std::move(tensor));
	}
	// Signal EOF downstream: tensor_ch.pop() in the main loop returns nullopt.
	tensor_ch.close();
}

double RVMMode::compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
                                  const cv::Mat& alpha_8u) {
	if (!writer.isOpened() || alpha_8u.empty())
		return 0.0;

	ManualTimer total_t;
	total_t.start();

	const int model_h = bg_model_u8_.rows;
	const int model_w = bg_model_u8_.cols;
	ManualTimer t;

	// 1. Resize alpha (CPU — RGA doesn't support single-channel YUV400 format)
	cv::Mat alpha_model;
	t.start();
	{
		alpha_model.create(model_h, model_w, CV_8UC1);
		cv::resize(alpha_8u, alpha_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
	}
	acc_resize_alpha_.record(t.stop());

	// 2. Resize frame (RGA hardware)
	cv::Mat frame_model;
	t.start();
	{
		frame_model.create(model_h, model_w, CV_8UC3);
		ImageDescriptor src(frame.data, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(frame_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		if (!rga_resize_->Execute(src, dst)) {
			cv::resize(frame, frame_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
		}
	}
	acc_resize_frame_.record(t.stop());

	// 3. CPU alpha blend: fg_bgr * alpha + bg_bgr * (1-alpha) → composed_bgr
	cv::Mat composed_model(model_h, model_w, CV_8UC3);
	t.start();
	{
		const int pixels = model_h * model_w;
		const uint8_t* fg_ptr = frame_model.ptr<uint8_t>(0);
		const uint8_t* bg_ptr = bg_model_u8_.ptr<uint8_t>(0);
		const uint8_t* a_ptr = alpha_model.ptr<uint8_t>(0);
		uint8_t* out = composed_model.ptr<uint8_t>(0);
		for (int i = 0; i < pixels; ++i) {
			const uint16_t alpha = a_ptr[i];
			const uint16_t inv = 255 - alpha;
			out[0] = static_cast<uint8_t>((fg_ptr[0] * alpha + bg_ptr[0] * inv + 1 +
			                               ((fg_ptr[0] * alpha + bg_ptr[0] * inv) >> 8)) >>
			                              8);
			out[1] = static_cast<uint8_t>((fg_ptr[1] * alpha + bg_ptr[1] * inv + 1 +
			                               ((fg_ptr[1] * alpha + bg_ptr[1] * inv) >> 8)) >>
			                              8);
			out[2] = static_cast<uint8_t>((fg_ptr[2] * alpha + bg_ptr[2] * inv + 1 +
			                               ((fg_ptr[2] * alpha + bg_ptr[2] * inv) >> 8)) >>
			                              8);
			fg_ptr += 3;
			bg_ptr += 3;
			out += 3;
		}
	}
	acc_blend_.record(t.stop());

	// 5. Upscale (RGA hardware)
	cv::Mat composed_full;
	t.start();
	{
		composed_full.create(frame.rows, frame.cols, CV_8UC3);
		ImageDescriptor src(composed_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(composed_full.data, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
		if (!rga_resize_->Execute(src, dst)) {
			cv::resize(composed_model, composed_full, frame.size(), 0, 0, cv::INTER_LINEAR);
		}
	}
	acc_upscale_.record(t.stop());

	// 6. Write
	t.start();
	writer.write(composed_full);
	acc_writer_.record(t.stop());

	return total_t.elapsed_ms();
}

bool RVMMode::initOutputDma(int src_width, int src_height) {
	auto& logger = helmsman::utils::Logger::GetInstance();
	const size_t buf_bytes =
	    static_cast<size_t>(src_width) * static_cast<size_t>(src_height) * 3;  // BGR888
	dma_output_buf_ = helmsman::dmakit::DmaBuffer::Allocate(buf_bytes);
	if (!dma_output_buf_) {
		logger.Warning("Failed to allocate DMA output buffer (" + std::to_string(buf_bytes) +
		                   " bytes). Falling back to VideoWriter.",
		               kRvmModuleName);
		return false;
	}
	// Map once so we can use the virtual address as RGA destination.
	dma_output_buf_->map();
	logger.Info("DMA output buffer allocated: fd=" + std::to_string(dma_output_buf_->fd()) +
	                ", size=" + std::to_string(buf_bytes) + " bytes (" + std::to_string(src_width) +
	                "x" + std::to_string(src_height) + " BGR)",
	            kRvmModuleName);
	return true;
}

int RVMMode::compositeToDma(const cv::Mat& frame, const cv::Mat& alpha_8u) {
	if (!dma_output_buf_ || alpha_8u.empty())
		return -1;

	const int model_h = bg_model_u8_.rows;
	const int model_w = bg_model_u8_.cols;
	ManualTimer t;

	// 1. Resize alpha (CPU — RGA doesn't support single-channel YUV400 format)
	cv::Mat alpha_model;
	t.start();
	{
		alpha_model.create(model_h, model_w, CV_8UC1);
		cv::resize(alpha_8u, alpha_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
	}
	acc_resize_alpha_.record(t.stop());

	// 2. Resize frame (RGA hardware)
	cv::Mat frame_model;
	t.start();
	{
		frame_model.create(model_h, model_w, CV_8UC3);
		ImageDescriptor src(frame.data, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(frame_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		if (!rga_resize_->Execute(src, dst)) {
			cv::resize(frame, frame_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
		}
	}
	acc_resize_frame_.record(t.stop());

	// 3. CPU alpha blend: fg_bgr * alpha + bg_bgr * (1-alpha) → composed_bgr
	cv::Mat composed_model(model_h, model_w, CV_8UC3);
	t.start();
	{
		const int pixels = model_h * model_w;
		const uint8_t* fg_ptr = frame_model.ptr<uint8_t>(0);
		const uint8_t* bg_ptr = bg_model_u8_.ptr<uint8_t>(0);
		const uint8_t* a_ptr = alpha_model.ptr<uint8_t>(0);
		uint8_t* out = composed_model.ptr<uint8_t>(0);
		for (int i = 0; i < pixels; ++i) {
			const uint16_t alpha = a_ptr[i];
			const uint16_t inv = 255 - alpha;
			out[0] = static_cast<uint8_t>((fg_ptr[0] * alpha + bg_ptr[0] * inv + 1 +
			                               ((fg_ptr[0] * alpha + bg_ptr[0] * inv) >> 8)) >>
			                              8);
			out[1] = static_cast<uint8_t>((fg_ptr[1] * alpha + bg_ptr[1] * inv + 1 +
			                               ((fg_ptr[1] * alpha + bg_ptr[1] * inv) >> 8)) >>
			                              8);
			out[2] = static_cast<uint8_t>((fg_ptr[2] * alpha + bg_ptr[2] * inv + 1 +
			                               ((fg_ptr[2] * alpha + bg_ptr[2] * inv) >> 8)) >>
			                              8);
			fg_ptr += 3;
			bg_ptr += 3;
			out += 3;
		}
	}
	acc_blend_.record(t.stop());

	// 5. Upscale directly into DMA buffer (RGA hardware, zero-copy)
	t.start();
	{
		void* dma_ptr = dma_output_buf_->map();
		if (dma_ptr) {
			ImageDescriptor src_desc(composed_model.data, model_w, model_h,
			                         RgaPixelFormat::kBgr888);
			ImageDescriptor dst_desc(dma_ptr, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
			if (!rga_resize_->Execute(src_desc, dst_desc)) {
				// Fallback: CPU upscale then memcpy to DMA
				cv::Mat composed_full;
				cv::resize(composed_model, composed_full, frame.size(), 0, 0, cv::INTER_LINEAR);
				memcpy(dma_ptr, composed_full.data,
				       composed_full.total() * composed_full.elemSize());
			}
		}
	}
	acc_upscale_.record(t.stop());

	return dma_output_buf_->fd();
}

double RVMMode::compositeToDrm(const cv::Mat& frame, const cv::Mat& alpha_8u, int panel_w,
                               int panel_h) {
	if (!drm_display_.IsOpen() || alpha_8u.empty())
		return 0.0;

	ManualTimer total_t;
	total_t.start();

	const int model_h = bg_model_u8_.rows;
	const int model_w = bg_model_u8_.cols;
	ManualTimer t;

	// 1. Resize alpha (CPU — RGA doesn't support single-channel YUV400 format)
	cv::Mat alpha_model;
	t.start();
	{
		alpha_model.create(model_h, model_w, CV_8UC1);
		cv::resize(alpha_8u, alpha_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
	}
	acc_resize_alpha_.record(t.stop());

	// 2. Resize frame (RGA hardware)
	cv::Mat frame_model;
	t.start();
	{
		frame_model.create(model_h, model_w, CV_8UC3);
		ImageDescriptor src(frame.data, frame.cols, frame.rows, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(frame_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		if (!rga_resize_->Execute(src, dst)) {
			cv::resize(frame, frame_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
		}
	}
	acc_resize_frame_.record(t.stop());

	// 3. CPU alpha blend: fg_bgr * alpha + bg_bgr * (1-alpha) → composed_bgr
	cv::Mat composed_model(model_h, model_w, CV_8UC3);
	t.start();
	{
		const int pixels = model_h * model_w;
		const uint8_t* fg_ptr = frame_model.ptr<uint8_t>(0);
		const uint8_t* bg_ptr = bg_model_u8_.ptr<uint8_t>(0);
		const uint8_t* a_ptr = alpha_model.ptr<uint8_t>(0);
		uint8_t* out = composed_model.ptr<uint8_t>(0);
		for (int i = 0; i < pixels; ++i) {
			const uint16_t alpha = a_ptr[i];
			const uint16_t inv = 255 - alpha;
			out[0] = static_cast<uint8_t>((fg_ptr[0] * alpha + bg_ptr[0] * inv + 1 +
			                               ((fg_ptr[0] * alpha + bg_ptr[0] * inv) >> 8)) >>
			                              8);
			out[1] = static_cast<uint8_t>((fg_ptr[1] * alpha + bg_ptr[1] * inv + 1 +
			                               ((fg_ptr[1] * alpha + bg_ptr[1] * inv) >> 8)) >>
			                              8);
			out[2] = static_cast<uint8_t>((fg_ptr[2] * alpha + bg_ptr[2] * inv + 1 +
			                               ((fg_ptr[2] * alpha + bg_ptr[2] * inv) >> 8)) >>
			                              8);
			fg_ptr += 3;
			bg_ptr += 3;
			out += 3;
		}
	}
	acc_blend_.record(t.stop());

	// 4. Upscale to panel size (RGA hardware)
	cv::Mat composed_panel;
	t.start();
	{
		composed_panel.create(panel_h, panel_w, CV_8UC3);
		ImageDescriptor src(composed_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(composed_panel.data, panel_w, panel_h, RgaPixelFormat::kBgr888);
		if (!rga_resize_->Execute(src, dst)) {
			cv::resize(composed_model, composed_panel, cv::Size(panel_w, panel_h), 0, 0,
			           cv::INTER_LINEAR);
		}
	}
	acc_upscale_.record(t.stop());

	// 5. BGR888 → XRGB8888 + ShowARGB
	t.start();
	{
		const int n_pixels = panel_w * panel_h;
		argb_buf_.resize(static_cast<size_t>(n_pixels) * 4);
		const uint8_t* bgr = composed_panel.ptr<uint8_t>(0);
		uint8_t* xrgb = argb_buf_.data();
		for (int i = 0; i < n_pixels; ++i) {
			xrgb[0] = bgr[0];  // B → B
			xrgb[1] = bgr[1];  // G → G
			xrgb[2] = bgr[2];  // R → R
			xrgb[3] = 0xFF;    // X (padding)
			bgr += 3;
			xrgb += 4;
		}
		drm_display_.ShowARGB(argb_buf_.data());
	}
	acc_drm_.record(t.stop());

	return total_t.elapsed_ms();
}

RvmRunSetup RVMMode::prepareRun(InferenceEngine* engine, const std::string& model_path,
                                const std::string& output_bin_path,
                                const std::string& background_path, bool timing_enabled) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	{
		ScopedTimer t("runRVM: model load", timing_enabled, logger, kRvmModuleName);
		engine->setOutputBinPath(output_bin_path);
		engine->load(model_path);
	}

	RvmRunSetup setup;
	setup.model_input_height = engine->getInputHeight() > 0 ? engine->getInputHeight() : 288;
	setup.model_input_width = engine->getInputWidth() > 0 ? engine->getInputWidth() : 512;

	initRecurrentStates(engine);

	backend_.setOutputPath(output_bin_path);
	backend_.setBackgroundPath(background_path);
	// Post-processor: disabled to match PyTorch baseline (no post-processing).
	// PyTorch inference: com = fgr * pha + bgr * (1 - pha) — raw model alpha, no GF.
	// Re-enable after confirming raw alpha quality: setPostProcessor(make_shared<GuidedFilterPostProcessor>(...))
	// backend_.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(8, 1e-4, 0.4f, 2, 5));

	return setup;
}

int RVMMode::run(InferenceEngine* engine, Frontend* frontend,
                 const std::string& model_path, const std::string& output_bin_path,
                 const std::string& background_path, bool timing_enabled, OutputMode output_mode) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// Stash paths as member variables so helper methods (e.g. future per-frame
	// post-processing) can reach them without extra parameters.
	output_bin_path_ = output_bin_path;
	background_path_ = background_path;
	frontend_ = frontend;

	ScopedTimer run_rvm_timer("runRVM total", timing_enabled, logger, kRvmModuleName);

	// =========================================================================
	// 1st — Setup phase
	//    - load the model into the inference engine
	//    - query the model's expected input resolution (fall back to 288×512 if
	//      the engine returns 0, which can happen with dynamic-shape ONNX models)
	//    - initialise the four RNN hidden-state tensors (r1i–r4i) to zero
	//    - wire output / background paths into frontend and backend members
	// =========================================================================
	const RvmRunSetup setup =
	    prepareRun(engine, model_path, output_bin_path, background_path, timing_enabled);
	const size_t model_input_height = setup.model_input_height;
	const size_t model_input_width = setup.model_input_width;

	// =========================================================================
	// 2nd — Video I/O setup
	//    - read source dimensions and fps from the InputSource abstraction
	//      (works for both Mp4InputSource and any future camera/IPC source)
	//    - open the VideoWriter; if it fails, compositeAndWrite() is a no-op
	//    - load or synthesise a solid-colour background for alpha compositing
	// =========================================================================
	const int src_width = frontend_->width();
	const int src_height = frontend_->height();
	dsr_ = 512.0f / static_cast<float>(std::max(src_width, src_height));
	// dance.mp4 1920×1080 → dsr_ = 512/1920 ≈ 0.2667
	const double src_fps = frontend_->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;
	const std::string output_video_path = output_bin_path + "/output_composited.mp4";

	// DMA zero-copy output disabled: experiment phase needs video file for quality comparison.
	// Re-enable after sweet-spot experiments: uncomment initOutputDma and restore the if-block.
	const bool use_dma_output = false;  // was: initOutputDma(src_width, src_height)
	cv::VideoWriter video_writer;
	int drm_panel_w = 0;
	int drm_panel_h = 0;
	if (output_mode == OutputMode::kDrm) {
		if (drm_display_.Init(src_width, src_height)) {
			std::tie(drm_panel_w, drm_panel_h) = drm_display_.PanelSize();
			logger.Info("DRM display initialized: panel " + std::to_string(drm_panel_w) + "x" +
			                std::to_string(drm_panel_h),
			            kRvmModuleName);
		} else {
			logger.Warning("DRM init failed. Falling back to mp4.", kRvmModuleName);
			output_mode = OutputMode::kMp4;
		}
	}
	if (output_mode == OutputMode::kMp4) {
		openVideoWriter(video_writer, output_video_path, src_width, src_height, output_fps);
	}

	cv::Mat bg_bgr = loadOrCreateBackground(src_width, src_height);

	// Pre-compute background at model resolution for fast compositing in compositeAndWrite().
	// Avoids converting bg from uint8→float32 at full resolution every frame.
	{
		cv::Mat bg_model;
		cv::resize(
		    bg_bgr, bg_model,
		    cv::Size(static_cast<int>(model_input_width), static_cast<int>(model_input_height)), 0,
		    0, cv::INTER_LINEAR);
		bg_model_u8_ = bg_model.clone();  // uint8 copy for fast compositing
	}

	// Create RGA hardware operations (stateless, reused every frame).
	// RGA resize replaces cv::resize for the downscale/upscale steps.
	// RGA composite replaces the CPU uint8 blend loop.
	rga_resize_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaResize>();

	// =========================================================================
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
	//    Timing is accumulated in acc_1st_worker_preprocess and reported after the loop.
	// =========================================================================
	SingleSlotChannel<cv::Mat> raw_ch;
	SingleSlotChannel<TensorData> tensor_ch;

	// -------------------------------------------------------------------------
	// Pipeline timing layout (s10 — full coverage)
	//
	// Per-frame wall clock breakdown
	//
	//   [main thread]                          [worker thread]
	//   acc_5th_main_loop_total  (whole iteration)
	//     ├── tensor_ch.pop()    ◄────────────  acc_1st_worker_preprocess
	//     │   (blocks if worker     pushes here   (run on worker:
	//     │    not done yet)                       BGR→tensor resize+norm)
	//     ├── acc_2nd_main_decode         ────────────►   raw_ch.pop()
	//     │   (read next frame                     (worker waits here)
	//     │    + push to raw_ch)
	//     ├── acc_3rd_main_infer           (NPU inference, current frame)
	//     └── acc_4th_main_composite            (composite + write, current frame)
	//             │
	//             ├── acc_resize_alpha_  (CPU resize alpha → model size)
	//             ├── acc_resize_frame_  (RGA resize frame → model size)
	//             ├── acc_blend_         (CPU alpha blend at model size)
	//             ├── acc_upscale_       (RGA upscale composed → full size)
	//             └── acc_writer_        (VideoWriter::write — see NOTE below)
	//
	// Whole-run timers (overlap with the above; cheap, kept for context)
	//
	//   ScopedTimer "Pipeline::run() total"   (pipeline.cpp)  — outermost
	//   ScopedTimer "runRVM total"            (this fn)        — wraps loop
	//   ScopedTimer "runRVM: model load"      (this fn)        — model load only
	//   [FPS]   line every 30 frames                            — moving fps
	//   [PerFrame] line every frame                             — infer + comp
	//
	// Identity (approx, ignoring tiny logging overhead):
	//   acc_5th_main_loop_total ≈ max(tensor_ch.pop wait, 0) + acc_2nd_main_decode + acc_3rd_main_infer + acc_4th_main_composite
	//   acc_4th_main_composite ≈ resize_alpha + resize_frame + blend + upscale + writer
	// -------------------------------------------------------------------------
	StageAccumulator acc_1st_worker_preprocess("1/5::worker::preprocess");
	StageAccumulator acc_2nd_main_decode("2/5::main::decode");
	StageAccumulator acc_3rd_main_infer("3/5::main::infer");
	StageAccumulator acc_4th_main_composite("4/5::main::composite");
	StageAccumulator acc_5th_main_loop_total("5/5::main::loop_total");

	// --------------------
	// Start the worker thread that runs the prefetch + preprocess loop, and
	std::thread prefetch_worker(&RVMMode::runPrefetchWorker, this, model_input_width,
	                            model_input_height, std::ref(raw_ch), std::ref(tensor_ch),
	                            std::ref(acc_1st_worker_preprocess));

	// --------------------
	// obtain the first raw frame and push it into thread-safe raw_ch
	cv::Mat current_frame;
	HardwareFrame current_hw_frame;
	if (!frontend_->readFrame(current_frame, current_hw_frame)) {
		logger.Info("No frames to process.", kRvmModuleName);
		raw_ch.close();  // unblock worker so it can exit cleanly
		prefetch_worker.join();
		return 0;
	}
	raw_ch.push(current_frame);  // worker starts preprocessing frame 0 immediately

	// =========================================================================
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
	// =========================================================================
	int frame_count = 0;
	auto fps_window_start = std::chrono::steady_clock::now();
	auto pipeline_start = fps_window_start;

	while (true) {

		// --------------------
		// start acc_5th_main_loop_total acuumulation here
		ManualTimer loop_t;  // accumulate for acc_5th_main_loop_total
		loop_t.start();

		// --------------------
		// Graceful SIGINT handling: close the feed channel so the worker thread
		// can exit, then break out — the main thread will join and flush below.
		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing video at frame " +
			                std::to_string(frame_count) + ".",
			            kRvmModuleName);
			raw_ch.close();
			break;
		}

		// --------------------
		// Block until the worker has finished preprocessing the current frame.
		auto tensor_opt = tensor_ch.pop();
		if (!tensor_opt) {
			break;  // channel closed (worker exited after last frame)
		}

		// --------------------
		// If we executes this far, we have the tensor data and the processing will be started imminently
		logger.Info("=== RVM Frame " + std::to_string(frame_count + 1) + " ===", kRvmModuleName);

		// --------------------
		// Read the frame that comes *after* the one we are about to infer,
		// and hand it to the worker immediately so preprocessing overlaps with
		// inference below (the dual-buffer trick).
		// s18: decode now goes through Frontend (OpenCV or MPP path).
		cv::Mat next_frame;
		HardwareFrame next_hw_frame;
		bool has_next;
		{
			ManualTimer decode_t;
			decode_t.start();
			has_next = frontend_->readFrame(next_frame, next_hw_frame);

			if (has_next) {
				raw_ch.push(next_frame);
			} else {
				raw_ch.close();  // no more frames — worker will drain and close tensor_ch
			}

			acc_2nd_main_decode.record(decode_t.stop());
		}

		// --------------- infer one frame ---------------
		// Run inference on the current tensor, postprocess the alpha matte,
		// and composite + write the output frame to the video file.
		ManualTimer infer_t;
		infer_t.start();
		cv::Mat alpha_8u = inferOneFrame(engine, *tensor_opt, current_frame);
		const double infer_ms = infer_t.stop();
		acc_3rd_main_infer.record(infer_ms);

		// --------------- composite one frame ---------------
		double comp_ms;
		if (use_dma_output) {
			ManualTimer dma_t;
			dma_t.start();
			const int output_fd = compositeToDma(current_frame, alpha_8u);
			comp_ms = dma_t.stop();
			if (frame_count == 0) {
				logger.Info("DMA output fd=" + std::to_string(output_fd), kRvmModuleName);
			}
		} else if (output_mode == OutputMode::kDrm) {
			comp_ms = compositeToDrm(current_frame, alpha_8u, drm_panel_w, drm_panel_h);
		} else {
			comp_ms = compositeAndWrite(video_writer, current_frame, alpha_8u);
		}
		acc_4th_main_composite.record(comp_ms);

		// --------------- end of one frame processing ---------------
		logger.Info("[PerFrame] frame=" + std::to_string(frame_count) +
		                "  infer=" + std::to_string(infer_ms) + "ms" +
		                "  composite=" + std::to_string(comp_ms) + "ms",
		            kRvmModuleName);

		// ---------------
		// FPS measurement: report every 30 frames
		if (frame_count % 30 == 0) {
			auto now = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration<double>(now - fps_window_start).count();
			logger.Info("[FPS] " + std::to_string(30.0 / elapsed) + " fps (last 30 frames in " +
			                std::to_string(elapsed) + "s)",
			            kRvmModuleName);
			fps_window_start = now;
		}

		// ---------------
		// s10: record per-iteration wall clock — must be the last thing before
		// loop exit / continue, so it covers everything done above.
		acc_5th_main_loop_total.record(loop_t.stop());

		// ---------------
		// echo the end message for current single frame
		logger.Info(" --- End of RVM Frame " + std::to_string(frame_count + 1) + " ---" + "\n",
		            kRvmModuleName);

		// ---------------
		// prepare for next frame(iteration)
		current_frame = std::move(next_frame);  // advance sliding window
		frame_count++;

		// ---------------
		// exit loop if no next frame; the worker thread will see the raw_ch close and exit cleanly
		if (!has_next) {
			break;
		}
	}

	// Wait for the worker thread to finish before destroying the channels.
	prefetch_worker.join();

	acc_1st_worker_preprocess.report(timing_enabled, logger, kRvmModuleName);
	acc_2nd_main_decode.report(timing_enabled, logger, kRvmModuleName);
	acc_3rd_main_infer.report(timing_enabled, logger, kRvmModuleName);
	acc_4th_main_composite.report(timing_enabled, logger, kRvmModuleName);
	acc_5th_main_loop_total.report(timing_enabled, logger, kRvmModuleName);

	acc_resize_alpha_.report(timing_enabled, logger, kRvmModuleName);
	acc_resize_frame_.report(timing_enabled, logger, kRvmModuleName);
	acc_blend_.report(timing_enabled, logger, kRvmModuleName);
	acc_upscale_.report(timing_enabled, logger, kRvmModuleName);
	// NOTE: acc_writer_ measures only the time VideoWriter::write() returns,
	// NOT actual encoder completion (FFmpeg buffers internally). Treat this
	// number as a lower bound for the true write cost.
	acc_writer_.report(timing_enabled, logger, kRvmModuleName);
	acc_drm_.report(timing_enabled, logger, kRvmModuleName);

	if (video_writer.isOpened()) {
		video_writer.release();
		logger.Info("Video compositing complete: " + std::to_string(frame_count) +
		                " frames written to " + output_video_path,
		            kRvmModuleName);
	}
	if (drm_display_.IsOpen()) {
		drm_display_.Close();
		logger.Info("DRM display closed after " + std::to_string(frame_count) + " frames.",
		            kRvmModuleName);
	}
	if (use_dma_output) {
		logger.Info("DMA output: " + std::to_string(frame_count) +
		                " frames composited to DMA fd=" + std::to_string(dma_output_buf_->fd()),
		            kRvmModuleName);
	}
	if (frame_count > 0) {
		const double total_elapsed =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - pipeline_start)
		        .count();
		const double avg_fps = static_cast<double>(frame_count) / total_elapsed;
		logger.Info("[FPS] Total: " + std::to_string(frame_count) + " frames in " +
		                std::to_string(total_elapsed) + "s = " + std::to_string(avg_fps) + " fps",
		            kRvmModuleName);
	}
	logger.Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count),
	            kRvmModuleName);
	return 0;
}
