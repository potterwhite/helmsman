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
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;

using helmsman::utils::timing::ManualTimer;
using helmsman::utils::timing::ScopedTimer;
using helmsman::utils::timing::StageAccumulator;

extern std::atomic<bool> g_stop_signal_received;

inline constexpr std::string_view kRvmModuleName = "RVMMode";

// Default model input dimensions (fallback when engine reports 0)
inline constexpr int kDefaultModelInputHeight = 288;
inline constexpr int kDefaultModelInputWidth = 512;

// Default downsample ratio (overwritten at runtime as 512/max(src_w, src_h))
inline constexpr float kDefaultDsr = 0.25f;

// Default fallback background color: BGR(155,255,120) = RGB(120,255,155)
inline const cv::Scalar kDefaultBgColor{155, 255, 120};

void RVMMode::_InitRecurrentStates(InferenceEngine* engine) {
	// Try to get recurrent state shapes from the engine (RKNN reports actual shapes).
	auto shapes = engine->GetRecurrentStateShapes();
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

bool RVMMode::_OpenVideoWriter(cv::VideoWriter& writer, const std::string& path, int width,
                              int height, double fps) {
	auto& logger = helmsman::utils::Logger::GetInstance();
	writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
	            cv::Size(width, height));
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

cv::Mat RVMMode::InitBackgroundImage(int width, int height) {
	cv::Mat bg;
	if (!config_.background_path.empty()) {
		bg = cv::imread(config_.background_path, cv::IMREAD_COLOR);
		if (!bg.empty()) {
			cv::resize(bg, bg, cv::Size(width, height));
			return bg;
		}
	}
	// Default fallback background: BGR(155,255,120) = RGB(120,255,155), matches e.py baseline
	return cv::Mat(height, width, CV_8UC3, kDefaultBgColor);
}

cv::Mat RVMMode::_InferOneFrame(InferenceEngine* engine, const TensorData& src,
                               const cv::Mat& guide_bgr) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	std::vector<TensorData> inputs = {src};
	state_mgr_.inject(inputs);

	if (engine->NeedsDownsampleRatio()) {
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {dsr_};
		inputs.push_back(std::move(dsr));
	}

	std::vector<TensorData> outputs;
	auto t0 = std::chrono::high_resolution_clock::now();
	engine->Infer(inputs, outputs);
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> dur = t1 - t0;
	logger.Info("infer() cost: " + std::to_string(dur.count()) + " ms.", kRvmModuleName);

	state_mgr_.update(outputs);
	return backend_->Postprocess(outputs, guide_bgr);
}

void RVMMode::_ProcessOneFrame(InferenceEngine* engine, const TensorData& tensor,
                               const cv::Mat& current_frame, int /*model_w*/, int /*model_h*/) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// --------------- infer one frame ---------------
	ManualTimer infer_t;
	infer_t.start();
	cv::Mat alpha_8u = _InferOneFrame(engine, tensor, current_frame);
	const double infer_ms = infer_t.stop();
	acc_lv02_01_03_main_infer_.record(infer_ms);

	// --------------- composite one frame ---------------
	double comp_ms;
	if (use_dma_output_) {
		ManualTimer dma_t;
		dma_t.start();
		const int output_fd = _CompositeToDma(current_frame, alpha_8u);
		comp_ms = dma_t.stop();
		if (frame_count_ == 0) {
			logger.Info("DMA output fd=" + std::to_string(output_fd), kRvmModuleName);
		}
	} else if (config_.output_mode == OutputMode::kDrm) {
		comp_ms = _CompositeToDrm(current_frame, alpha_8u, drm_panel_w_, drm_panel_h_);
	} else {
		comp_ms = _CompositeAndWrite(video_writer_, current_frame, alpha_8u);
	}
	acc_lv02_01_04_main_composite_.record(comp_ms);

	// --------------- log per-frame stats ---------------
	logger.Info("[PerFrame] frame=" + std::to_string(frame_count_) +
	                "  infer=" + std::to_string(infer_ms) + "ms" +
	                "  composite=" + std::to_string(comp_ms) + "ms",
	            kRvmModuleName);

	// FPS measurement: report every 30 frames
	if (frame_count_ % 30 == 0) {
		auto now = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(now - fps_window_start_).count();
		logger.Info("[FPS] " + std::to_string(30.0 / elapsed) + " fps (last 30 frames in " +
		                std::to_string(elapsed) + "s)",
		            kRvmModuleName);
		fps_window_start_ = now;
	}
}

double RVMMode::_CompositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
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
	acc_lv02_01_04_01_resize_alpha_.record(t.stop());

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
	acc_lv02_01_04_02_resize_frame_.record(t.stop());

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
	acc_lv02_01_04_03_blend_.record(t.stop());

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
	acc_lv02_01_04_04_upscale_.record(t.stop());

	// 6. Write
	t.start();
	writer.write(composed_full);
	acc_lv02_01_04_05_writer_.record(t.stop());

	return total_t.elapsed_ms();
}

bool RVMMode::_InitOutputDma(int src_width, int src_height) {
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

int RVMMode::_CompositeToDma(const cv::Mat& frame, const cv::Mat& alpha_8u) {
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
	acc_lv02_01_04_01_resize_alpha_.record(t.stop());

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
	acc_lv02_01_04_02_resize_frame_.record(t.stop());

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
	acc_lv02_01_04_03_blend_.record(t.stop());

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
	acc_lv02_01_04_04_upscale_.record(t.stop());

	return dma_output_buf_->fd();
}

double RVMMode::_CompositeToDrm(const cv::Mat& frame, const cv::Mat& alpha_8u, int panel_w,
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
	acc_lv02_01_04_01_resize_alpha_.record(t.stop());

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
	acc_lv02_01_04_02_resize_frame_.record(t.stop());

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
	acc_lv02_01_04_03_blend_.record(t.stop());

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
	acc_lv02_01_04_04_upscale_.record(t.stop());

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
	acc_lv02_01_04_06_drm_.record(t.stop());

	return total_t.elapsed_ms();
}

RvmModelState RVMMode::InitModelState(InferenceEngine* engine) {
	RvmModelState setup;
	setup.model_input_height =
	    engine->GetInputHeight() > 0 ? engine->GetInputHeight() : kDefaultModelInputHeight;
	setup.model_input_width =
	    engine->GetInputWidth() > 0 ? engine->GetInputWidth() : kDefaultModelInputWidth;

	_InitRecurrentStates(engine);

	return setup;
}

void RVMMode::_ReportAllAccumulatedTimers(void) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	acc_lv02_01_main_loop_total_.report(config_.timing_enabled, logger, kRvmModuleName);
	frontend_->preprocess_acc().report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_02_main_decode_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_03_main_infer_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_04_main_composite_.report(config_.timing_enabled, logger, kRvmModuleName);

	acc_lv02_01_04_01_resize_alpha_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_04_02_resize_frame_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_04_03_blend_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_04_04_upscale_.report(config_.timing_enabled, logger, kRvmModuleName);
	// NOTE: acc_lv02_01_04_05_writer_ measures only the time VideoWriter::write() returns,
	// NOT actual encoder completion (FFmpeg buffers internally). Treat this
	// number as a lower bound for the true write cost.
	acc_lv02_01_04_05_writer_.report(config_.timing_enabled, logger, kRvmModuleName);
	acc_lv02_01_04_06_drm_.report(config_.timing_enabled, logger, kRvmModuleName);
}

void RVMMode::_DoCleaningThings(const std::chrono::steady_clock::time_point& pipeline_start,
                                  const std::string& output_video_path) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	if (video_writer_.isOpened()) {
		video_writer_.release();
		logger.Info("Video compositing complete: " + std::to_string(frame_count_) +
		                " frames written to " + output_video_path,
		            kRvmModuleName);
	}

	if (drm_display_.IsOpen()) {
		drm_display_.Close();
		logger.Info("DRM display closed after " + std::to_string(frame_count_) + " frames.",
		            kRvmModuleName);
	}

	if (use_dma_output_) {
		logger.Info("DMA output: " + std::to_string(frame_count_) +
		                " frames composited to DMA fd=" + std::to_string(dma_output_buf_->fd()),
		            kRvmModuleName);
	}

	if (frame_count_ > 0) {
		const double total_elapsed =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - pipeline_start)
		        .count();
		const double avg_fps = static_cast<double>(frame_count_) / total_elapsed;
		logger.Info("[FPS] Total: " + std::to_string(frame_count_) + " frames in " +
		                std::to_string(total_elapsed) + "s = " + std::to_string(avg_fps) + " fps",
		            kRvmModuleName);
	}

	logger.Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count_),
	            kRvmModuleName);
}

void RVMMode::InitOutputSink(const int src_width, const int src_height,
                                   const double src_fps, const std::string& output_video_path,
                                   const OutputMode output_mode) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	// const int src_width = frontend_->width();
	// const int src_height = frontend_->height();
	dsr_ = 512.0f / static_cast<float>(std::max(src_width, src_height));
	// dance.mp4 1920×1080 → dsr_ = 512/1920 ≈ 0.2667
	// const double src_fps = frontend_->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;

	if (output_mode == OutputMode::kDrm) {
		if (drm_display_.Init(src_width, src_height)) {
			std::tie(drm_panel_w_, drm_panel_h_) = drm_display_.PanelSize();
			logger.Info("DRM display initialized: panel " + std::to_string(drm_panel_w_) + "x" +
			                std::to_string(drm_panel_h_),
			            kRvmModuleName);
		} else {
			logger.Warning("DRM init failed. Falling back to mp4.", kRvmModuleName);
			config_.output_mode = OutputMode::kMp4;
		}
	}
	if (output_mode == OutputMode::kMp4) {
		_OpenVideoWriter(video_writer_, output_video_path, src_width, src_height, output_fps);
	}
}

void RVMMode::_PreprocessBgUint8(cv::Mat bg_bgr, const int src_width,
                                   const int src_height) {
	// Pre-compute background at model resolution for fast compositing in _CompositeAndWrite().
	// Avoids converting bg from uint8→float32 at full resolution every frame.

	cv::Mat bg_model;
	cv::resize(bg_bgr, bg_model,
	           cv::Size(src_width, src_height), 0, 0,
	           cv::INTER_LINEAR);
	bg_model_u8_ = bg_model.clone();  // uint8 copy for fast compositing
}

int RVMMode::Run() {
	auto& logger = helmsman::utils::Logger::GetInstance();

	ScopedTimer run_rvm_timer("Lv02::RVMMode::run() total", config_.timing_enabled, logger,
	                          kRvmModuleName);

	// =========================================================================
	// 1st — Setup phase
	//    - query the model's expected input resolution (fall back to 288×512 if
	//      the engine returns 0, which can happen with dynamic-shape ONNX models)
	//    - initialise the four RNN hidden-state tensors (r1i–r4i) to zero
	// =========================================================================
	const RvmModelState setup = InitModelState(engine_);

	// =========================================================================
	// 2nd — Video I/O setup
	//    - read source dimensions and fps from the InputSource abstraction
	//      (works for both Mp4InputSource and any future camera/IPC source)
	//    - open the VideoWriter; if it fails, _CompositeAndWrite() is a no-op
	//    - load or synthesise a solid-colour background for alpha compositing
	// =========================================================================
	InitOutputSink(setup.model_input_width, setup.model_input_height, frontend_->fps(),
	                     config_.output_bin_path + "/output_composited.mp4", config_.output_mode);

	cv::Mat bg_bgr = InitBackgroundImage(setup.model_input_width, setup.model_input_height);

	// Preprocess the background once at model resolution,
	// so we can skip this step in the main loop and save time on the CPU→GPU path.
	// The background is static so we only need
	_PreprocessBgUint8(bg_bgr, setup.model_input_width, setup.model_input_height);

	// Create RGA hardware operations (stateless, reused every frame).
	// RGA resize replaces cv::resize for the downscale/upscale steps.
	// RGA composite replaces the CPU uint8 blend loop.
	rga_resize_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaResize>();

	// =========================================================================
	// 3rd — Main inference loop
	//    Frontend::ProcessOneFrame() handles both sync and pipeline modes internally.
	// =========================================================================
	fps_window_start_ = std::chrono::steady_clock::now();
	const auto pipeline_start = fps_window_start_;

	_RunMainLoop(engine_, setup);

	_ReportAllAccumulatedTimers();

	const std::string output_video_path = config_.output_bin_path + "/output_composited.mp4";
	_DoCleaningThings(pipeline_start, output_video_path);

	return 0;
}

// =========================================================================
// _RunMainLoop — unified main loop using Frontend::ProcessOneFrame()
//
//   Frontend handles both sync and pipeline modes internally.
//   This loop just calls ProcessOneFrame() and processes each result.
// =========================================================================
void RVMMode::_RunMainLoop(InferenceEngine* engine, const RvmModelState& setup) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	while (true) {
		ManualTimer loop_t;
		loop_t.start();

		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing video at frame " +
			                std::to_string(frame_count_) + ".",
			            kRvmModuleName);
			frontend_->Stop();
			break;
		}

		auto result = frontend_->ProcessOneFrame(setup.model_input_width,
		                                          setup.model_input_height);
		if (!result)
			break;

		logger.Info("=== RVM Frame " + std::to_string(frame_count_ + 1) + " ===", kRvmModuleName);

		_ProcessOneFrame(engine, result->tensor, result->frame, setup.model_input_width,
		                 setup.model_input_height);

		acc_lv02_01_main_loop_total_.record(loop_t.stop());

		logger.Info(" --- End of RVM Frame " + std::to_string(frame_count_ + 1) + " ---\n",
		            kRvmModuleName);

		frame_count_++;
	}
}
