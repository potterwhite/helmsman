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
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <thread>
#include "RGAKit/rga_resize.h"
#include "Utils/timing/timer.h"
#include "common/common-define.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;
using helmsman::rgakit::RgaResize;

using helmsman::utils::timing::ManualTimer;
using helmsman::utils::timing::ScopedTimer;
using helmsman::utils::timing::StageAccumulator;

extern std::atomic<bool> g_stop_signal_received;

inline constexpr std::string_view kRvmModuleName = "RVMMode";

// Default model input dimensions (fallback when engine reports 0)
inline constexpr int kDefaultModelInputHeight = 288;
inline constexpr int kDefaultModelInputWidth = 512;

// Default fallback background color: BGR(155,255,120) = RGB(120,255,155)
inline const cv::Scalar kDefaultBgColor{155, 255, 120};

void RVMMode::SetEngine(InferenceEngine* engine) {
	engine_ = engine;
}
void RVMMode::SetFrontend(FrontendBase* frontend) {
	frontend_ = frontend;
}
void RVMMode::SetBackend(MattingBackend* backend) {
	backend_ = backend;
}
void RVMMode::SetConfig(const AppConfig& config) {
	config_ = config;
}

bool RVMMode::_OpenVideoWriter(cv::VideoWriter& writer, const std::string& path, int width,
                               int height, double fps) {

	writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
	if (!writer.isOpened()) {
		GetLogger().Warning(
		    "Failed to open VideoWriter: " + path + ". Composited video will NOT be saved.",
		    kRvmModuleName);
		return false;
	}
	GetLogger().Info("VideoWriter opened: " + path + " (" + std::to_string(width) + "x" +
	                     std::to_string(height) + " @ " + std::to_string(fps) + " fps)",
	                 kRvmModuleName);
	return true;
}

void RVMMode::InitBackgroundImage(int width, int height) {
	cv::Mat bg;
	if (!config_.background_path.empty()) {
		bg = cv::imread(config_.background_path, cv::IMREAD_COLOR);
	}
	if (bg.empty()) {
		bg = cv::Mat(height, width, CV_8UC3, kDefaultBgColor);
	} else {
		cv::Mat resized(height, width, CV_8UC3);
		ImageDescriptor src(bg.data, bg.cols, bg.rows, RgaPixelFormat::kBgr888);
		ImageDescriptor dst(resized.data, width, height, RgaPixelFormat::kBgr888);
		if (!RgaResize::Instance().Execute(src, dst)) {
			fprintf(stderr, "[FATAL] RGA resize failed for background init — hardware error\n");
			std::abort();
		}
		bg = resized;
	}
	backend_->SetBackgroundModelImage(bg.clone());
}

double RVMMode::_Composite(const cv::Mat& frame, const cv::Mat& alpha_8u, int model_w, int model_h,
                           int output_w, int output_h, cv::Mat& composed) {
	// 1st. sanity check
	if (alpha_8u.empty())
		return 0.0;

	// 2nd. start accumulating
	ManualTimer t;
	t.start();

	// 3rd. backend compositing
	composed = backend_->Composite(frame, alpha_8u, model_w, model_h, output_w, output_h);

	// 4th. stop compositing timer and record
	t.stop();
	acc_lv03_04_02_mainloop_backend_composite_.record(t.elapsed_ms());

	return t.elapsed_ms();
}

double RVMMode::_Display(const cv::Mat& composed, int output_w, int output_h) {

	// 1st. sanity check
	if (composed.empty())
		return 0.0;

	// 2nd. start accumulating
	ManualTimer t;
	t.start();

	// 3rd. start displaying
	// DMA output path: currently disabled (use_dma_output_ = false).
	// if (use_dma_output_) {
	//     void* dma_ptr = dma_output_buf_->map();
	//     if (dma_ptr) {
	//         memcpy(dma_ptr, composed.data, composed.total() * composed.elemSize());
	//     }
	// }
	if (config_.output_mode == OutputMode::kDrm) {
		// ----- DRM show MP4 Mode -----
		const int n_pixels = output_w * output_h;
		argb_buf_.resize(static_cast<size_t>(n_pixels) * 4);
		const uint8_t* bgr = composed.ptr<uint8_t>(0);
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

	} else {
		// ----- OpenCV Write MP4 Mode -----
		video_writer_.write(composed);
	}

	// 4th. stop accumulating and record
	t.stop();
	acc_lv03_04_03_mainloop_backend_display_.record(t.elapsed_ms());

	return t.elapsed_ms();
}

// DMA output path: currently disabled (use_dma_output_ = false).
// _InitOutputDma() allocates a DMA buffer for zero-copy output.
// Re-enable when ready to experiment with DMA-based output.
//
// bool RVMMode::_InitOutputDma(int src_width, int src_height) {
//     auto& logger = helmsman::utils::Logger::GetInstance();
//     const size_t buf_bytes =
//         static_cast<size_t>(src_width) * static_cast<size_t>(src_height) * 3;  // BGR888
//     dma_output_buf_ = helmsman::dmakit::DmaBuffer::Allocate(buf_bytes);
//     if (!dma_output_buf_) {
//         logger.Warning("Failed to allocate DMA output buffer (" + std::to_string(buf_bytes) +
//                            " bytes). Falling back to VideoWriter.",
//                        kRvmModuleName);
//         return false;
//     }
//     dma_output_buf_->map();
//     logger.Info("DMA output buffer allocated: fd=" + std::to_string(dma_output_buf_->fd()) +
//                     ", size=" + std::to_string(buf_bytes) + " bytes (" + std::to_string(src_width) +
//                     "x" + std::to_string(src_height) + " BGR)",
//                 kRvmModuleName);
//     return true;
// }

RvmModelState RVMMode::InitModelState(InferenceEngine* engine) {

	RvmModelState setup;

	setup.model_input_height =
	    engine->GetInputHeight() > 0 ? engine->GetInputHeight() : kDefaultModelInputHeight;
	setup.model_input_width =
	    engine->GetInputWidth() > 0 ? engine->GetInputWidth() : kDefaultModelInputWidth;

	return setup;
}

void RVMMode::_ReportAllAccumulatedTimers(void) {
	if (!config_.timing_enabled)
		return;

	GetLogger().Info("────────── Frontend ──────────", kRvmModuleName);
	frontend_->read_input_acc().report(true, GetLogger(), kRvmModuleName, "read_input_source");
	frontend_->decode_acc().report(true, GetLogger(), kRvmModuleName, "decode_frame");
	frontend_->color_convert_acc().report(true, GetLogger(), kRvmModuleName, "convert_to_bgr");
	frontend_->preprocess_acc().report(true, GetLogger(), kRvmModuleName, "preprocess");
	frontend_->resize_acc().report(true, GetLogger(), kRvmModuleName, "  resize");

	GetLogger().Info("────────── Inference ──────────", kRvmModuleName);
	engine_->infer_acc().report(true, GetLogger(), kRvmModuleName, "infer");

	GetLogger().Info("────────── Backend ──────────", kRvmModuleName);
	acc_lv03_04_02_mainloop_backend_composite_.report(true, GetLogger(), kRvmModuleName,
	                                                  "composite");
	acc_lv03_04_03_mainloop_backend_display_.report(true, GetLogger(), kRvmModuleName, "  display");

	GetLogger().Info("────────── Overall ──────────", kRvmModuleName);
	acc_lv03_01_mainloop.report(true, GetLogger(), kRvmModuleName, "mainloop (per frame)");
}

void RVMMode::_DoCleaningThings(const std::chrono::steady_clock::time_point& pipeline_start,
                                const std::string& output_video_path) {
	if (video_writer_.isOpened()) {
		video_writer_.release();
		GetLogger().Info("Video compositing complete: " + std::to_string(frame_count_) +
		                     " frames written to " + output_video_path,
		                 kRvmModuleName);
	}

	if (drm_display_.IsOpen()) {
		drm_display_.Close();
		GetLogger().Info("DRM display closed after " + std::to_string(frame_count_) + " frames.",
		                 kRvmModuleName);
	}

	// if (use_dma_output_) {
	//     GetLogger().Info("DMA output: " + std::to_string(frame_count_) +
	//                          " frames composited to DMA fd=" + std::to_string(dma_output_buf_->fd()),
	//                      kRvmModuleName);
	// }

	if (frame_count_ > 0) {
		const double total_elapsed =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - pipeline_start)
		        .count();
		const double avg_fps = static_cast<double>(frame_count_) / total_elapsed;
		GetLogger().Info("[FPS] Total: " + std::to_string(frame_count_) + " frames in " +
		                     std::to_string(total_elapsed) + "s = " + std::to_string(avg_fps) +
		                     " fps",
		                 kRvmModuleName);
	}

	GetLogger().Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count_),
	                 kRvmModuleName);
}

void RVMMode::InitOutputSink(const int src_width, const int src_height, const double src_fps,
                             const std::string& output_video_path, const OutputMode output_mode) {
	// const int src_width = frontend_->width();
	// const int src_height = frontend_->height();
	engine_->SetDownsampleRatio(512.0f / static_cast<float>(std::max(src_width, src_height)));
	// dance.mp4 1920×1080 → dsr = 512/1920 ≈ 0.2667
	// const double src_fps = frontend_->fps();
	const double output_fps = (src_fps > 0) ? src_fps : 30.0;

	if (output_mode == OutputMode::kDrm) {
		if (drm_display_.Init(src_width, src_height)) {
			std::tie(drm_panel_w_, drm_panel_h_) = drm_display_.PanelSize();
			GetLogger().Info("DRM display initialized: panel " + std::to_string(drm_panel_w_) +
			                     "x" + std::to_string(drm_panel_h_),
			                 kRvmModuleName);
		} else {
			GetLogger().Warning("DRM init failed. Falling back to mp4.", kRvmModuleName);
			config_.output_mode = OutputMode::kMp4;
		}
	}
	if (output_mode == OutputMode::kMp4) {
		_OpenVideoWriter(video_writer_, output_video_path, src_width, src_height, output_fps);
	}
}

// =========================================================================
// _RunMainLoop — unified main loop using Frontend::ProcessOneFrame()
//
//   Frontend handles both sync and multithread modes internally.
//   This loop just calls ProcessOneFrame() and processes each result.
// =========================================================================
void RVMMode::_RunMainLoop(InferenceEngine* engine, const RvmModelState& setup) {
	ScopedTimer init_mainloop_timer("Lv03::pipeline::RVMMode::_RunMainLoop()",
	                                config_.timing_enabled, GetLogger(), kRvmModuleName);

	while (true) {
		// start accumulating
		ManualTimer loop_t;
		loop_t.start();

		// user friendly ensurance
		if (g_stop_signal_received.load()) {
			GetLogger().Info("Stop signal received. Finishing video at frame " +
			                     std::to_string(frame_count_) + ".",
			                 kRvmModuleName);
			frontend_->Stop();
			break;
		}

		// -------------------------------------------------------------
		// --- 1st: frontend_->ProcessOneFrame ---
		// - In sync mode: reads one frame, preprocesses it, and returns the tensor and frame.
		// - In multithread mode: waits for the next completed frame from the pipeline, and returns the tensor and frame.
		//
		// sync or multithread mode has been configured at pipeline layer previously
		auto result = frontend_->ProcessOneFrame(setup.model_input_width, setup.model_input_height);
		if (!result)
			break;

		// GetLogger().Info("=== RVM Frame " + std::to_string(frame_count_ + 1) + " ===",
		//                  kRvmModuleName);

		const int model_w = setup.model_input_width;
		const int model_h = setup.model_input_height;

		// -------------------------------------------------------------
		// --- 2nd: inference engine - infer ---
		std::vector<TensorData> outputs;
		const double infer_ms = engine->Infer({result->tensor}, outputs);

		// -------------------------------------------------------------
		// --- 3rd: backend - postprocess ---
		cv::Mat alpha_8u = backend_->Postprocess(outputs, result->frame);

		// -------------------------------------------------------------
		// --- 4th: backend - composite ---
		const int output_w =
		    (config_.output_mode == OutputMode::kDrm) ? drm_panel_w_ : result->frame.cols;
		const int output_h =
		    (config_.output_mode == OutputMode::kDrm) ? drm_panel_h_ : result->frame.rows;
		cv::Mat composed;
		const double composite_ms =
		    _Composite(result->frame, alpha_8u, model_w, model_h, output_w, output_h, composed);

		// -------------------------------------------------------------
		// --- 5th: backend - display ---
		const double display_ms = _Display(composed, output_w, output_h);

		// -------------------------------------------------------------
		// --- end: per-frame timing block ---
		const double frame_total = loop_t.elapsed_ms();
		const int fn = static_cast<int>(frame_count_ + 1);
		const auto fm = [&](double v) -> std::string {
			std::ostringstream o;
			o << std::fixed << std::setprecision(3) << v;
			return o.str();
		};

		GetLogger().Info("───── Frame " + std::to_string(fn) + " ─────", kRvmModuleName);
		GetLogger().Info("  frontend(read_input_source: " + fm(result->read_ms) +
		                     "ms; decode_frame: " + fm(result->decode_ms) +
		                     "ms; convert_to_bgr: " + fm(result->color_convert_ms) +
		                     "ms; preprocess: " + fm(result->preprocess_ms) + "ms)",
		                 kRvmModuleName);
		GetLogger().Info("  inference(infer: " + fm(infer_ms) + "ms)", kRvmModuleName);
		GetLogger().Info(
		    "  backend(composite: " + fm(composite_ms) + "ms; display: " + fm(display_ms) + "ms)",
		    kRvmModuleName);
		GetLogger().Info("  total: " + fm(frame_total) + "ms", kRvmModuleName);

		// // --- old per-frame stats (commented out) ---
		// GetLogger().Info("[PerFrame] frame=" + std::to_string(frame_count) +
		//                      "  [3rd]infer=" + std::to_string(infer_ms) + "ms" +
		//                      "  composite=" + std::to_string(composite_ms) + "ms" +
		//                      "  display=" + std::to_string(display_ms) + "ms",
		//                  kRvmModuleName);

		// FPS measurement: report every 30 frames
		if (frame_count_ % 30 == 0) {
			auto now = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration<double>(now - fps_window_start_).count();
			GetLogger().Info("[FPS] " + std::to_string(30.0 / elapsed) +
			                     " fps (last 30 frames in " + std::to_string(elapsed) + "s)",
			                 kRvmModuleName);
			fps_window_start_ = now;
		}

		// stop main loop timer and record
		acc_lv03_01_mainloop.record(loop_t.stop());

		// increment frame count at the very end, so that logs show the current frame number (starting from 1)
		frame_count_++;
	}
}

int RVMMode::Run() {
	ScopedTimer run_rvm_timer("Lv02::pipeline::RVMMode::run()", config_.timing_enabled, GetLogger(),
	                          kRvmModuleName);

	// =========================================================================
	// 1st - Setup phase
	//    - query the model's expected input resolution (fall back to 288×512 if
	//      the engine returns 0, which can happen with dynamic-shape ONNX models)
	//    - initialise the four RNN hidden-state tensors (r1i–r4i) to zero
	// =========================================================================
	const RvmModelState setup = InitModelState(engine_);
	engine_->InitRecurrentStates();

	// =========================================================================
	// 2nd - Video I/O setup
	//    - read source dimensions and fps from the InputSource abstraction
	//      (works for both Mp4InputSource and any future camera/IPC source)
	//    - open the VideoWriter; if it fails, _Composite() / _Display() are no-ops
	//    - load or synthesise a solid-colour background for alpha compositing
	// =========================================================================
	InitOutputSink(setup.model_input_width, setup.model_input_height, frontend_->fps(),
	               config_.output_bin_path + "/output_composited.mp4", config_.output_mode);

	InitBackgroundImage(setup.model_input_width, setup.model_input_height);

	// =========================================================================
	// 3rd - Main inference loop
	//    Frontend::ProcessOneFrame() handles both sync and multithread modes internally.
	// =========================================================================
	fps_window_start_ = std::chrono::steady_clock::now();
	const auto pipeline_start = fps_window_start_;

	_RunMainLoop(engine_, setup);

	// =========================================================================
	// 4th - Finish up and report timing stats
	// =========================================================================
	_ReportAllAccumulatedTimers();

	const std::string output_video_path = config_.output_bin_path + "/output_composited.mp4";
	_DoCleaningThings(pipeline_start, output_video_path);

	return 0;
}
