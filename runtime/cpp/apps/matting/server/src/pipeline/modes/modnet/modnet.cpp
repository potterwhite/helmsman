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

#include "pipeline/modes/modnet/modnet.h"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include "RGAKit/rga_resize.h"
#include "Utils/timing/timer.h"
#include "common/common-define.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;
using helmsman::rgakit::RgaResize;

using helmsman::utils::timing::ManualTimer;
using helmsman::utils::timing::ScopedTimer;

extern std::atomic<bool> g_stop_signal_received;

inline constexpr std::string_view kModnetModuleName = "MODNetMode";

// Default fallback background color: BGR(155,255,120) = RGB(120,255,155)
inline const cv::Scalar kDefaultBgColor{155, 255, 120};

void MODNetMode::SetEngine(InferenceEngine* engine) { engine_ = engine; }
void MODNetMode::SetFrontend(FrontEnd* frontend) { frontend_ = frontend; }
void MODNetMode::SetBackend(BackEnd* backend) { backend_ = backend; }
void MODNetMode::SetAppConfig(const AppConfig& config) { config_ = config; }

void MODNetMode::_InitOutputSink(int src_width, int src_height, double fps) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	if (config_.output_mode == OutputMode::kDrm) {
		if (drm_display_.Init(src_width, src_height)) {
			std::tie(drm_panel_w_, drm_panel_h_) = drm_display_.PanelSize();
			logger.Info("DRM display initialized: panel " + std::to_string(drm_panel_w_) +
			                 "x" + std::to_string(drm_panel_h_),
			            kModnetModuleName);
		} else {
			logger.Warning("DRM init failed. Falling back to mp4.", kModnetModuleName);
			config_.output_mode = OutputMode::kMp4;
		}
	}
	if (config_.output_mode == OutputMode::kMp4) {
		const double output_fps = (fps > 0) ? fps : 30.0;
		const std::string alpha_path = config_.output_bin_path + "/output_alpha.mp4";
		video_writer_.open(alpha_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), output_fps,
		                   cv::Size(src_width, src_height));
		if (video_writer_.isOpened()) {
			logger.Info("VideoWriter opened: " + alpha_path + " (" + std::to_string(src_width) + "x" +
			                 std::to_string(src_height) + " @ " + std::to_string(output_fps) + " fps)",
			            kModnetModuleName);
		} else {
			logger.Warning("Failed to open VideoWriter: " + alpha_path, kModnetModuleName);
		}

		const std::string comp_path = config_.output_bin_path + "/output_composited.mp4";
		composited_writer_.open(comp_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), output_fps,
		                        cv::Size(src_width, src_height));
		if (composited_writer_.isOpened()) {
			logger.Info("VideoWriter opened: " + comp_path + " (" + std::to_string(src_width) + "x" +
			                 std::to_string(src_height) + " @ " + std::to_string(output_fps) + " fps)",
			            kModnetModuleName);
		} else {
			logger.Warning("Failed to open VideoWriter: " + comp_path, kModnetModuleName);
		}
	}
}

void MODNetMode::InitBackgroundImage(int width, int height) {
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

double MODNetMode::_Composite(const cv::Mat& frame, const cv::Mat& alpha_8u, int model_w,
                               int model_h, int output_w, int output_h, cv::Mat& composed) {
	if (alpha_8u.empty())
		return 0.0;

	ManualTimer t;
	t.start();
	composed = backend_->Composite(frame, alpha_8u, model_w, model_h, output_w, output_h);
	return t.stop();
}

void MODNetMode::_Display(const cv::Mat& result, int output_w, int output_h) {
	if (result.empty())
		return;

	if (config_.output_mode == OutputMode::kDrm) {
		// ----- DRM show Mode -----
		cv::Mat resized;
		if (result.cols != output_w || result.rows != output_h) {
			cv::resize(result, resized, cv::Size(output_w, output_h), 0, 0, cv::INTER_LINEAR);
		} else {
			resized = result;
		}

		const int n_pixels = output_w * output_h;
		argb_buf_.resize(static_cast<size_t>(n_pixels) * 4);
		uint8_t* xrgb = argb_buf_.data();

		if (resized.channels() == 1) {
			// Single channel alpha matte - display as grayscale
			const uint8_t* alpha = resized.ptr<uint8_t>(0);
			for (int i = 0; i < n_pixels; ++i) {
				xrgb[0] = alpha[0];  // B
				xrgb[1] = alpha[0];  // G
				xrgb[2] = alpha[0];  // R
				xrgb[3] = 0xFF;      // X (padding)
				alpha++;
				xrgb += 4;
			}
		} else {
			// 3-channel BGR image
			const uint8_t* bgr = resized.ptr<uint8_t>(0);
			for (int i = 0; i < n_pixels; ++i) {
				xrgb[0] = bgr[0];  // B -> B
				xrgb[1] = bgr[1];  // G -> G
				xrgb[2] = bgr[2];  // R -> R
				xrgb[3] = 0xFF;    // X (padding)
				bgr += 3;
				xrgb += 4;
			}
		}
		drm_display_.ShowARGB(argb_buf_.data());

	}
}

int MODNetMode::Run() {
	auto& logger = helmsman::utils::Logger::GetInstance();

	const int model_input_height = engine_->GetInputHeight() > 0 ? engine_->GetInputHeight() : 512;
	const int model_input_width = engine_->GetInputWidth() > 0 ? engine_->GetInputWidth() : 512;

	// 1. Init output sink (DRM or VideoWriter) — once before the loop
	_InitOutputSink(frontend_->width(), frontend_->height(), frontend_->fps());

	// 1b. Init background image for compositing
	InitBackgroundImage(model_input_width, model_input_height);

	// 2. Main inference loop
	fps_window_start_ = std::chrono::steady_clock::now();

	while (true) {
		// Check stop signal
		if (g_stop_signal_received.load()) {
			logger.Info("Stop signal received. Finishing at frame " + std::to_string(frame_count_) + ".",
			            kModnetModuleName);
			frontend_->Stop();
			break;
		}

		// Frame header
		if (frame_count_ > 0)
			logger.Info("", kModnetModuleName);
		logger.Info("───── Frame " + std::to_string(frame_count_ + 1) + " ─────", kModnetModuleName);

		ManualTimer loop_t;
		loop_t.start();

		// 2a. Frontend: preprocess one frame
		auto result = frontend_->ProcessOneFrame(model_input_width, model_input_height);
		if (!result)
			break;

		// 2b. Inference
		std::vector<TensorData> outputs;
		double infer_ms = engine_->Infer({result->tensor}, outputs);

		// 2c. BackEnd: postprocess → alpha matte
		double postprocess_ms = 0.0;
		cv::Mat alpha;
		{
			ManualTimer pp_t;
			pp_t.start();
			alpha = backend_->Postprocess(outputs, result->frame);
			pp_t.stop();
			postprocess_ms = pp_t.elapsed_ms();
		}

		// 2d. Composite: blend alpha with background
		const int output_w =
		    (config_.output_mode == OutputMode::kDrm) ? drm_panel_w_ : result->frame.cols;
		const int output_h =
		    (config_.output_mode == OutputMode::kDrm) ? drm_panel_h_ : result->frame.rows;
		cv::Mat composed;
		const double composite_ms =
		    _Composite(result->frame, alpha, model_input_width, model_input_height, output_w, output_h,
		               composed);

		// 2e. Display (DRM or VideoWriter)
		double display_ms = 0.0;
		{
			ManualTimer d_t;
			d_t.start();
			_Display(composed, output_w, output_h);
			display_ms = d_t.stop();
		}

		// 2e2. Write video files (always, regardless of display mode)
		if (video_writer_.isOpened()) {
			cv::Mat alpha_bgr;
			if (alpha.channels() == 1) {
				cv::cvtColor(alpha, alpha_bgr, cv::COLOR_GRAY2BGR);
			} else {
				alpha_bgr = alpha;
			}
			if (alpha_bgr.cols != output_w || alpha_bgr.rows != output_h) {
				cv::resize(alpha_bgr, alpha_bgr, cv::Size(output_w, output_h));
			}
			video_writer_.write(alpha_bgr);
		}
		if (composited_writer_.isOpened() && !composed.empty()) {
			composited_writer_.write(composed);
		}

		// 2f. Per-frame timing
		const double frame_total = loop_t.stop();
		const auto fm = [&](double v) -> std::string {
			std::ostringstream o;
			o << std::fixed << std::setprecision(3) << v;
			return o.str();
		};
		logger.Info("  frontend(preprocess: " + fm(result->preprocess_ms) + "ms)", kModnetModuleName);
		logger.Info("  inference(total: " + fm(infer_ms) + "ms)", kModnetModuleName);
		logger.Info("  backend(postprocess: " + fm(postprocess_ms) + "ms; composite: " +
		                 fm(composite_ms) + "ms; display: " + fm(display_ms) + "ms)",
		            kModnetModuleName);
		logger.Info("  total: " + fm(frame_total) + "ms", kModnetModuleName);

		// FPS measurement: report every 30 frames
		if (frame_count_ > 0 && frame_count_ % 30 == 0) {
			auto now = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration<double>(now - fps_window_start_).count();
			logger.Info("[FPS] " + std::to_string(30.0 / elapsed) + " fps (last 30 frames in " +
			                 std::to_string(elapsed) + "s)",
			            kModnetModuleName);
			fps_window_start_ = now;
		}

		frame_count_++;
	}

	// 3. Cleanup
	if (video_writer_.isOpened()) {
		video_writer_.release();
		logger.Info("Video alpha output complete: " + std::to_string(frame_count_) + " frames written.",
		            kModnetModuleName);
	}
	if (composited_writer_.isOpened()) {
		composited_writer_.release();
		logger.Info("Video composited output complete: " + std::to_string(frame_count_) + " frames written.",
		            kModnetModuleName);
	}

	if (drm_display_.IsOpen()) {
		drm_display_.Close();
		logger.Info("DRM display closed after " + std::to_string(frame_count_) + " frames.",
		            kModnetModuleName);
	}

	logger.Info("MODNet pipeline finished. Total frames: " + std::to_string(frame_count_),
	            kModnetModuleName);

	return 0;
}
