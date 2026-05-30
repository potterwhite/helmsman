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

#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include "DRMKit/drm_display.h"
// #include "DmaKit/dma_buffer.h"  // DMA output: currently disabled
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend/frontend-core/frontend-base.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"

/**
 * Holds the resolved model input dimensions returned by RVMMode::InitModelState().
 * Kept outside the class — no need to nest a plain data struct inside a class.
 */
struct RvmModelState {
	int model_input_height;
	int model_input_width;
};

class RVMMode {
   public:
	void SetEngine(InferenceEngine* engine);
	void SetFrontend(FrontendBase* frontend);
	void SetBackend(MattingBackend* backend);
	void SetConfig(const AppConfig& config);

	int Run();

   private:
	/**
     * Load the model and resolve model input dimensions.
     * Returns the resolved model dimensions needed by the multithread worker.
     */
	RvmModelState InitModelState(InferenceEngine* engine);

	bool _OpenVideoWriter(cv::VideoWriter& writer, const std::string& path, int width, int height,
	                      double fps);

	void InitBackgroundImage(int width, int height);

	// Composite: blend alpha with background, write result to composed, return elapsed ms.
	double _Composite(const cv::Mat& frame, const cv::Mat& alpha_8u, int model_w,
	                  int model_h, int output_w, int output_h, cv::Mat& composed);

	// Display: deliver composited frame to output sink, return elapsed ms.
	double _Display(const cv::Mat& composed, int output_w, int output_h);

	/**
	 * Unified main loop. Uses Frontend::ProcessOneFrame() which handles
	 * both sync and multithread modes internally.
	 */
	void _RunMainLoop(InferenceEngine* engine, const RvmModelState& setup);

	/**
	* Report all accumulated timers via logger. Called at the end of run() after the main loop exits.
	*/
	void _ReportAllAccumulatedTimers(void);

	/**
	 * Perform any necessary cleanup of resources (e.g. DRM buffer release) before exiting run().
	 */
	void _DoCleaningThings(const std::chrono::steady_clock::time_point& pipeline_start,
	                       const std::string& output_video_path);

	/**
	 * Shared output path for both output modes (MP4 file or DRM display). Sets up the VideoWriter if needed.
	 */
	void InitOutputSink(const int src_width, const int src_height, const double src_fps,
	                    const std::string& output_video_path, const OutputMode output_mode);

   private:
	// Member variables
	InferenceEngine* engine_ = nullptr;  // Non-owning; owned by Pipeline
	FrontendBase* frontend_ = nullptr;   // Non-owning; owned by Pipeline
	MattingBackend* backend_ = nullptr;  // Non-owning; owned by Pipeline
	AppConfig config_;                   // Copy of the app config, set via SetConfig()
	// DMA zero-copy output: currently disabled.
	// std::unique_ptr<helmsman::dmakit::DmaBuffer> dma_output_buf_;

	// DRM display (initialized when output_mode == kDrm)
	helmsman::drmkit::DrmDisplay drm_display_;
	std::vector<uint8_t> argb_buf_;  // reusable buffer for BGR→XRGB conversion

	/* -------------------------------------------------------------------------
	// Pipeline timing layout (s10 — full coverage)
	//
	// Whole-run timers (overlap with the above; cheap, kept for context)
	//
	//   ScopedTimer "Lv01::pipeline.run() total"                 (pipeline.cpp)   — outermost
	//   ScopedTimer "Lv02::pipeline::RVMMode::run()"             (this fn)        — wraps loop
	//   ScopedTimer "Lv03::pipeline::RVMMode::_RunMainLoop()"    (this fn)        — model load only
	//
	// Per-frame wall clock breakdown
	//
	//   [main thread]                                                      [worker thread]
	//   acc_lv03_01_mainloop  (whole iteration)
	//     ├── tensor_ch.pop()                          ◄────────────       acc_lv03_02_worker_preprocess_
	//     │   (blocks if worker                        pushes here         (run on worker:
	//     │    not done yet)                                                BGR→tensor resize+norm)
	//     ├── acc_lv03_02_01_mainloop_frontend_decode_ ────────────►       raw_ch.pop()
	//     │   (read next frame                                             (worker waits here)
	//     │    + push to raw_ch)
	//     ├── acc_lv03_03_mainloop_inferenceengine_infer_     (NPU inference, current frame)
	//     ├── acc_lv03_04_02_mainloop_backend_composite_      (composite + write, current frame)
	//     └── acc_lv03_04_03_mainloop_backend_display_        ()
	//
	//
	//   [FPS]   line every 30 frames                                         — moving fps
	//   [PerFrame] line every frame                                          — infer + comp
	//
	// Identity (approx, ignoring tiny logging overhead):
	//   acc_lv03_01_mainloop ≈ max(tensor_ch.pop wait, 0) + acc_lv03_02_01_mainloop_frontend_decode_ + acc_lv03_03_mainloop_inferenceengine_infer_ + acc_lv03_04_02_mainloop_backend_composite_
	//   acc_lv03_04_02_mainloop_backend_composite_ ≈ Backend::Composite() + writer
	// ------------------------------------------------------------------------- */
	using sa = helmsman::utils::timing::StageAccumulator;

	sa acc_lv03_01_mainloop{"Lv03-01::mainloop"};
	sa acc_lv03_02_01_mainloop_frontend_decode_{"  Lv03-02-01::mainloop::frontend::decode"};
	// acc_lv03_03_mainloop_inferenceengine_infer_ moved to InferenceEngine::infer_acc_
	sa acc_lv03_04_01_mainloop_backend_postprocess_{"  Lv03-04-01::mainloop::backend::postprocess"};
	sa acc_lv03_04_02_mainloop_backend_composite_{"  Lv03-04-02::mainloop::backend::composite"};
	sa acc_lv03_04_03_mainloop_backend_display_{"    Lv03-04-03::mainloop::backend::display"};

	size_t frame_count_ = 0;
	std::chrono::steady_clock::time_point fps_window_start_;

	cv::VideoWriter video_writer_;

	// DMA zero-copy output: currently disabled.
	// const bool use_dma_output_ = false;

	int drm_panel_w_ = 0;
	int drm_panel_h_ = 0;
};