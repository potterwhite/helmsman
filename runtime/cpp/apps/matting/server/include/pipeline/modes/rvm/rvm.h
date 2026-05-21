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
#include "DmaKit/dma_buffer.h"
#include "RGAKit/rga_resize.h"
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/infra/recurrent-state-manager.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend/frontend.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"

/**
 * Holds the resolved model input dimensions returned by RVMMode::prepareRun().
 * Kept outside the class — no need to nest a plain data struct inside a class.
 */
struct RvmRunSetup {
	size_t model_input_height;
	size_t model_input_width;
};

class RVMMode {
   public:
	int run(InferenceEngine* engine, Frontend* frontend, const AppConfig& config);

   private:
	/**
     * Load the model, resolve model input dimensions, initialise recurrent
     * states, and wire up frontend / backend paths.
     * Returns the resolved model dimensions needed by the prefetch worker.
     */
	RvmRunSetup prepareRun(InferenceEngine* engine);

	void initRecurrentStates(InferenceEngine* engine);
	bool openVideoWriter(cv::VideoWriter& writer, const std::string& path, int width, int height,
	                     double fps);
	cv::Mat loadOrCreateBackground(int width, int height);
	cv::Mat inferOneFrame(InferenceEngine* engine, const TensorData& src, const cv::Mat& guide_bgr);
	// Returns total composite time in ms (for per-frame logging).
	double compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
	                         const cv::Mat& alpha_8u);

	// DRM composite: blend + upscale + BGR→XRGB + ShowARGB.
	// Returns total composite time in ms.
	double compositeToDrm(const cv::Mat& frame, const cv::Mat& alpha_8u, int panel_w, int panel_h);

	// DMA zero-copy composite: composites into a pre-allocated DMA buffer.
	// Returns the DMA buffer fd (valid until next call or destruction).
	// The buffer is allocated once at initOutputDma() and reused every frame.
	int compositeToDma(const cv::Mat& frame, const cv::Mat& alpha_8u);

	// Allocate the output DMA buffer for the given source dimensions.
	// Must be called before compositeToDma(). Returns false on failure.
	bool initOutputDma(int src_width, int src_height);

	/**
     * Body of the prefetch worker thread.
     * Loops: pop raw frame from raw_ch → preprocess → push tensor to tensor_ch.
     * Closes tensor_ch on exit to signal EOF to the main inference loop.
     */
	void runPrefetchWorker(size_t model_w, size_t model_h, SingleSlotChannel<cv::Mat>& raw_ch,
	                       SingleSlotChannel<TensorData>& tensor_ch,
	                       helmsman::utils::timing::StageAccumulator& preprocess_acc);

   private:
	// Member variables
	float dsr_ = 0.25f;             ///< downsample_ratio, overwritten in run()
	Frontend* frontend_ = nullptr;  // Non-owning; owned by Pipeline
	AppConfig config_;              // Copy of the app config, set at run() entry
	MattingBackend backend_;
	RecurrentStateManager state_mgr_;
	cv::Mat bg_model_u8_;  // Pre-computed background at model resolution (CV_8UC3)

	// RGA hardware operations (stateless, created once, reused every frame)
	std::unique_ptr<helmsman::rgakit::RgaResize> rga_resize_;

	// DMA zero-copy output buffer (allocated once, reused every frame)
	std::unique_ptr<helmsman::dmakit::DmaBuffer> dma_output_buf_;

	// DRM display (initialized when output_mode == kDrm)
	helmsman::drmkit::DrmDisplay drm_display_;
	std::vector<uint8_t> argb_buf_;  // reusable buffer for BGR→XRGB conversion

	/* -------------------------------------------------------------------------
	// Pipeline timing layout (s10 — full coverage)
	//
	// Per-frame wall clock breakdown
	//
	//   [main thread]                            [worker thread]
	//   acc_lv02_01_main_loop_total_  (whole iteration)
	//     ├── tensor_ch.pop()    ◄────────────   acc_lv02_01_01_worker_preprocess_
	//     │   (blocks if worker     pushes here  (run on worker:
	//     │    not done yet)                       BGR→tensor resize+norm)
	//     ├── acc_lv02_01_02_main_decode_         ────────────►   raw_ch.pop()
	//     │   (read next frame                   (worker waits here)
	//     │    + push to raw_ch)
	//     ├── acc_lv02_01_03_main_infer_          (NPU inference, current frame)
	//     └── acc_lv02_01_04_main_composite_      (composite + write, current frame)
	//             │
	//             ├── acc_lv02_01_04_01_resize_alpha_  (CPU resize alpha → model size)
	//             ├── acc_lv02_01_04_02_resize_frame_  (RGA resize frame → model size)
	//             ├── acc_lv02_01_04_03_blend_         (CPU alpha blend at model size)
	//             ├── acc_lv02_01_04_04_upscale_       (RGA upscale composed → full size)
	//             ├── acc_lv02_01_04_05_writer_        (VideoWriter::write — see NOTE below)
	//             └── acc_lv02_01_04_06_drm_           ()
	//
	// Whole-run timers (overlap with the above; cheap, kept for context)
	//
	//   ScopedTimer "Lv01::main::pipeline.run() total"           (pipeline.cpp)   — outermost
	//   ScopedTimer "Lv02::RVMMode::run() total"            (this fn)        — wraps loop
	//   ScopedTimer "Lv03::RVMMode::prepareRun() load"      (this fn)        — model load only
	//
	//   [FPS]   line every 30 frames                                         — moving fps
	//   [PerFrame] line every frame                                          — infer + comp
	//
	// Identity (approx, ignoring tiny logging overhead):
	//   acc_lv02_01_main_loop_total_ ≈ max(tensor_ch.pop wait, 0) + acc_lv02_01_02_main_decode_ + acc_lv02_01_03_main_infer_ + acc_lv02_01_04_main_composite_
	//   acc_lv02_01_04_main_composite_ ≈ resize_alpha + resize_frame + blend + upscale + writer
	// ------------------------------------------------------------------------- */
	using sa = helmsman::utils::timing::StageAccumulator;

	sa acc_lv02_01_main_loop_total_{"Lv02-01::main::loop_total"};
	sa acc_lv02_01_01_worker_preprocess_{"  Lv02-01-01::worker::preprocess"};
	sa acc_lv02_01_02_main_decode_{"  Lv02-01-02::main::decode"};
	sa acc_lv02_01_03_main_infer_{"  Lv02-01-03::main::infer"};
	sa acc_lv02_01_04_main_composite_{"  Lv02-01-04::main::composite"};

	sa acc_lv02_01_04_01_resize_alpha_{"    Lv02-01-04-01::comp::resize_alpha"};
	sa acc_lv02_01_04_02_resize_frame_{"    Lv02-01-04-02::comp::resize_frame"};
	sa acc_lv02_01_04_03_blend_{"    Lv02-01-04-03::comp::blend"};
	sa acc_lv02_01_04_04_upscale_{"    Lv02-01-04-04::comp::upscale"};
	sa acc_lv02_01_04_05_writer_{"    Lv02-01-04-05::comp::writer"};
	sa acc_lv02_01_04_06_drm_{"    Lv02-01-04-06::comp::drm_show"};
};