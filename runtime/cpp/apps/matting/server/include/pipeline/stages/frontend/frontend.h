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

// =============================================================================
// frontend.h — Frontend base class for the matting pipeline
//
// FrontendBase is the abstract base class. Platform-specific subclasses
// override ReadFrame() to provide hardware-decode or OpenCV software decode.
//
// Shared logic (preprocessing, pipeline mode, timing) lives in the base class.
//
// Use FrontendBase::Create() to instantiate the correct subclass at runtime.
//
// =============================================================================

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <thread>
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/stages/frontend/02-decoder/base-frame-decoder.h"  // HardwareFrame
#include "pipeline/stages/frontend/04-preprocess/preprocessor.h"

/**
 * Result of processing one frame through the Frontend pipeline.
 * Contains both the raw frame (for compositing) and the preprocessed tensor (for inference).
 */
struct FrameResult {
    cv::Mat frame;          // Raw BGR frame (for compositing)
    HardwareFrame hw_frame; // Hardware frame metadata (for hardware path)
    TensorData tensor;      // Preprocessed tensor (for inference)
};

/**
 * Abstract base class for the Frontend stage.
 *
 * Subclasses override ReadFrame() to provide platform-specific frame decoding.
 * All other methods (preprocessing, pipeline mode, timing) are shared.
 */
class FrontendBase {
public:
    virtual ~FrontendBase();

    // Non-copyable, non-movable (owned by unique_ptr in Pipeline).
    FrontendBase(const FrontendBase&) = delete;
    FrontendBase& operator=(const FrontendBase&) = delete;
    FrontendBase(FrontendBase&&) = delete;
    FrontendBase& operator=(FrontendBase&&) = delete;

    // Preprocess a CPU frame into TensorData.
    TensorData preprocess(const cv::Mat& frame, int model_w, int model_h);

    // Unified frame processing interface.
    // In sync mode: reads and preprocesses one frame on the calling thread.
    // In pipeline mode: returns the next preprocessed frame from the prefetch pipeline.
    // Returns std::nullopt on EOF.
    std::optional<FrameResult> ProcessOneFrame(int model_w, int model_h);

    // Signal the prefetch worker to stop. Safe to call multiple times.
    // Closes the internal channel and joins the worker thread.
    void Stop();

    // Access the preprocess timing accumulator (thread-safe record(), main-thread report()).
    const helmsman::utils::timing::StageAccumulator& preprocess_acc() const;

    // Whether the hardware decode path is active.
    bool IsHardwarePath() const { return use_hardware_; }

    // Source properties (available after construction).
    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }

    // Static factory: creates the platform-specific FrontendBase subclass.
    // If use_hardware is true, uses the hardware decode path (platform-dependent).
    // If use_pipeline is true, enables the prefetch worker thread.
    // Throws std::runtime_error on failure.
    static std::unique_ptr<FrontendBase> Create(const std::string& input_path,
                                                bool use_hardware,
                                                bool use_pipeline = false);

protected:
    // Subclass constructor: sets pipeline mode. Source properties default to 0.
    // Subclasses call SetSourceProperties() after opening the source.
    FrontendBase(bool use_hardware, bool use_pipeline);

    // Set source dimensions and fps. Called by subclasses after opening the source.
    void SetSourceProperties(int width, int height, double fps);

    // Pure virtual: read the next decoded frame.
    // For hardware path: fills cpu_frame via color converter.
    // For OpenCV path: fills cpu_frame directly.
    // Returns false on EOF or error.
    virtual bool ReadFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame) = 0;

private:
    // Worker thread entry point. Loops: pop raw frame from raw_ch_ -> preprocess -> push tensor to tensor_ch_.
    void PrefetchWorkerLoop(int model_w, int model_h);

    int width_;
    int height_;
    double fps_;
    bool use_hardware_;

    // Shared preprocessor (both paths)
    Preprocessor preprocessor_;

    // Pipeline mode
    bool use_pipeline_;

    // Channel infrastructure (pipeline mode only)
    std::unique_ptr<SingleSlotChannel<cv::Mat>> raw_ch_;
    std::unique_ptr<SingleSlotChannel<TensorData>> tensor_ch_;
    std::thread prefetch_worker_;

    // Pipeline state
    bool pipeline_eof_ = false;
    bool pipeline_started_ = false;

    // Pipeline-mode buffered results (frame N+1 decoded while processing frame N)
    cv::Mat next_frame_;
    HardwareFrame next_hw_frame_;

    // Timing
    helmsman::utils::timing::StageAccumulator preprocess_acc_{"  Lv02-01-01::worker::preprocess"};
};
