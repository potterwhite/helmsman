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
// frontend-base.h — Frontend base class (Template Method pattern)
//
// Owns the pipeline infrastructure: preprocessing, prefetch worker thread,
// and double-buffer orchestration. Subclasses implement _ReadFrame() to
// supply decoded frames via the platform-specific decode path.
//
// Use FrontendBase::Create() to instantiate the correct subclass at runtime.
//
// =============================================================================

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <opencv2/core.hpp>
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/stages/frontend/stages/02-decoder/base-frame-decoder.h"  // HardwareFrame
#include "pipeline/stages/frontend/stages/04-preprocess/preprocessor.h"

/**
 * Result from _ReadFrame() — one decoded frame.
 */
struct ReadResult {
    cv::Mat frame;
    HardwareFrame hw_frame;
};

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
 * Base class for the Frontend stage (Template Method pattern).
 *
 * Owns the pipeline infrastructure: preprocessing, prefetch worker thread,
 * and double-buffer orchestration. The algorithm skeleton in ProcessOneFrame()
 * is fixed; subclasses implement _ReadFrame() to supply decoded frames.
 */
class FrontendBase {
public:
    virtual ~FrontendBase();

    // Non-copyable, non-movable (owned by unique_ptr in Pipeline).
    FrontendBase(const FrontendBase&) = delete;
    FrontendBase& operator=(const FrontendBase&) = delete;
    FrontendBase(FrontendBase&&) = delete;
    FrontendBase& operator=(FrontendBase&&) = delete;

    // Unified frame processing interface (Template Method).
    // In sync mode: reads and preprocesses one frame on the calling thread.
    // In pipeline mode: returns the next preprocessed frame from the prefetch pipeline.
    // Returns std::nullopt on EOF.
    std::optional<FrameResult> ProcessOneFrame(int model_w, int model_h);

    // Signal the prefetch worker to stop. Safe to call multiple times.
    void Stop();

    // Access the preprocess timing accumulator (thread-safe record(), main-thread report()).
    const helmsman::utils::timing::StageAccumulator& preprocess_acc() const;

    // Access the resize timing accumulator (sub-step of preprocess).
    const helmsman::utils::timing::StageAccumulator& resize_acc() const;

    // Whether the hardware decode path is active.
    bool IsHardwarePath() const;

    // Source properties (available after construction).
    int width() const;
    int height() const;
    double fps() const;

    // Static factory: creates the platform-specific FrontendBase subclass.
    // If use_hardware is true, uses the hardware decode path (platform-dependent).
    // If use_pipeline is true, enables the prefetch worker thread.
    // Throws std::runtime_error on failure.
    static std::unique_ptr<FrontendBase> Create(const std::string& input_path,
                                                bool use_hardware,
                                                bool use_pipeline = false);

protected:
    // Subclass constructor: sets hardware flag and pipeline mode.
    // Source properties default to 0. Subclasses call SetSourceProperties() after opening the source.
    explicit FrontendBase(bool use_hardware, bool use_pipeline);

    // Set source dimensions and fps. Called by subclasses after opening the source.
    void SetSourceProperties(int width, int height, double fps);

    // --- Subclass contract (Template Method hooks) ---

    // Read the next decoded frame. Return std::nullopt on EOF.
    // Called from the main thread (in both sync and pipeline modes).
    virtual std::optional<ReadResult> _ReadFrame() = 0;

    // Open the source and set source properties (width, height, fps).
    // Called from the constructor. Throws std::runtime_error on failure.
    virtual void _OpenSource(const std::string& input_path) = 0;

private:
    // Worker thread entry point (pipeline mode only).
    void _WorkerLoop(int model_w, int model_h);

    // Source properties
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    bool use_hardware_;

    // Pipeline infrastructure
    Preprocessor preprocessor_;
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
    helmsman::utils::timing::StageAccumulator acc_lv03_02_worker_preprocess_{
        "  Lv03-02::worker::preprocess"};
};
