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
// frontend-pipeline.h — Reusable pipeline utility for Frontend subclasses
//
// Encapsulates preprocessing, prefetch worker thread, and double-buffer
// orchestration. Subclasses compose this with a Reader callback.
//
// =============================================================================

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/stages/frontend/00-base/frontend.h"
#include "pipeline/stages/frontend/02-decoder/base-frame-decoder.h"  // HardwareFrame
#include "pipeline/stages/frontend/04-preprocess/preprocessor.h"

/**
 * Result from the Reader callback — one decoded frame.
 */
struct ReadResult {
    cv::Mat frame;
    HardwareFrame hw_frame;
};

/**
 * Reusable pipeline utility for Frontend subclasses.
 *
 * Owns the Preprocessor and the pipeline infrastructure (channels, prefetch
 * worker thread, double-buffer orchestration). Subclasses provide a Reader
 * callback that supplies decoded frames.
 */
class FrontendPipeline {
public:
    // Reader callback: returns the next decoded frame, or std::nullopt on EOF.
    // Called ONLY from the main thread (during bootstrap and each ProcessOneFrame call).
    using Reader = std::function<std::optional<ReadResult>()>;

    // reader: frame-reading callback (owned by the subclass via lambda).
    // use_pipeline: if true, enables the prefetch worker thread with double-buffering.
    FrontendPipeline(Reader reader, bool use_pipeline);

    ~FrontendPipeline();

    // Non-copyable, non-movable (owns a thread).
    FrontendPipeline(const FrontendPipeline&) = delete;
    FrontendPipeline& operator=(const FrontendPipeline&) = delete;
    FrontendPipeline(FrontendPipeline&&) = delete;
    FrontendPipeline& operator=(FrontendPipeline&&) = delete;

    // Preprocess a CPU frame into TensorData.
    TensorData preprocess(const cv::Mat& frame, int model_w, int model_h);

    // Unified frame processing interface.
    // In sync mode: reads and preprocesses one frame on the calling thread.
    // In pipeline mode: returns the next preprocessed frame from the prefetch pipeline.
    // Returns std::nullopt on EOF.
    std::optional<FrameResult> ProcessOneFrame(int model_w, int model_h);

    // Signal the prefetch worker to stop. Safe to call multiple times.
    void Stop();

    // Access the preprocess timing accumulator.
    const helmsman::utils::timing::StageAccumulator& preprocess_acc() const;

private:
    // Worker thread entry point (pipeline mode only).
    void _WorkerLoop(int model_w, int model_h);

    Reader reader_;
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
    helmsman::utils::timing::StageAccumulator acc_lv03_02_worker_preprocess_{"  Lv03-02::worker::preprocess"};
};
