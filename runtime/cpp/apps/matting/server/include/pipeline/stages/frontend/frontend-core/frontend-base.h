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
// Owns the multithread infrastructure: preprocessing, prefetch worker thread,
// and double-buffer orchestration. Subclasses override stage methods to
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
#include "pipeline/stages/frontend/stages/01-input-source/base-input-source.h"  // RawPacket
#include "pipeline/stages/frontend/stages/02-decoder/base-frame-decoder.h"  // HardwareFrame
#include "pipeline/stages/frontend/stages/04-preprocess/preprocessor.h"

/**
 * Result from _ReadInputSource01() — one decoded frame.
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
 * Owns the multithread infrastructure: preprocessing, prefetch worker thread,
 * and double-buffer orchestration. The algorithm skeleton in ProcessOneFrame()
 * is fixed; subclasses override stage methods to supply decoded frames.
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
    // Dispatches to _ProcessSync or _ProcessMultithread based on multithread_enabled_.
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
    // If multithread_enabled is true, enables the prefetch worker thread.
    // Throws std::runtime_error on failure.
    static std::unique_ptr<FrontendBase> Create(const std::string& input_path,
                                                bool use_hardware,
                                                bool multithread_enabled = false);

protected:
    // Subclass constructor: sets hardware flag and multithread mode.
    // Source properties default to 0. Subclasses call SetSourceProperties() after opening the source.
    explicit FrontendBase(bool use_hardware, bool multithread_enabled);

    // Set source dimensions and fps. Called by subclasses after opening the source.
    void SetSourceProperties(int width, int height, double fps);

    // --- Stage 01-03: Frame decode pipeline (virtual, subclass-overridable) ---

    // Stage 01: Read input source. Default implementation chains _ReadRawPacket +
    // _DecodeFrame02 + _ConvertToBgr with a retry loop for hardware decoders.
    // NoHwFrontend overrides this directly since cv::VideoCapture handles 01-03 atomically.
    // Returns false on EOF.
    virtual bool _ReadInputSource01(ReadResult& result);

    // Stage 02: Decode one compressed packet into a hardware frame.
    // Returns true if a decoded frame is available, false if decoder needs more data.
    virtual bool _DecodeFrame02(const RawPacket& pkt, HardwareFrame& hw_frame);

    // Stage 03: Convert hardware frame to BGR cv::Mat.
    // Returns true on success, false on failure.
    virtual bool _ConvertToBgr03(const HardwareFrame& hw_frame, cv::Mat& frame);

    // Open the source and set source properties (width, height, fps).
    // Called from the constructor. Throws std::runtime_error on failure.
    virtual void _OpenSource(const std::string& input_path) = 0;

private:
    // Stage 04: Preprocess BGR frame into TensorData for inference. Non-virtual.
    TensorData _PreprocessForInference04(const cv::Mat& frame, int model_w, int model_h);

    // Read one raw packet from the input source. Used by _ReadInputSource01 default impl.
    virtual bool _ReadRawPacket(RawPacket& pkt);

    // Sync mode: 4 stages on calling thread.
    std::optional<FrameResult> _ProcessSync(int model_w, int model_h);

    // Multithread mode: stages 01-03 on main thread, stage 04 on worker thread.
    std::optional<FrameResult> _ProcessMultithread(int model_w, int model_h);

    // Worker thread entry point (multithread mode only). Runs stage 04.
    void _MultithreadWorkerLoop(int model_w, int model_h);

    // Source properties
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    bool use_hardware_;

    // Multithread infrastructure
    Preprocessor preprocessor_;
    bool multithread_enabled_;

    // Channel infrastructure (multithread mode only)
    std::unique_ptr<SingleSlotChannel<cv::Mat>> raw_ch_;
    std::unique_ptr<SingleSlotChannel<TensorData>> tensor_ch_;
    std::thread prefetch_worker_;

    // Multithread state
    bool mt_eof_ = false;
    bool mt_started_ = false;

    // Multithread-mode buffered results (frame N+1 decoded while processing frame N)
    cv::Mat next_frame_;
    HardwareFrame next_hw_frame_;
    HardwareFrame stored_hw_frame_;  // hw_frame for current frame (stages 01-03 done, stage 04 pending)

    // Timing
    helmsman::utils::timing::StageAccumulator acc_lv03_02_worker_preprocess_{
        "  Lv03-02::worker::preprocess"};
};
