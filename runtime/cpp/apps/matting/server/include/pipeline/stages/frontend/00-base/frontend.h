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
// frontend.h — Frontend abstract interface for the matting pipeline
//
// FrontendBase is the pure abstract interface. Platform-specific subclasses
// own all internal execution logic (preprocessing, pipeline orchestration,
// timing) and implement the virtual methods.
//
// Use FrontendBase::Create() to instantiate the correct subclass at runtime.
//
// =============================================================================

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <opencv2/core.hpp>
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/stages/frontend/02-decoder/base-frame-decoder.h"  // HardwareFrame

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
 * Abstract interface for the Frontend stage.
 *
 * Subclasses own all implementation: frame reading, preprocessing,
 * pipeline orchestration, and timing.
 */
class FrontendBase {
public:
    virtual ~FrontendBase();

    // Non-copyable, non-movable (owned by unique_ptr in Pipeline).
    FrontendBase(const FrontendBase&) = delete;
    FrontendBase& operator=(const FrontendBase&) = delete;
    FrontendBase(FrontendBase&&) = delete;
    FrontendBase& operator=(FrontendBase&&) = delete;

    // Unified frame processing interface.
    // In sync mode: reads and preprocesses one frame on the calling thread.
    // In pipeline mode: returns the next preprocessed frame from the prefetch pipeline.
    // Returns std::nullopt on EOF.
    virtual std::optional<FrameResult> ProcessOneFrame(int model_w, int model_h) = 0;

    // Signal the prefetch worker to stop. Safe to call multiple times.
    virtual void Stop() = 0;

    // Access the preprocess timing accumulator (thread-safe record(), main-thread report()).
    virtual const helmsman::utils::timing::StageAccumulator& preprocess_acc() const = 0;

    // Access the resize timing accumulator (sub-step of preprocess).
    virtual const helmsman::utils::timing::StageAccumulator& resize_acc() const = 0;

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
    // Subclass constructor: sets hardware flag. Source properties default to 0.
    // Subclasses call SetSourceProperties() after opening the source.
    explicit FrontendBase(bool use_hardware);

    // Set source dimensions and fps. Called by subclasses after opening the source.
    void SetSourceProperties(int width, int height, double fps);

private:
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    bool use_hardware_;
};
