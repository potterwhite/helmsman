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
// frontend.h — Frontend for the matting pipeline
//
// The Frontend orchestrates four internal modules:
//   1. BaseInputSource — reads raw compressed data from the signal source
//   2. BaseFrameDecoder — decodes compressed packets into frames
//   3. BaseColorConverter — converts hardware frames to BGR cv::Mat
//   4. Preprocessor — converts frames into TensorData for inference
//
// Hardware path: BaseInputSource → BaseFrameDecoder → BaseColorConverter → BGR
// OpenCV path: cv::VideoCapture directly (fallback)
//
// =============================================================================

#pragma once

#include <memory>
#include <optional>
#include <opencv2/videoio.hpp>
#include <thread>
#include "Utils/timing/timer.h"
#include "common/types.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/stages/frontend/01-input-source/base-input-source.h"
#include "pipeline/stages/frontend/02-decoder/base-frame-decoder.h"
#include "pipeline/stages/frontend/03-color-convert/base-color-converter.h"
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

class Frontend {
public:
    // Hardware decode path constructor (DI).
    Frontend(std::unique_ptr<BaseInputSource> source,
             std::unique_ptr<BaseFrameDecoder> decoder,
             std::unique_ptr<BaseColorConverter> converter,
             bool use_pipeline = false);

    // OpenCV shortcut path constructor.
    explicit Frontend(const std::string& video_path, bool use_pipeline = false);

    ~Frontend();

    // Non-copyable, non-movable (owned by unique_ptr in Pipeline).
    Frontend(const Frontend&) = delete;
    Frontend& operator=(const Frontend&) = delete;
    Frontend(Frontend&&) = delete;
    Frontend& operator=(Frontend&&) = delete;

    // Read the next decoded frame.
    // For hardware path: fills cpu_frame via color converter.
    // For OpenCV path: fills cpu_frame directly.
    // Returns false on EOF or error.
    bool ReadFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame);

    // Preprocess a CPU frame into TensorData.
    TensorData preprocess(const cv::Mat& frame, size_t model_w, size_t model_h);

    // Unified frame processing interface.
    // In sync mode: reads and preprocesses one frame on the calling thread.
    // In pipeline mode: returns the next preprocessed frame from the prefetch pipeline.
    // Returns std::nullopt on EOF.
    std::optional<FrameResult> ProcessOneFrame(size_t model_w, size_t model_h);

    // Signal the prefetch worker to stop. Safe to call multiple times.
    // Closes the internal channel and joins the worker thread.
    void Stop();

    // Access the preprocess timing accumulator (thread-safe record(), main-thread report()).
    const helmsman::utils::timing::StageAccumulator& preprocess_acc() const;

    // Whether the hardware decode path is active.
    bool IsHardwarePath() const { return use_hardware_; }

    // Source properties (available after construction).
    int width() const;
    int height() const;
    double fps() const;

private:
    // Worker thread entry point. Loops: pop raw frame from raw_ch_ -> preprocess -> push tensor to tensor_ch_.
    void PrefetchWorkerLoop(size_t model_w, size_t model_h);

    bool use_hardware_ = false;

    // Hardware decode path
    std::unique_ptr<BaseInputSource> source_;
    std::unique_ptr<BaseFrameDecoder> decoder_;
    std::unique_ptr<BaseColorConverter> color_converter_;

    // OpenCV shortcut path
    cv::VideoCapture cv_cap_;

    // Shared preprocessor (both paths)
    Preprocessor preprocessor_;

    // Pipeline mode
    bool use_pipeline_ = false;

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
