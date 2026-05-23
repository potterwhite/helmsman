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
// frontend.cpp — FrontendBase shared implementation
//
// Platform-specific frame reading is delegated to subclasses via ReadFrame().
// All shared logic (preprocessing, pipeline mode, timing) lives here.
//
// =============================================================================

#include "pipeline/stages/frontend/frontend.h"

// ---------------------------------------------------------------------------
// Protected constructor
// ---------------------------------------------------------------------------
FrontendBase::FrontendBase(bool use_hardware, bool use_pipeline)
    : width_(0), height_(0), fps_(0.0),
      use_hardware_(use_hardware), use_pipeline_(use_pipeline) {
    if (use_pipeline_) {
        raw_ch_ = std::make_unique<SingleSlotChannel<cv::Mat>>();
        tensor_ch_ = std::make_unique<SingleSlotChannel<TensorData>>();
    }
}

// ---------------------------------------------------------------------------
// SetSourceProperties — called by subclasses after opening the source
// ---------------------------------------------------------------------------
void FrontendBase::SetSourceProperties(int width, int height, double fps) {
    width_ = width;
    height_ = height;
    fps_ = fps;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FrontendBase::~FrontendBase() {
    Stop();
}

// ---------------------------------------------------------------------------
// preprocess — convert a BGR frame into TensorData
// ---------------------------------------------------------------------------
TensorData FrontendBase::preprocess(const cv::Mat& frame, int model_w, int model_h) {
    return preprocessor_.preprocess(frame, model_w, model_h);
}

// ---------------------------------------------------------------------------
// PrefetchWorkerLoop — worker thread entry point
// ---------------------------------------------------------------------------
void FrontendBase::PrefetchWorkerLoop(int model_w, int model_h) {
    while (true) {
        auto frame_opt = raw_ch_->pop();
        if (!frame_opt)
            break;

        helmsman::utils::timing::ManualTimer t;
        t.start();
        auto tensor = preprocessor_.preprocess(*frame_opt, model_w, model_h);
        preprocess_acc_.record(t.stop());

        tensor_ch_->push(std::move(tensor));
    }
    tensor_ch_->close();
}

// ---------------------------------------------------------------------------
// ProcessOneFrame — unified frame processing interface
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontendBase::ProcessOneFrame(int model_w, int model_h) {
    if (!use_pipeline_) {
        // Sync mode: read and preprocess on calling thread
        cv::Mat frame;
        HardwareFrame hw_frame;
        if (!ReadFrame(frame, hw_frame))
            return std::nullopt;

        helmsman::utils::timing::ManualTimer t;
        t.start();
        auto tensor = preprocessor_.preprocess(frame, model_w, model_h);
        preprocess_acc_.record(t.stop());

        return FrameResult{std::move(frame), hw_frame, std::move(tensor)};
    }

    // Pipeline mode
    if (pipeline_eof_)
        return std::nullopt;

    if (!pipeline_started_) {
        // Phase 1: bootstrap — read frame 1, preprocess, read frame 2
        pipeline_started_ = true;

        cv::Mat frame_1;
        HardwareFrame hw_frame_1;
        if (!ReadFrame(frame_1, hw_frame_1))
            return std::nullopt;

        // Start the worker thread now that we have the first frame
        prefetch_worker_ = std::thread(&FrontendBase::PrefetchWorkerLoop, this, model_w, model_h);

        // Push frame 1 to worker for preprocessing
        raw_ch_->push(frame_1);

        // Pop tensor 1 (blocks until worker finishes)
        auto tensor_1 = tensor_ch_->pop();
        if (!tensor_1) {
            pipeline_eof_ = true;
            return std::nullopt;
        }

        // Read frame 2 and push to worker (dual-buffer overlap)
        bool has_next = ReadFrame(next_frame_, next_hw_frame_);
        if (has_next) {
            raw_ch_->push(next_frame_);
        } else {
            raw_ch_->close();
        }

        return FrameResult{std::move(frame_1), hw_frame_1, std::move(*tensor_1)};
    }

    // Phase 2: subsequent calls — pop tensor for buffered frame, read next
    auto tensor = tensor_ch_->pop();
    if (!tensor) {
        pipeline_eof_ = true;
        return std::nullopt;
    }

    // Save current buffered frame to return
    cv::Mat return_frame = std::move(next_frame_);
    HardwareFrame return_hw_frame = next_hw_frame_;

    // Read the next frame and push to worker
    bool has_next = ReadFrame(next_frame_, next_hw_frame_);
    if (has_next) {
        raw_ch_->push(next_frame_);
    } else {
        raw_ch_->close();
    }

    return FrameResult{std::move(return_frame), return_hw_frame, std::move(*tensor)};
}

// ---------------------------------------------------------------------------
// Stop — signal prefetch worker to stop
// ---------------------------------------------------------------------------
void FrontendBase::Stop() {
    if (raw_ch_)
        raw_ch_->close();
    if (prefetch_worker_.joinable())
        prefetch_worker_.join();
}

// ---------------------------------------------------------------------------
// preprocess_acc — access preprocess timing accumulator
// ---------------------------------------------------------------------------
const helmsman::utils::timing::StageAccumulator& FrontendBase::preprocess_acc() const {
    return preprocess_acc_;
}
