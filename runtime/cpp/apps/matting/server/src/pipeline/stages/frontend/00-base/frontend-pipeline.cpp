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
// frontend-pipeline.cpp — FrontendPipeline implementation
//
// Pipeline logic extracted from FrontendBase. Handles sync mode (direct read +
// preprocess) and pipeline mode (prefetch worker thread with double-buffering).
//
// =============================================================================

#include "pipeline/stages/frontend/00-base/frontend-pipeline.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
FrontendPipeline::FrontendPipeline(Reader reader, bool use_pipeline)
    : reader_(std::move(reader)), use_pipeline_(use_pipeline) {
    if (use_pipeline_) {
        raw_ch_ = std::make_unique<SingleSlotChannel<cv::Mat>>();
        tensor_ch_ = std::make_unique<SingleSlotChannel<TensorData>>();
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FrontendPipeline::~FrontendPipeline() {
    Stop();
}

// ---------------------------------------------------------------------------
// preprocess — convert a BGR frame into TensorData
// ---------------------------------------------------------------------------
TensorData FrontendPipeline::preprocess(const cv::Mat& frame, int model_w, int model_h) {
    return preprocessor_.preprocess(frame, model_w, model_h);
}

// ---------------------------------------------------------------------------
// _WorkerLoop — worker thread entry point
// ---------------------------------------------------------------------------
void FrontendPipeline::_WorkerLoop(int model_w, int model_h) {
    while (true) {
        auto frame_opt = raw_ch_->pop();
        if (!frame_opt)
            break;

        helmsman::utils::timing::ManualTimer t;
        t.start();
        auto tensor = preprocessor_.preprocess(*frame_opt, model_w, model_h);
        acc_lv03_02_worker_preprocess_.record(t.stop());

        tensor_ch_->push(std::move(tensor));
    }
    tensor_ch_->close();
}

// ---------------------------------------------------------------------------
// ProcessOneFrame — unified frame processing interface
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontendPipeline::ProcessOneFrame(int model_w, int model_h) {
    if (!use_pipeline_) {
        // Sync mode: read and preprocess on calling thread
        auto read_result = reader_();
        if (!read_result)
            return std::nullopt;

        helmsman::utils::timing::ManualTimer t;
        t.start();
        auto tensor = preprocessor_.preprocess(read_result->frame, model_w, model_h);
        acc_lv03_02_worker_preprocess_.record(t.stop());

        FrameResult result;
        result.frame = std::move(read_result->frame);
        result.hw_frame = read_result->hw_frame;
        result.tensor = std::move(tensor);
        return result;
    }

    // Pipeline mode
    if (pipeline_eof_)
        return std::nullopt;

    if (!pipeline_started_) {
        // Phase 1: bootstrap — read frame 1, preprocess, read frame 2
        pipeline_started_ = true;

        auto read1 = reader_();
        if (!read1)
            return std::nullopt;

        cv::Mat frame_1 = std::move(read1->frame);
        HardwareFrame hw_frame_1 = read1->hw_frame;

        // Start the worker thread now that we have the first frame
        prefetch_worker_ = std::thread(&FrontendPipeline::_WorkerLoop, this, model_w, model_h);

        // Push frame 1 to worker for preprocessing
        raw_ch_->push(frame_1);

        // Pop tensor 1 (blocks until worker finishes)
        auto tensor_1 = tensor_ch_->pop();
        if (!tensor_1) {
            pipeline_eof_ = true;
            return std::nullopt;
        }

        // Read frame 2 and push to worker (dual-buffer overlap)
        auto read2 = reader_();
        if (read2) {
            next_frame_ = std::move(read2->frame);
            next_hw_frame_ = read2->hw_frame;
            raw_ch_->push(next_frame_);
        } else {
            raw_ch_->close();
        }

        FrameResult result;
        result.frame = std::move(frame_1);
        result.hw_frame = hw_frame_1;
        result.tensor = std::move(*tensor_1);
        return result;
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
    auto read_next = reader_();
    if (read_next) {
        next_frame_ = std::move(read_next->frame);
        next_hw_frame_ = read_next->hw_frame;
        raw_ch_->push(next_frame_);
    } else {
        raw_ch_->close();
    }

    FrameResult result;
    result.frame = std::move(return_frame);
    result.hw_frame = return_hw_frame;
    result.tensor = std::move(*tensor);
    return result;
}

// ---------------------------------------------------------------------------
// Stop — signal prefetch worker to stop
// ---------------------------------------------------------------------------
void FrontendPipeline::Stop() {
    if (raw_ch_)
        raw_ch_->close();
    if (prefetch_worker_.joinable())
        prefetch_worker_.join();
}

// ---------------------------------------------------------------------------
// preprocess_acc — access preprocess timing accumulator
// ---------------------------------------------------------------------------
const helmsman::utils::timing::StageAccumulator& FrontendPipeline::preprocess_acc() const {
    return acc_lv03_02_worker_preprocess_;
}

const helmsman::utils::timing::StageAccumulator& FrontendPipeline::resize_acc() const {
    return preprocessor_.resize_acc();
}
