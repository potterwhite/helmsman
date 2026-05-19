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
// frontend.h — Frontend router for the matting pipeline
//
// The Frontend orchestrates three internal modules:
//   1. _IInputSource — reads raw compressed data from the signal source
//   2. _IFrameDecoder — decodes compressed packets into frames
//   3. _Preprocessor — converts frames into TensorData for inference
//
// Phase 1: MPP path: _IInputSource → _IFrameDecoder → HardwareFrame (pure separation)
// Phase 2: MPP path + RGA NV12→BGR (hardware decode + color convert)
// OpenCV path: cv::VideoCapture directly (fallback)
//
// =============================================================================

#pragma once

#include <memory>
#include <opencv2/videoio.hpp>
#include "pipeline/stages/frontend/input-source/i-input-source.h"
#include "pipeline/stages/frontend/decoder/i-frame-decoder.h"
#include "pipeline/stages/frontend/preprocess/preprocessor.h"
#include "RGAKit/rga_cvtcolor.h"
#include "common/data_structure.h"

class Frontend {
public:
    // MPP hardware path constructor.
    Frontend(std::unique_ptr<_IInputSource> source,
             std::unique_ptr<_IFrameDecoder> decoder);

    // OpenCV shortcut path constructor (Phase 1 only).
    explicit Frontend(const std::string& video_path);

    ~Frontend();

    // Non-copyable, non-movable (owned by unique_ptr in Pipeline).
    Frontend(const Frontend&) = delete;
    Frontend& operator=(const Frontend&) = delete;
    Frontend(Frontend&&) = delete;
    Frontend& operator=(Frontend&&) = delete;

    // Read the next decoded frame.
    // For MPP path: fills hw_frame, cpu_frame is empty.
    // For OpenCV path: fills cpu_frame, hw_frame.fd = -1.
    // Returns false on EOF or error.
    bool readFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame);

    // Preprocess a CPU frame into TensorData (OpenCV path only).
    TensorData preprocess(const cv::Mat& frame, size_t model_w, size_t model_h);

    // Whether the MPP hardware path is active.
    bool isHardwarePath() const { return use_hardware_; }

    // Source properties (available after construction).
    int width() const;
    int height() const;
    double fps() const;

private:
    bool use_hardware_ = false;

    // MPP hardware path
    std::unique_ptr<_IInputSource> source_;
    std::unique_ptr<_IFrameDecoder> decoder_;

    // RGA NV12→BGR converter (hardware path only)
    std::unique_ptr<helmsman::rgakit::RgaCvtColor> nv12_to_bgr_;
    // BGR output buffer for RGA conversion (DMA-backed)
    cv::Mat bgr_buf_;

    // OpenCV shortcut path (Phase 1)
    cv::VideoCapture cv_cap_;

    // Shared preprocessor (both paths)
    _Preprocessor preprocessor_;
};
