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
#include <opencv2/videoio.hpp>
#include "pipeline/stages/frontend/01-input-source/base-input-source.h"
#include "pipeline/stages/frontend/02-decoder/base-frame-decoder.h"
#include "pipeline/stages/frontend/04-preprocess/preprocessor.h"
#include "pipeline/stages/frontend/03-color-convert/base-color-converter.h"
#include "common/types.h"

class Frontend {
public:
    // Hardware decode path constructor (DI).
    Frontend(std::unique_ptr<BaseInputSource> source,
             std::unique_ptr<BaseFrameDecoder> decoder,
             std::unique_ptr<BaseColorConverter> converter);

    // OpenCV shortcut path constructor.
    explicit Frontend(const std::string& video_path);

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

    // Whether the hardware decode path is active.
    bool IsHardwarePath() const { return use_hardware_; }

    // Source properties (available after construction).
    int width() const;
    int height() const;
    double fps() const;

private:
    bool use_hardware_ = false;

    // Hardware decode path
    std::unique_ptr<BaseInputSource> source_;
    std::unique_ptr<BaseFrameDecoder> decoder_;
    std::unique_ptr<BaseColorConverter> color_converter_;

    // OpenCV shortcut path
    cv::VideoCapture cv_cap_;

    // Shared preprocessor (both paths)
    Preprocessor preprocessor_;
};
