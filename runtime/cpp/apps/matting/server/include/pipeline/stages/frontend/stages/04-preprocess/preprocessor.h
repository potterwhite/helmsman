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
// preprocessor.h — Image preprocessor for inference (internal to Frontend)
//
// Converts a BGR cv::Mat frame into a TensorData structure ready for the
// inference engine. Migrated from the old ImageFrontend class.
//
// =============================================================================

#pragma once

#include <string>
#include "CVKit/base/base.h"
#include "Utils/timing/timer.h"
#include "pipeline/stages/frontend/stages/04-preprocess/base-preprocessor.h"

class Preprocessor : public BasePreprocessor {
public:
    using sa = helmsman::utils::timing::StageAccumulator;

    Preprocessor();
    ~Preprocessor() override;

    // Preprocess a BGR frame into a TensorData for inference.
    TensorData preprocess(const cv::Mat& bgr_frame,
                          int model_width,
                          int model_height) override;

    // Configure output binary dump directory.
    void SetOutputBinPath(const std::string& path);
    void SetDumpEnabled(bool enabled);

    // Access timing accumulators (thread-safe record(), main-thread report()).
    const sa& resize_acc() const { return acc_resize_; }

private:
    std::string output_bin_path_;
    bool dump_enabled_ = false;
    helmsman::cvkit::Base cvkit_;

    sa acc_resize_{"  Lv03-02-01::worker::preprocess::resize"};
};
