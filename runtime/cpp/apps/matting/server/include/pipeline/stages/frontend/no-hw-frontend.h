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
// no-hw-frontend.h — OpenCV software-decode Frontend subclass (fallback)
//
// Uses cv::VideoCapture for software decode when no hardware decoder is available.
//
// =============================================================================

#pragma once

#include <opencv2/videoio.hpp>
#include <string>
#include "pipeline/stages/frontend/frontend.h"

class NoHwFrontend : public FrontendBase {
public:
    // Opens the video with cv::VideoCapture.
    // Throws std::runtime_error on failure.
    explicit NoHwFrontend(const std::string& video_path, bool use_pipeline = false);

protected:
    bool ReadFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame) override;

private:
    cv::VideoCapture cv_cap_;
};
