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
// no-hw-frontend.cpp — OpenCV software-decode Frontend subclass (fallback)
//
// =============================================================================

#include "pipeline/stages/frontend/no-hw-frontend.h"

#include <cstdio>
#include <stdexcept>

NoHwFrontend::NoHwFrontend(const std::string& video_path, bool use_pipeline)
    : FrontendBase(false, use_pipeline) {
    if (!cv_cap_.open(video_path)) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }

    int w = static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cv_cap_.get(cv::CAP_PROP_FPS);

    SetSourceProperties(w, h, fps);
}

bool NoHwFrontend::ReadFrame(cv::Mat& cpu_frame, HardwareFrame& /*hw_frame*/) {
    if (!cv_cap_.isOpened()) {
        return false;
    }
    return cv_cap_.read(cpu_frame);
}
