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

#include "pipeline/stages/frontend/frontend-core/impl/no-hw-frontend.h"

#include <stdexcept>

NoHwFrontend::NoHwFrontend(const std::string& video_path, bool multithread_enabled)
    : FrontEnd(false, multithread_enabled) {
    _OpenSource(video_path);
}

void NoHwFrontend::_OpenSource(const std::string& input_path) {
    if (!cv_cap_.open(input_path)) {
        throw std::runtime_error("Failed to open video: " + input_path);
    }

    int w = static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cv_cap_.get(cv::CAP_PROP_FPS);

    SetSourceProperties(w, h, fps);
}

// ---------------------------------------------------------------------------
// Stage 01: cv::VideoCapture handles demux + decode + color convert atomically.
// Returns complete BGR frame in result.frame. Stages 02 and 03 are no-ops.
// ---------------------------------------------------------------------------
bool NoHwFrontend::_ReadInputSource01(RawPacket& pkt, ReadResult& result) {
    if (!cv_cap_.isOpened()) {
        pkt.is_eof = true;
        return true;
    }

    if (!cv_cap_.read(result.frame)) {
        pkt.is_eof = true;
        return true;
    }

    return true;
}

// Stage 02: no-op — frame already decoded by cv::VideoCapture in Stage 01.
bool NoHwFrontend::_DecodeFrame02(const RawPacket& /*pkt*/, ReadResult& result) {
    return !result.frame.empty();
}

// Stage 03: no-op — frame already BGR from cv::VideoCapture in Stage 01.
bool NoHwFrontend::_ConvertToBgr03(ReadResult& result) {
    return !result.frame.empty();
}
