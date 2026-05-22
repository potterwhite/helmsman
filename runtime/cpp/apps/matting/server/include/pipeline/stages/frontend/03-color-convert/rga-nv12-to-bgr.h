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
// rga-nv12-to-bgr.h — RGA hardware NV12→BGR color converter (internal)
//
// Uses Rockchip RGA to convert NV12 DMA buffer frames to BGR cv::Mat.
//
// =============================================================================

#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include "pipeline/stages/frontend/03-color-convert/base-color-converter.h"
#include "RGAKit/rga_cvtcolor.h"

class RgaNv12ToBgr : public BaseColorConverter {
public:
    RgaNv12ToBgr();
    ~RgaNv12ToBgr() override;

    // Non-copyable, movable.
    RgaNv12ToBgr(const RgaNv12ToBgr&) = delete;
    RgaNv12ToBgr& operator=(const RgaNv12ToBgr&) = delete;
    RgaNv12ToBgr(RgaNv12ToBgr&&) noexcept;
    RgaNv12ToBgr& operator=(RgaNv12ToBgr&&) noexcept;

    // Convert NV12 hardware frame to BGR cv::Mat via RGA.
    bool convert(const HardwareFrame& hw_frame, cv::Mat& cpu_frame) override;

private:
    std::unique_ptr<helmsman::rgakit::RgaCvtColor> cvt_;
    cv::Mat bgr_buf_;
};
