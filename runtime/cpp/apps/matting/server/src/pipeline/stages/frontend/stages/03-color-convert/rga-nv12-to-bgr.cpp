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
// rga-nv12-to-bgr.cpp — RGA hardware NV12→BGR color converter (internal)
//
// =============================================================================

#include "pipeline/stages/frontend/stages/03-color-convert/rga-nv12-to-bgr.h"

#include <cstdio>

RgaNv12ToBgr::RgaNv12ToBgr() {
	cvt_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaCvtColor>(
	    helmsman::rgakit::RgaPixelFormat::kNv12, helmsman::rgakit::RgaPixelFormat::kBgr888,
	    helmsman::rgakit::RgaCscMode::kYuvToRgbBt601Limit);
}

RgaNv12ToBgr::~RgaNv12ToBgr() = default;

RgaNv12ToBgr::RgaNv12ToBgr(RgaNv12ToBgr&&) noexcept = default;
RgaNv12ToBgr& RgaNv12ToBgr::operator=(RgaNv12ToBgr&&) noexcept = default;

bool RgaNv12ToBgr::convert(const HardwareFrame& hw_frame, cv::Mat& cpu_frame) {
	if (hw_frame.fd < 0 || !cvt_) {
		return false;
	}

	// Allocate BGR buffer on first frame (or dimension change)
	if (bgr_buf_.empty() || bgr_buf_.cols != hw_frame.width ||
	    bgr_buf_.rows != hw_frame.height) {
		bgr_buf_ = cv::Mat(hw_frame.height, hw_frame.width, CV_8UC3);
	}

	auto src = helmsman::rgakit::ImageDescriptor::FromFd(
	    hw_frame.fd, hw_frame.width, hw_frame.height,
	    helmsman::rgakit::RgaPixelFormat::kNv12);
	auto dst =
	    helmsman::rgakit::ImageDescriptor(bgr_buf_.data, bgr_buf_.cols, bgr_buf_.rows,
	                                      helmsman::rgakit::RgaPixelFormat::kBgr888);

	if (cvt_->Execute(src, dst)) {
		cpu_frame = bgr_buf_;
		return true;
	}

	fprintf(stderr, "[RgaNv12ToBgr] NV12→BGR conversion failed\n");
	return false;
}
