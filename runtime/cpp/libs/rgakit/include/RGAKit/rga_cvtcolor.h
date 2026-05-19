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
// RGAKit/rga_cvtColor.h — Hardware-accelerated color conversion via RGA
//
// Wraps the RGA imcvtcolor() function. Converts between pixel formats and
// color spaces using the RGA2 hardware color conversion unit.
//
// Common conversions in our RVM pipeline:
//
//   1. BGR → NV12:  For feeding frames to the MPP hardware encoder.
//      MPP only accepts NV12 (YCbCr 4:2:0 semi-planar), but OpenCV
//      outputs BGR. RGA does this conversion in ~1ms for 1080p.
//
//   2. NV12 → BGR:  For displaying decoded video frames.
//      MPP decoder outputs NV12, but we need BGR for OpenCV display.
//
//   3. RGB → BGR:  For feeding OpenCV images to neural networks that
//      expect RGB, or vice versa.
//
// The conversion also handles the YUV value range (limited vs full)
// and the conversion matrix (BT.601 for SD, BT.709 for HD).
//
// Usage:
//   auto cvt = CreateOperation<RgaCvtColor>(
//       RgaPixelFormat::kBgr888,
//       RgaPixelFormat::kNv12,
//       RgaCscMode::kRgbToYuvBt601Limit);
//   cvt->Execute(src_desc, dst_desc);
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaCvtColor — convert between pixel formats using RGA hardware
//
// Configuration:
//   - src_format: source pixel format
//   - dst_format: destination pixel format
//   - csc_mode: color space conversion mode (default: auto-detect)
//
// Execute(src, dst):
//   - src: source image in src_format
//   - dst: destination image in dst_format
//   - Returns true on success
//
// The format in ImageDescriptor must match the format specified here.
// If they conflict, the behavior is undefined (RGA uses the descriptor format).
// ---------------------------------------------------------------------------
class RgaCvtColor : public RgaOperation {
public:
    // Construct a color converter.
    // src_format: pixel format of the source image
    // dst_format: pixel format of the destination image
    // csc_mode: color space conversion algorithm
    RgaCvtColor(RgaPixelFormat src_format,
                RgaPixelFormat dst_format,
                RgaCscMode csc_mode = RgaCscMode::kDefault);

    // Convert src from src_format to dst_format.
    // src and dst must have valid data pointers and correct dimensions.
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& src, ImageDescriptor& dst);

private:
    RgaPixelFormat src_format_;
    RgaPixelFormat dst_format_;
    RgaCscMode csc_mode_;
};

}  // namespace rgakit
}  // namespace helmsman
