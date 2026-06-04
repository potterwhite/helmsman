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
// RGAKit/rga_crop.h — Hardware-accelerated image crop via RGA
//
// Wraps the RGA imcrop() function. Extracts a rectangular region from the
// source image and writes it to the destination using RGA2 hardware.
//
// The crop rectangle is specified in source image coordinates. The destination
// image must be allocated with dimensions matching rect.width × rect.height.
//
// Performance: RGA crop is near-zero CPU cost — it's a DMA remapping operation.
//
// Usage:
//   RgaCrop crop;
//   ImageDescriptor src(buf, 1920, 1080, kBgr888);
//   ImageDescriptor dst(buf_out, 960, 544, kBgr888);
//   im_rect rect = {100, 50, 960, 544};  // x, y, width, height
//   crop.Execute(src, dst, rect);
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// CropRect — rectangle descriptor for crop operations
//
// Mirrors the RGA SDK's im_rect struct. Defined here to avoid pulling the
// RGA SDK header into our public API (same pattern as RgaPixelFormat).
// ---------------------------------------------------------------------------
struct CropRect {
    int x = 0;       // upper-left x coordinate
    int y = 0;       // upper-left y coordinate
    int width = 0;   // crop width
    int height = 0;  // crop height
};

// ---------------------------------------------------------------------------
// RgaCrop — crop a rectangular region from an image using RGA hardware
//
// Execute(src, dst, rect):
//   - src: source image descriptor
//   - dst: destination image descriptor (must be rect.width × rect.height)
//   - rect: crop rectangle in source coordinates
//   - Returns true on success
// ---------------------------------------------------------------------------
class RgaCrop : public RgaOperation {
public:
    RgaCrop() = default;

    // Crop a region from src defined by rect, write to dst.
    // dst dimensions must match rect.width × rect.height.
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& src, ImageDescriptor& dst,
                 const CropRect& rect);
};

}  // namespace rgakit
}  // namespace helmsman
