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
// RGAKit/rga_flip.h — Hardware-accelerated image flip via RGA
//
// Wraps the RGA imflip() function. Flips an image horizontally (mirror
// left-right) or vertically (mirror top-bottom) using the RGA2 hardware.
//
// Flip is a zero-cost operation on RGA — it just changes the pixel read
// order, so it runs at memory bandwidth speed with no CPU overhead.
//
// Usage:
//   auto flip = CreateOperation<RgaFlip>(RgaFlipMode::kHorizontal);
//   flip->Execute(src_desc, dst_desc);
//
// src and dst must have the same dimensions and format.
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaFlip — flip an image horizontally or vertically using RGA hardware
//
// Configuration:
//   - flip_mode: horizontal (left-right) or vertical (top-bottom)
//
// Execute(src, dst):
//   - src: source image
//   - dst: destination image (same dimensions as src)
//   - Returns true on success
// ---------------------------------------------------------------------------
class RgaFlip : public RgaOperation {
public:
    explicit RgaFlip(RgaFlipMode flip_mode);

    // Flip src according to the configured mode, write to dst.
    // src and dst must have the same width, height, and format.
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& src, ImageDescriptor& dst);

private:
    RgaFlipMode flip_mode_;
};

}  // namespace rgakit
}  // namespace helmsman
