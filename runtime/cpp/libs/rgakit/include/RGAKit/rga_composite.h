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
// RGAKit/rga_composite.h — Hardware-accelerated 3-image alpha compositing
//
// Wraps the RGA imcomposite() function. Performs Porter-Duff alpha blending
// of three images: foreground (srcA), background (srcB), and output (dst).
//
// This is the key operation for video compositing in the RVM pipeline:
//
//   composed = alpha * foreground + (1 - alpha) * background
//
// Where:
//   - foreground (srcA): the original video frame (BGR, 1920×1080)
//   - background (srcB): the replacement background (BGR, 1920×1080)
//   - alpha channel: embedded in the foreground image (BGRA, 1920×1080)
//   - dst: the composited output (BGR, 1920×1080)
//
// Performance: RGA composite is ~14× faster than manual float32 alpha blend
// on the CPU, because:
//   1. Hardware operates on uint8 directly (no float conversion)
//   2. No memory allocation for intermediate float32 buffers
//   3. Dedicated pixel pipeline with hardware parallelism
//
// Usage:
//   // Prepare foreground with alpha channel (BGRA)
//   ImageDescriptor fg(fg_data, 1920, 1080, RgaPixelFormat::kBgra8888);
//   ImageDescriptor bg(bg_data, 1920, 1080, RgaPixelFormat::kBgr888);
//   ImageDescriptor out(out_data, 1920, 1080, RgaPixelFormat::kBgr888);
//
//   auto composite = CreateOperation<RgaComposite>(RgaBlendMode::kSrcOver);
//   composite->Execute(fg, bg, out);
//
// Note on alpha channel:
//   RGA composite expects the foreground to have an alpha channel (BGRA/RGBA).
//   The alpha values in the foreground image control the blending weight.
//   If your alpha is a separate single-channel image, you need to merge it
//   into the foreground as the A channel first (e.g. using OpenCV merge).
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaComposite — 3-image Porter-Duff alpha compositing via RGA hardware
//
// Configuration:
//   - blend_mode: Porter-Duff blending mode (default: SRC_OVER)
//
// Execute(fg, bg, dst):
//   - fg:  foreground image (must have alpha channel: BGRA/RGBA)
//   - bg:  background image (BGR/RGB, no alpha needed)
//   - dst: output image (same size and format as bg)
//   - Returns true on success
//
// All three images must have the same dimensions.
// The fg format must include an alpha channel (e.g. kBgra8888, kRgba8888).
// The bg and dst formats should match (e.g. kBgr888).
// ---------------------------------------------------------------------------
class RgaComposite : public RgaOperation {
public:
    explicit RgaComposite(RgaBlendMode mode = RgaBlendMode::kSrcOver);

    // Composite foreground over background, write to dst.
    // fg must have alpha channel (BGRA/RGBA format).
    // All images must have the same width × height.
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& fg,
                 const ImageDescriptor& bg,
                 ImageDescriptor& dst);

private:
    RgaBlendMode blend_mode_;
};

}  // namespace rgakit
}  // namespace helmsman
