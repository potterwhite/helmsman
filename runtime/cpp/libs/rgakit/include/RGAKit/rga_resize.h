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
// RGAKit/rga_resize.h — Hardware-accelerated image resize via RGA
//
// Wraps the RGA imresize() function. Resizes an image from src dimensions
// to dst dimensions using the RGA2 hardware scaler.
//
// The RGA hardware scaler supports:
//   - Bilinear interpolation (default, good quality)
//   - Nearest neighbor (fast, pixelated)
//   - Bicubic (better quality, slower)
//
// Performance: RGA resize is ~10× faster than cv::resize on RK3588 for
// large images (e.g. 1920×1080 → 288×512), because it uses dedicated
// hardware instead of the ARM CPU.
//
// Usage:
//   auto resize = CreateOperation<RgaResize>(288, 512);
//   resize->Execute(src_desc, dst_desc);
//
// Note: src and dst dimensions are taken from the ImageDescriptors.
// The target_width/target_height stored in this class are for reference only.
// The actual resize dimensions come from the src and dst descriptors.
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaResize — resize/scale an image using RGA hardware
//
// Configuration:
//   - interpolation: resize algorithm (default: bilinear)
//
// Execute(src, dst):
//   - src: source image descriptor (any size)
//   - dst: destination image descriptor (target size)
//   - Returns true on success
//
// The RGA hardware will read from src.data at src.width × src.height,
// and write to dst.data at dst.width × dst.height.
// ---------------------------------------------------------------------------
class RgaResize : public RgaOperation {
public:
    // Interpolation method for resize.
    // RGA SDK values: INTER_NEAREST=0, INTER_LINEAR=1, INTER_CUBIC=2
    enum class Interpolation : int {
        kNearest = 0,   // Nearest neighbor — fastest, pixelated
        kBilinear = 1,  // Bilinear — good balance (default)
        kBicubic = 2,   // Bicubic — better quality, slightly slower
    };

    explicit RgaResize(Interpolation interp = Interpolation::kBilinear);

    // Resize src to dst dimensions.
    // src and dst must have valid data pointers and matching formats.
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& src, ImageDescriptor& dst);

private:
    Interpolation interpolation_;
};

}  // namespace rgakit
}  // namespace helmsman
