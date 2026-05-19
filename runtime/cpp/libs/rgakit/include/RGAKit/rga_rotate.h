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
// RGAKit/rga_rotate.h — Hardware-accelerated image rotation via RGA
//
// Wraps the RGA imrotate() function. Rotates an image by 90°, 180°, or 270°
// using the RGA2 hardware rotation unit.
//
// RGA rotation is lossless (no interpolation artifacts) because 90°/180°/270°
// are simple pixel rearrangements, not arbitrary-angle rotations.
//
// Note: For arbitrary-angle rotation (e.g. 45°), you need a different approach
// (affine transform). RGA does not support arbitrary angles natively.
//
// Usage:
//   auto rot = CreateOperation<RgaRotate>(RgaRotation::k90);
//   rot->Execute(src_desc, dst_desc);
//
// After 90°/270° rotation:
//   dst.width  = src.height
//   dst.height = src.width
// (dimensions are swapped)
//
// After 180° rotation:
//   dst.width  = src.width
//   dst.height = src.height
// (dimensions unchanged)
//
// =============================================================================

#pragma once

#include "RGAKit/rga_operation.h"

namespace helmsman {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaRotate — rotate an image by 90°/180°/270° using RGA hardware
//
// Configuration:
//   - rotation: angle to rotate
//
// Execute(src, dst):
//   - src: source image
//   - dst: destination image (must have correct dimensions for the rotation)
//   - Returns true on success
// ---------------------------------------------------------------------------
class RgaRotate : public RgaOperation {
public:
    explicit RgaRotate(RgaRotation rotation);

    // Rotate src by the configured angle, write to dst.
    // For 90°/270°: dst dimensions must be (src.height × src.width).
    // For 180°: dst dimensions must be (src.width × src.height).
    // Returns true on success, false on RGA error.
    bool Execute(const ImageDescriptor& src, ImageDescriptor& dst);

private:
    RgaRotation rotation_;
};

}  // namespace rgakit
}  // namespace helmsman
