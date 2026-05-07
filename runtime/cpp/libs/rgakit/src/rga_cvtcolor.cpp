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
// rga_cvtcolor.cpp — RGA hardware color conversion implementation
//
// Calls the RGA imcvtcolor() function to convert between pixel formats
// and color spaces using the hardware color conversion unit.
//
// What imcvtcolor() does:
//   1. Reads pixels in the source format (e.g. BGR 8:8:8)
//   2. Applies a color space conversion matrix if needed
//      (e.g. BT.601 RGB→YUV for SD video, BT.709 for HD)
//   3. Writes pixels in the destination format (e.g. NV12)
//
// The conversion is done entirely in hardware — no CPU math.
//
// Key format combinations for our pipeline:
//   BGR888 → NV12 (kRgbToYuvBt601Limit):  for MPP encoder input
//   NV12 → BGR888 (kYuvToRgbBt601Limit):  for MPP decoder output
//   BGR888 → RGB888:                       for neural network input
//
// =============================================================================

#include "RGAKit/rga_cvtcolor.h"

#include "RGAKit/rga_operation.h"

#include "im2d.h"

namespace arcforge {
namespace rgakit {

static rga_buffer_t ToRgaBuffer(const ImageDescriptor& desc) {
    return wrapbuffer_virtualaddr(
        desc.data,
        desc.width,
        desc.height,
        desc.wstride > 0 ? desc.wstride : desc.width,
        desc.hstride > 0 ? desc.hstride : desc.height,
        static_cast<int>(desc.format));
}

// ---------------------------------------------------------------------------
// RgaCvtColor constructor
// ---------------------------------------------------------------------------
RgaCvtColor::RgaCvtColor(RgaPixelFormat src_format,
                         RgaPixelFormat dst_format,
                         RgaCscMode csc_mode)
    : src_format_(src_format),
      dst_format_(dst_format),
      csc_mode_(csc_mode) {}

// ---------------------------------------------------------------------------
// RgaCvtColor::Execute — convert pixel format via RGA hardware
//
// The src_format and dst_format stored in this class are for documentation
// purposes. The actual format used by RGA comes from the ImageDescriptor's
// format field (passed via wrapbuffer_virtualaddr).
//
// However, the csc_mode IS used by imcvtcolor() — it selects the conversion
// matrix (BT.601 vs BT.709, limited vs full range).
// ---------------------------------------------------------------------------
bool RgaCvtColor::Execute(const ImageDescriptor& src, ImageDescriptor& dst) {
    if (!src.data || !dst.data) {
        fprintf(stderr, "[RGAKit] RgaCvtColor::Execute: null data pointer\n");
        return false;
    }

    rga_buffer_t rga_src = ToRgaBuffer(src);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    IM_STATUS ret = imcvtcolor(
        rga_src,
        rga_dst,
        static_cast<int>(src_format_),
        static_cast<int>(dst_format_),
        static_cast<int>(csc_mode_),
        1,  // sync: wait for completion
        nullptr);  // release_fence_fd: unused

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr,
                "[RGAKit] RgaCvtColor::Execute: imcvtcolor failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace arcforge
