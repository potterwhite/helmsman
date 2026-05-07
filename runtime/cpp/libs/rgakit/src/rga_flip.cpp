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
// rga_flip.cpp — RGA hardware flip implementation
//
// Calls the RGA imflip() function to flip an image horizontally or vertically.
//
// Flip operations:
//   Horizontal (kHorizontal): mirror left-right, e.g. ABCD → DCBA
//   Vertical (kVertical):     mirror top-bottom, e.g. row0↔rowN
//
// Flip is a zero-cost operation — RGA just changes the pixel read order.
// No data copy or transformation is needed, so it runs at memory bandwidth.
//
// =============================================================================

#include "RGAKit/rga_flip.h"

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
// RgaFlip constructor
// ---------------------------------------------------------------------------
RgaFlip::RgaFlip(RgaFlipMode flip_mode)
    : flip_mode_(flip_mode) {}

// ---------------------------------------------------------------------------
// RgaFlip::Execute — flip image via RGA hardware
//
// src and dst must have the same dimensions and format.
// ---------------------------------------------------------------------------
bool RgaFlip::Execute(const ImageDescriptor& src, ImageDescriptor& dst) {
    if (!src.data || !dst.data) {
        fprintf(stderr, "[RGAKit] RgaFlip::Execute: null data pointer\n");
        return false;
    }

    // Validate dimensions: src and dst must match.
    if (src.width != dst.width || src.height != dst.height) {
        fprintf(stderr,
                "[RGAKit] RgaFlip::Execute: dimension mismatch "
                "(src=%dx%d, dst=%dx%d)\n",
                src.width, src.height,
                dst.width, dst.height);
        return false;
    }

    rga_buffer_t rga_src = ToRgaBuffer(src);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    IM_STATUS ret = imflip(
        rga_src,
        rga_dst,
        static_cast<int>(flip_mode_),
        1,  // sync: wait for completion
        nullptr);  // release_fence_fd: unused

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr,
                "[RGAKit] RgaFlip::Execute: imflip failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace arcforge
