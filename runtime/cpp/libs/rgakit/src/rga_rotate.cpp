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
// rga_rotate.cpp — RGA hardware rotation implementation
//
// Calls the RGA imrotate() function to rotate an image by 90°, 180°, or 270°.
//
// RGA rotation is a simple pixel rearrangement — no interpolation needed:
//   90° CW:  (x, y) → (height-1-y, x)
//   180°:    (x, y) → (width-1-x, height-1-y)
//   270° CW: (x, y) → (y, width-1-x)
//
// This means rotation is lossless and runs at memory bandwidth speed.
//
// =============================================================================

#include "RGAKit/rga_rotate.h"

#include "RGAKit/rga_operation.h"

#include "im2d.h"

namespace arcforge {
namespace rgakit {

static rga_buffer_t ToRgaBuffer(const ImageDescriptor& desc) {
    // wrapbuffer_virtualaddr is a MACRO: (vir_addr, width, height, format, wstride, hstride)
    // Must always pass wstride+hstride in C++ to avoid zero-size array error.
    const int w = desc.width;
    const int h = desc.height;
    const int fmt = static_cast<int>(desc.format);
    const int ws = desc.wstride > 0 ? desc.wstride : desc.width;
    const int hs = desc.hstride > 0 ? desc.hstride : desc.height;
    return wrapbuffer_virtualaddr(desc.data, w, h, fmt, ws, hs);
}

// ---------------------------------------------------------------------------
// RgaRotate constructor
// ---------------------------------------------------------------------------
RgaRotate::RgaRotate(RgaRotation rotation)
    : rotation_(rotation) {}

// ---------------------------------------------------------------------------
// RgaRotate::Execute — rotate image via RGA hardware
//
// For 90°/270° rotation, dst dimensions must be (src.height × src.width).
// For 180° rotation, dst dimensions must be (src.width × src.height).
// ---------------------------------------------------------------------------
bool RgaRotate::Execute(const ImageDescriptor& src, ImageDescriptor& dst) {
    if (!src.data || !dst.data) {
        fprintf(stderr, "[RGAKit] RgaRotate::Execute: null data pointer\n");
        return false;
    }

    // Validate dimensions based on rotation angle.
    bool dim_ok = false;
    if (rotation_ == RgaRotation::k180) {
        dim_ok = (dst.width == src.width && dst.height == src.height);
    } else {
        // 90° or 270°: dimensions are swapped.
        dim_ok = (dst.width == src.height && dst.height == src.width);
    }
    if (!dim_ok) {
        fprintf(stderr,
                "[RGAKit] RgaRotate::Execute: dimension mismatch "
                "(src=%dx%d, dst=%dx%d, rotation=%d)\n",
                src.width, src.height,
                dst.width, dst.height,
                static_cast<int>(rotation_));
        return false;
    }

    rga_buffer_t rga_src = ToRgaBuffer(src);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    IM_STATUS ret = imrotate(
        rga_src,
        rga_dst,
        static_cast<int>(rotation_),
        1,  // sync: wait for completion
        nullptr);  // release_fence_fd: unused

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr,
                "[RGAKit] RgaRotate::Execute: imrotate failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace arcforge
