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
// rga_resize.cpp — RGA hardware resize implementation
//
// Calls the RGA imresize() function to scale an image using the hardware
// scaler built into the RK3588's RGA2 block.
//
// How imresize() works internally:
//   1. RGA driver reads pixels from src via DMA/virtual address
//   2. Hardware scaler applies the chosen interpolation filter
//   3. Scaled pixels are written to dst via DMA/virtual address
//   4. CPU is idle during the operation — it's all hardware
//
// The RGA hardware scaler supports arbitrary scale factors (e.g. 1920→288,
// 1080→512) with no restrictions on aspect ratio. The caller is responsible
// for choosing a sensible target size.
//
// =============================================================================

#include "RGAKit/rga_resize.h"

#include "RGAKit/rga_operation.h"

// RGA SDK headers — the SDK handles its own C/C++ linkage via IM_API/IM_C_API
// macros, so we do NOT wrap in extern "C" (that would conflict with the SDK's
// C++ overloads which have default arguments).
#include "im2d.h"

namespace arcforge {
namespace rgakit {

// ---------------------------------------------------------------------------
// Helper: build an rga_buffer_t from our ImageDescriptor
//
// rga_buffer_t is the SDK's image descriptor. We convert our C++ struct to
// the C struct that the im2d API expects.
//
// In C++ mode, wrapbuffer_virtualaddr() is a regular function (not a macro),
// declared in im2d_buffer.h. It fills an rga_buffer_t with:
//   - vir_addr:  pointer to pixel data
//   - width/height: image dimensions
//   - wstride/hstride: stride (we use width/height = tightly packed)
//   - format: pixel format constant
// ---------------------------------------------------------------------------
static rga_buffer_t ToRgaBuffer(const ImageDescriptor& desc) {
    const int w = desc.width;
    const int h = desc.height;
    const int fmt = static_cast<int>(desc.format);
    const int ws = desc.wstride > 0 ? desc.wstride : desc.width;
    const int hs = desc.hstride > 0 ? desc.hstride : desc.height;
    if (desc.fd >= 0)
        return wrapbuffer_fd(desc.fd, w, h, fmt, ws, hs);
    return wrapbuffer_virtualaddr(desc.data, w, h, fmt, ws, hs);
}

// ---------------------------------------------------------------------------
// RgaResize constructor
// ---------------------------------------------------------------------------
RgaResize::RgaResize(Interpolation interp)
    : interpolation_(interp) {}

// ---------------------------------------------------------------------------
// RgaResize::Execute — resize src to dst dimensions via RGA hardware
//
// Returns true on success. On failure, prints an error to stderr with the
// RGA error code.
//
// Note: imresize() is synchronous by default (sync=1) — it blocks until
// the hardware operation completes. This is the safest mode. For async,
// we would need to manage fence file descriptors.
// ---------------------------------------------------------------------------
bool RgaResize::Execute(const ImageDescriptor& src, ImageDescriptor& dst) {
    if (!src.data || !dst.data) {
        fprintf(stderr, "[RGAKit] RgaResize::Execute: null data pointer\n");
        return false;
    }

    rga_buffer_t rga_src = ToRgaBuffer(src);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    IM_STATUS ret = imresize(
        rga_src,
        rga_dst,
        0,  // fx: unused (dst dimensions determine the scale)
        0,  // fy: unused
        static_cast<int>(interpolation_),
        1,  // sync: wait for completion
        nullptr);  // release_fence_fd: unused in sync mode

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[RGAKit] RgaResize::Execute: imresize failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace arcforge
