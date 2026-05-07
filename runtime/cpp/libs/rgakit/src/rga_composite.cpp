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
// rga_composite.cpp — RGA hardware 3-image alpha compositing implementation
//
// Calls the RGA imcomposite() function to blend three images:
//   dst = blend(fg, bg) using Porter-Duff compositing
//
// How imcomposite() works:
//   1. Reads foreground (srcA) pixels — must have alpha channel (BGRA/RGBA)
//   2. Reads background (srcB) pixels — BGR/RGB
//   3. For each pixel: dst = fg.alpha * fg.rgb + (1 - fg.alpha) * bg.rgb
//   4. Writes composited pixels to dst
//
// This replaces the CPU-intensive float32 alpha blend in our pipeline:
//
//   Before (CPU, ~122ms for 1080p):
//     alpha_f32 = alpha_8u / 255.0
//     composed = alpha_3ch * fg_f32 + (1 - alpha_3ch) * bg_f32
//     composed_8u = composed * 255.0
//
//   After (RGA hardware, ~1ms for 1080p):
//     imcomposite(fg_bgra, bg_bgr, dst_bgr)
//
// The RGA does the entire operation in uint8, in hardware, with no
// intermediate float32 buffers. This is the single biggest performance
// win for our video compositing pipeline.
//
// =============================================================================

#include "RGAKit/rga_composite.h"

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
// RgaComposite constructor
// ---------------------------------------------------------------------------
RgaComposite::RgaComposite(RgaBlendMode mode)
    : blend_mode_(mode) {}

// ---------------------------------------------------------------------------
// RgaComposite::Execute — alpha composite fg over bg via RGA hardware
//
// The fg image MUST have an alpha channel (BGRA8888 or RGBA8888).
// The alpha values in fg control the blending weight per pixel:
//   alpha=255 → foreground fully visible
//   alpha=0   → background fully visible
//   alpha=128 → 50/50 blend
//
// All three images must have the same width × height.
// ---------------------------------------------------------------------------
bool RgaComposite::Execute(const ImageDescriptor& fg,
                           const ImageDescriptor& bg,
                           ImageDescriptor& dst) {
    if (!fg.data || !bg.data || !dst.data) {
        fprintf(stderr,
                "[RGAKit] RgaComposite::Execute: null data pointer\n");
        return false;
    }

    // Validate: all images must have the same dimensions.
    if (fg.width != bg.width || fg.width != dst.width ||
        fg.height != bg.height || fg.height != dst.height) {
        fprintf(stderr,
                "[RGAKit] RgaComposite::Execute: dimension mismatch "
                "(fg=%dx%d, bg=%dx%d, dst=%dx%d)\n",
                fg.width, fg.height,
                bg.width, bg.height,
                dst.width, dst.height);
        return false;
    }

    rga_buffer_t rga_fg = ToRgaBuffer(fg);
    rga_buffer_t rga_bg = ToRgaBuffer(bg);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    // imcomposite(srcA, srcB, dst, mode, sync, release_fence_fd)
    //   srcA = foreground (with alpha channel)
    //   srcB = background
    //   dst  = output (composited result)
    IM_STATUS ret = imcomposite(
        rga_fg,
        rga_bg,
        rga_dst,
        static_cast<int>(blend_mode_),
        1,  // sync: wait for completion
        nullptr);  // release_fence_fd: unused

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr,
                "[RGAKit] RgaComposite::Execute: imcomposite failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace arcforge
