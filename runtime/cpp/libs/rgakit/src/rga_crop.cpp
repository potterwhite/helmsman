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
// rga_crop.cpp — RGA hardware crop implementation
//
// Calls the RGA imcrop() function to extract a rectangular region from an
// image using the RGA2 hardware blitter.
//
// imcrop() is essentially a DMA remapping — it reads pixels from the source
// rectangle and writes them to the destination buffer. Near-zero CPU cost.
//
// =============================================================================

#include "RGAKit/rga_crop.h"

#include "RGAKit/rga_operation.h"
#include "im2d.h"

namespace helmsman {
namespace rgakit {

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

bool RgaCrop::Execute(const ImageDescriptor& src, ImageDescriptor& dst,
                      const CropRect& rect) {
    if ((!src.data && src.fd < 0) || (!dst.data && dst.fd < 0)) {
        fprintf(stderr, "[RGAKit] RgaCrop::Execute: null data pointer\n");
        return false;
    }

    rga_buffer_t rga_src = ToRgaBuffer(src);
    rga_buffer_t rga_dst = ToRgaBuffer(dst);

    im_rect r;
    r.x = rect.x;
    r.y = rect.y;
    r.width = rect.width;
    r.height = rect.height;

    IM_STATUS ret = imcrop(rga_src, rga_dst, r, 1, nullptr);

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[RGAKit] RgaCrop::Execute: imcrop failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace helmsman
