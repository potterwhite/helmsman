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
// PURPOSE:
//   Calls the RGA imcomposite() function to blend three images:
//     dst = blend(fg, bg) using Porter-Duff compositing
//
// HOW imcomposite() WORKS (per Rockchip documentation):
//   1. Reads foreground (srcA) pixels — must have alpha channel (BGRA/RGBA)
//   2. Reads background (srcB) pixels — BGR/BGRA/RGB/RGBA
//   3. For each pixel: dst = fg.alpha * fg.rgb + (1 - fg.alpha) * bg.rgb
//   4. Writes composited pixels to dst (BGR/RGB, no alpha needed)
//
// EXPECTED PERFORMANCE GAIN:
//   Before (CPU float32 blend, ~5-10ms for 1080p):
//     alpha_f32 = alpha_8u / 255.0
//     composed = alpha_3ch * fg_f32 + (1 - alpha_3ch) * bg_f32
//     composed_8u = composed * 255.0
//
//   After (RGA hardware, ~1ms for 1080p):
//     imcomposite(fg_bgra, bg_bgra, dst_bgr)
//
// KNOWN ISSUES (discovered 2026-05-09, s39j debug):
//   The RGA imcomposite() returns SUCCESS (ret=1) but produces WRONG output.
//   When fg.alpha=0x00, the expected output is dst=bg, but actual dst ≠ bg.
//   This means RGA hardware does NOT perform standard Porter-Duff compositing
//   as documented. The exact blending formula used by the hardware is unclear.
//
//   Root cause of the s39i/s39j bug: the OLD code used wrong wrapbuffer_virtualaddr
//   parameter order, which caused imcomposite to return NOT_SUPPORTED (-1).
//   This triggered the CPU fallback path in rvm.cpp, which produced correct output.
//   The NEW code (ca2dc5a) fixed the parameter order, so imcomposite returns
//   SUCCESS — but the hardware output is wrong, and no fallback is triggered.
//
// CURRENT STATUS:
//   RGA composite is UNRELIABLE for our use case. The recommended fix is to
//   use RGA only for resize operations and fall back to CPU for compositing.
//   The CPU uint8 blend (integer multiply + shift, no float) takes ~5-10ms
//   for 1080p, which is acceptable.
//
// =============================================================================

#include "RGAKit/rga_composite.h"

#include "RGAKit/rga_operation.h"

#include "im2d.h"

namespace helmsman {
namespace rgakit {

// ToRgaBuffer — convert our ImageDescriptor to an RGA rga_buffer_t
//
// This is the bridge between our C++ abstraction and the RGA C SDK.
//
// rga_buffer_t is the RGA SDK's image descriptor. It tells the hardware:
//   - WHERE the pixels are: vir_addr (CPU pointer), phy_addr, or DMA fd
//   - HOW BIG the image is: width × height
//   - HOW pixels are laid out: wstride (bytes/row), hstride (rows), format
//
// Two modes:
//   1. DMA fd mode (desc.fd >= 0): uses wrapbuffer_fd() — zero-copy, fastest
//   2. Virtual address mode: uses wrapbuffer_virtualaddr() — CPU malloc'd memory
//
// CRITICAL: wrapbuffer_virtualaddr is a C MACRO, not a function!
//   Parameter order: (vir_addr, width, height, format, [wstride, hstride])
//   - format is the 4th parameter, NOT the 6th!
//   - In C++, you MUST pass all 6 args (zero-length array is illegal in C++)
//   - Wrong order compiles fine but produces garbage output or crashes
//
// The wstride/hstride default to width/height when not specified (packed layout).
// Hardware buffers may have larger strides (aligned to 16/64 bytes).
static rga_buffer_t ToRgaBuffer(const ImageDescriptor& desc) {
    const int w = desc.width;
    const int h = desc.height;
    const int fmt = static_cast<int>(desc.format);
    // wstride/hstride: use explicit value if set, otherwise default to width/height
    const int ws = desc.wstride > 0 ? desc.wstride : desc.width;
    const int hs = desc.hstride > 0 ? desc.hstride : desc.height;
    // DMA fd mode: zero-copy, RGA reads directly from DMA buffer
    if (desc.fd >= 0)
        return wrapbuffer_fd(desc.fd, w, h, fmt, ws, hs);
    // Virtual address mode: CPU memory, RGA driver maps it for DMA access
    return wrapbuffer_virtualaddr(desc.data, w, h, fmt, ws, hs);
}

// ---------------------------------------------------------------------------
// RgaComposite constructor
// ---------------------------------------------------------------------------
RgaComposite::RgaComposite(RgaBlendMode mode)
    : blend_mode_(mode) {}

// ---------------------------------------------------------------------------
// RgaComposite::Execute — alpha composite fg over bg via RGA hardware
//
// PARAMETERS:
//   fg  — foreground image, MUST have alpha channel (BGRA8888 or RGBA8888)
//   bg  — background image (BGR, BGRA, RGB, or RGBA)
//   dst — output image (BGR or RGB, composited result)
//
// ALPHA BEHAVIOR (per documentation):
//   The alpha values in fg control the blending weight per pixel:
//     alpha=255 → foreground fully visible (dst = fg)
//     alpha=0   → background fully visible (dst = bg)
//     alpha=128 → 50/50 blend
//   Formula: dst.rgb = (fg.alpha/255) * fg.rgb + (1 - fg.alpha/255) * bg.rgb
//
// BLEND MODE:
//   Uses RgaBlendMode::kSrcOver = IM_ALPHA_BLEND_SRC_OVER = 1 << 6 = 64
//   This is the standard "paint foreground over background" Porter-Duff mode.
//   The RGA SDK also defines IM_ALPHA_BLEND_PRE_MUL = 1 << 25 for
//   premultiplied alpha, but we don't use it (our alpha is straight/unassociated).
//
// KNOWN ISSUES:
//   As of 2026-05-09, RGA imcomposite returns SUCCESS but produces wrong output.
//   The hardware blending formula does not match the documented Porter-Duff behavior.
//   See file header for detailed analysis.
//
// RETURN VALUE:
//   true  — RGA hardware composite succeeded (but output may be wrong!)
//   false — RGA operation failed (null ptr, dimension mismatch, or hardware error)
//           Caller should fall back to CPU blend when false is returned.
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

    // DEBUG: dump input pixels
    {
        static int dump_count = 0;
        if (dump_count < 3) {
            const uint8_t* fg_ptr = static_cast<const uint8_t*>(fg.data);
            const uint8_t* bg_ptr = static_cast<const uint8_t*>(bg.data);
            fprintf(stderr, "[RGAKit-DEBUG] frame=%d fg(%dx%d fmt=%d): "
                    "[%02x %02x %02x %02x] [%02x %02x %02x %02x] [%02x %02x %02x %02x]\n",
                    dump_count, fg.width, fg.height, static_cast<int>(fg.format),
                    fg_ptr[0], fg_ptr[1], fg_ptr[2], fg_ptr[3],
                    fg_ptr[4], fg_ptr[5], fg_ptr[6], fg_ptr[7],
                    fg_ptr[8], fg_ptr[9], fg_ptr[10], fg_ptr[11]);
            fprintf(stderr, "[RGAKit-DEBUG] frame=%d bg(%dx%d fmt=%d): "
                    "[%02x %02x %02x %02x] [%02x %02x %02x %02x] [%02x %02x %02x %02x]\n",
                    dump_count, bg.width, bg.height, static_cast<int>(bg.format),
                    bg_ptr[0], bg_ptr[1], bg_ptr[2], bg_ptr[3],
                    bg_ptr[4], bg_ptr[5], bg_ptr[6], bg_ptr[7],
                    bg_ptr[8], bg_ptr[9], bg_ptr[10], bg_ptr[11]);
            fprintf(stderr, "[RGAKit-DEBUG] rga_fg: vir=%p w=%d h=%d ws=%d hs=%d fmt=%d\n",
                    rga_fg.vir_addr, rga_fg.width, rga_fg.height,
                    rga_fg.wstride, rga_fg.hstride, rga_fg.format);
            fprintf(stderr, "[RGAKit-DEBUG] rga_bg: vir=%p w=%d h=%d ws=%d hs=%d fmt=%d\n",
                    rga_bg.vir_addr, rga_bg.width, rga_bg.height,
                    rga_bg.wstride, rga_bg.hstride, rga_bg.format);
            fprintf(stderr, "[RGAKit-DEBUG] rga_dst: vir=%p w=%d h=%d ws=%d hs=%d fmt=%d\n",
                    rga_dst.vir_addr, rga_dst.width, rga_dst.height,
                    rga_dst.wstride, rga_dst.hstride, rga_dst.format);
            fprintf(stderr, "[RGAKit-DEBUG] blend_mode=%d\n", static_cast<int>(blend_mode_));
        }
        dump_count++;
    }

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

    // DEBUG: dump output pixels
    {
        static int dump_out = 0;
        if (dump_out < 3) {
            const uint8_t* dst_ptr = static_cast<const uint8_t*>(dst.data);
            fprintf(stderr, "[RGAKit-DEBUG] frame=%d dst(%dx%d fmt=%d) ret=%d: "
                    "[%02x %02x %02x] [%02x %02x %02x] [%02x %02x %02x]\n",
                    dump_out, dst.width, dst.height, static_cast<int>(dst.format),
                    static_cast<int>(ret),
                    dst_ptr[0], dst_ptr[1], dst_ptr[2],
                    dst_ptr[3], dst_ptr[4], dst_ptr[5],
                    dst_ptr[6], dst_ptr[7], dst_ptr[8]);
        }
        dump_out++;
    }

    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr,
                "[RGAKit] RgaComposite::Execute: imcomposite failed, ret=%d\n",
                static_cast<int>(ret));
        return false;
    }

    return true;
}

}  // namespace rgakit
}  // namespace helmsman
