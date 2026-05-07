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
// RGAKit/rga_operation.h — Abstract base class and image descriptor for RGA
//
// RGA (Rockchip Graphics Accelerator) is a hardware 2D graphics engine built
// into RK3588. It can perform pixel-level operations at near-zero CPU cost:
//
//   - Resize / scale        (imresize)
//   - Color conversion      (imcvtcolor)   e.g. BGR↔NV12, RGB↔YUV
//   - Alpha compositing     (imcomposite)  Porter-Duff "SRC over DST"
//   - Rotation              (imrotate)     90°/180°/270°
//   - Flip                  (imflip)       horizontal/vertical
//   - Crop                  (imcrop)
//   - Copy                  (imcopy)
//
// Unlike MPP (which manages a VPU context with lifecycle), RGA operations are
// stateless: each call is a one-shot hardware blit. There is no Init()/Close().
//
// Architecture:
//
//   RgaOperation  (abstract base — this file)
//     ├── RgaResize       (resize/scale)
//     ├── RgaCvtColor     (color space / pixel format conversion)
//     ├── RgaComposite    (3-image alpha blending)
//     ├── RgaRotate       (90°/180°/270° rotation)
//     └── RgaFlip         (horizontal/vertical flip)
//
// ImageDescriptor (this file) wraps a raw pixel buffer with its dimensions
// and format, so RGA knows how to interpret the memory.
//
// How to use:
//
//   1. Create an operation via factory:
//        auto resize = rgakit::CreateOperation<RgaResize>(288, 512);
//   2. Prepare image descriptors:
//        ImageDescriptor src = ImageDescriptor::FromBgr(mat);
//        ImageDescriptor dst(buf, 288, 512, RK_FORMAT_BGR_888);
//   3. Execute:
//        resize->Execute(src, dst);
//
// How to extend:
//
//   To add a new operation (e.g. mosaic):
//     1. Create RgaMosaic subclass with Execute()
//     2. Implement Execute() using the im2d C API
//     3. Use CreateOperation<RgaMosaic>(...) to instantiate
//
// RGA SDK reference:
//   im2d API: .../external/linux-rga/im2d_api/
//   Types:    rga_buffer_t, im_rect, IM_STATUS
//   Wrap:     wrapbuffer_virtualaddr(), wrapbuffer_fd()
//
// =============================================================================

#pragma once

#include "RGAKit/pch.h"

namespace arcforge {
namespace rgakit {

// ---------------------------------------------------------------------------
// RgaPixelFormat — pixel format constants matching the RGA hardware
//
// These values are identical to RK_FORMAT_* in rga.h. We define our own enum
// to avoid pulling the C SDK header into our public API.
//
// Common formats for video pipelines:
//   kBgr888   — OpenCV default (cv::Mat CV_8UC3)
//   kRgb888   — many neural network inputs
//   kRgba8888 — OpenGL, Android
//   kNv12     — VPU native format (MPP encoder input)
//   kNv16     — 4:2:2 semi-planar
//   kYuv420P  — 3-plane YUV 4:2:0 (FFmpeg default)
// ---------------------------------------------------------------------------
enum class RgaPixelFormat : int {
    kRgba8888 = 0x0 << 8,    // R:G:B:A 8:8:8:8
    kRgbx8888 = 0x1 << 8,    // R:G:B:X 8:8:8:8 (no alpha)
    kRgb888   = 0x2 << 8,    // R:G:B 8:8:8
    kBgra8888 = 0x3 << 8,    // B:G:R:A 8:8:8:8
    kRgb565   = 0x4 << 8,    // R:G:B 5:6:5
    kBgr888   = 0x7 << 8,    // B:G:R 8:8:8 (OpenCV default)
    kNv12     = 0xa << 8,    // YCbCr 4:2:0 semi-planar (NV12, VPU native)
    kNv16     = 0x8 << 8,    // YCbCr 4:2:2 semi-planar
    kYuv420P  = 0xb << 8,    // YCbCr 4:2:0 3-plane (planar)
    kYuv420Sp = 0xa << 8,    // Same as kNv12 (alias)
    kYuv400   = 0x15 << 8,   // Y-only 8-bit (grayscale, single channel)
};

// ---------------------------------------------------------------------------
// RgaCscMode — color space conversion mode for imcvtcolor()
//
// Selects the conversion matrix and value range.
// Naming: IM_<SRC>_TO_<DST>_BT<STANDARD>_<RANGE>
//
// BT.601 — standard definition (SD video, 480p/576p)
// BT.709 — high definition (HD video, 720p/1080p/4K)
// LIMIT  — limited range: Y [16..235], UV [16..240]
// FULL   — full range: Y [0..255], UV [0..255]
//
// For our RVM pipeline (1080p video):
//   BGR → NV12 for MPP encoder: use kRgbToYuvBt601Limit
//   NV12 → BGR from MPP decoder: use kYuvToRgbBt601Limit
// ---------------------------------------------------------------------------
enum class RgaCscMode : int {
    kDefault         = 0,
    kYuvToRgbBt601Limit = 1 << 0,   // YUV → RGB, BT.601, limited range
    kYuvToRgbBt601Full  = 2 << 0,   // YUV → RGB, BT.601, full range
    kYuvToRgbBt709Limit = 3 << 0,   // YUV → RGB, BT.709, limited range
    kRgbToYuvBt601Full  = 1 << 2,   // RGB → YUV, BT.601, full range
    kRgbToYuvBt601Limit = 2 << 2,   // RGB → YUV, BT.601, limited range
    kRgbToYuvBt709Limit = 3 << 2,   // RGB → YUV, BT.709, limited range
};

// ---------------------------------------------------------------------------
// RgaBlendMode — Porter-Duff alpha blending modes for imblend()/imcomposite()
//
// Porter-Duff compositing defines how two images combine based on their alpha
// channels. "SRC" = foreground, "DST" = background.
//
// For RVM video compositing (alpha * FG + (1-alpha) * BG):
//   Use kSrcOver — the standard "paint foreground over background" mode.
//   This is also the default if you don't specify a mode.
//
// Other modes exist for special effects (e.g. kSrc for direct copy,
// kDstOver for painting behind, etc.).
// ---------------------------------------------------------------------------
enum class RgaBlendMode : int {
    kSrcOver = 1 << 6,   // Default: result = SRC*alpha + DST*(1-alpha)
    kSrc     = 1 << 7,   // Result = SRC only (ignore DST)
    kDst     = 1 << 8,   // Result = DST only (ignore SRC)
    kSrcIn   = 1 << 9,   // SRC inside DST alpha
    kDstIn   = 1 << 10,  // DST inside SRC alpha
    kSrcOut  = 1 << 11,  // SRC outside DST alpha
    kDstOut  = 1 << 12,  // DST outside SRC alpha
    kDstOver = 1 << 13,  // Result = DST over SRC
    kSrcAtop = 1 << 14,  // SRC atop DST
    kDstAtop = 1 << 15,  // DST atop SRC
    kXor     = 1 << 16,  // XOR
};

// ---------------------------------------------------------------------------
// RgaRotation — rotation angles for imrotate()
// ---------------------------------------------------------------------------
enum class RgaRotation : int {
    k90  = 1 << 0,   // 90° clockwise
    k180 = 1 << 1,   // 180°
    k270 = 1 << 2,   // 270° clockwise (= 90° counter-clockwise)
};

// ---------------------------------------------------------------------------
// RgaFlipMode — flip modes for imflip()
// ---------------------------------------------------------------------------
enum class RgaFlipMode : int {
    kHorizontal = 1 << 3,   // Mirror left-right
    kVertical   = 1 << 4,   // Mirror top-bottom
};

// ---------------------------------------------------------------------------
// ImageDescriptor — describes a pixel buffer for RGA operations
//
// RGA needs to know:
//   - Where the pixels are in memory (data pointer or DMA fd)
//   - How big the image is (width × height)
//   - How the pixels are laid out (stride, format)
//
// For most use cases, use the static From*() helpers:
//   - FromBgr(cv::Mat)     — OpenCV BGR image
//   - FromRaw(ptr, w, h)   — raw buffer with default stride
//
// For advanced use (DMA buffers, hardware interop), set fd directly.
//
// Note on strides:
//   wstride (width stride) = number of pixels per row, including any padding.
//   hstride (height stride) = number of rows, including any padding.
//   For a tightly-packed buffer: wstride == width, hstride == height.
//   For hardware buffers, strides may be larger (aligned to 16 or 64 bytes).
// ---------------------------------------------------------------------------
struct ImageDescriptor {
    void* data = nullptr;        // Virtual address of pixel data
    int fd = -1;                 // DMA file descriptor (-1 = not using DMA)
    int width = 0;               // Image width in pixels
    int height = 0;              // Image height in pixels
    int wstride = 0;             // Width stride in pixels (>= width)
    int hstride = 0;             // Height stride in pixels (>= height)
    RgaPixelFormat format = RgaPixelFormat::kBgr888;  // Pixel format

    // Default constructor — empty descriptor (must be filled manually).
    ImageDescriptor() = default;

    // Full constructor — specify everything.
    ImageDescriptor(void* d, int w, int h, RgaPixelFormat fmt)
        : data(d), width(w), height(h), wstride(w), hstride(h), format(fmt) {}

    // Full constructor with custom strides.
    ImageDescriptor(void* d, int w, int h, int ws, int hs, RgaPixelFormat fmt)
        : data(d), width(w), height(h), wstride(ws), hstride(hs), format(fmt) {}

    // Convenience: create from raw pointer with tightly-packed layout.
    static ImageDescriptor FromRaw(void* ptr, int w, int h,
                                   RgaPixelFormat fmt) {
        return ImageDescriptor(ptr, w, h, fmt);
    }
};

// ---------------------------------------------------------------------------
// RgaOperation — abstract base class for all RGA operations
//
// Unlike MppCodec (which manages a VPU context), RgaOperation is stateless:
// each Execute() call is a one-shot hardware blit. There is no Init()/Close().
//
// Subclasses hold configuration (target size, format, mode) and implement
// Execute() with the specific parameters they need.
//
// Non-copyable (could hold DMA resources in the future).
// Movable.
// ---------------------------------------------------------------------------
class RgaOperation {
public:
    virtual ~RgaOperation() = default;

    // No copies — may hold hardware resources.
    RgaOperation(const RgaOperation&) = delete;
    RgaOperation& operator=(const RgaOperation&) = delete;

    // Move is allowed.
    RgaOperation(RgaOperation&&) noexcept = default;
    RgaOperation& operator=(RgaOperation&&) noexcept = default;

protected:
    // Only subclasses can construct the base class.
    RgaOperation() = default;
};

// ---------------------------------------------------------------------------
// Factory function template — creates concrete RgaOperation subclasses
//
// Usage:
//   auto op = CreateOperation<RgaResize>(288, 512);
//   auto op = CreateOperation<RgaCvtColor>(RgaPixelFormat::kBgr888,
//                                          RgaPixelFormat::kNv12);
//   auto op = CreateOperation<RgaComposite>(RgaBlendMode::kSrcOver);
//
// Returns nullptr if arguments are invalid.
// ---------------------------------------------------------------------------
template <typename T, typename... Args>
std::unique_ptr<T> CreateOperation(Args&&... args) {
    auto op = std::make_unique<T>(std::forward<Args>(args)...);
    return op;
}

}  // namespace rgakit
}  // namespace arcforge
