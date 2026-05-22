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
// mpp-frame-decoder.h — RK3588 VPU hardware frame decoder (internal)
//
// Wraps MppDecoder from mppkit to implement _IFrameDecoder.
// Decodes H.264/H.265 compressed packets into NV12 dmabuf frames.
//
// =============================================================================

#pragma once

#include <memory>
#include "pipeline/stages/frontend/decoder/i-frame-decoder.h"
#include "MPPKit/mpp_codec.h"

namespace helmsman {
namespace mppkit {
class MppDecoder;
}  // namespace mppkit
}  // namespace helmsman

class _MppFrameDecoder : public _IFrameDecoder {
public:
    explicit _MppFrameDecoder(helmsman::mppkit::DecoderConfig config);
    ~_MppFrameDecoder() override;

    // Non-copyable, movable.
    _MppFrameDecoder(const _MppFrameDecoder&) = delete;
    _MppFrameDecoder& operator=(const _MppFrameDecoder&) = delete;
    _MppFrameDecoder(_MppFrameDecoder&&) noexcept;
    _MppFrameDecoder& operator=(_MppFrameDecoder&&) noexcept;

    // Initialize the hardware decoder. Must be called before decode().
    bool init();

    // Decode a compressed packet into a hardware frame (NV12 dmabuf fd).
    bool decode(const uint8_t* data, size_t size, HardwareFrame& out) override;

private:
    helmsman::mppkit::DecoderConfig config_;
    std::unique_ptr<helmsman::mppkit::MppDecoder> decoder_;
};
