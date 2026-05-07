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
// MPPKit/mpp_decoder.h — Hardware video decoder using the RK3588 VPU
//
// Placeholder for future use. The full implementation will be added when the
// decoding path is needed (e.g. for video playback or transcoding).
//
// =============================================================================

#pragma once

#include "MPPKit/mpp_codec.h"

namespace arcforge {
namespace mppkit {

class MppDecoder : public MppCodec {
public:
    explicit MppDecoder(DecoderConfig config);
    ~MppDecoder() override;

    MppDecoder(MppDecoder&&) noexcept;
    MppDecoder& operator=(MppDecoder&&) noexcept;

    bool Init() override;
    void Close() override;
    bool IsOpen() const override;

    // Decode the next frame from the bitstream into nv12_out.
    // Returns true if a frame was decoded, false at EOF or error.
    bool DecodeNextFrame(uint8_t* nv12_out);

private:
    struct DecoderImpl;
    std::unique_ptr<DecoderImpl> decoder_impl_;
};

}  // namespace mppkit
}  // namespace arcforge
