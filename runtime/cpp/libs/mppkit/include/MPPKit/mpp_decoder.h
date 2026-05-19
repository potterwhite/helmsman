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
// Decodes H.264/H.265 bitstream packets into NV12 frames via hardware.
// Returns decoded frames as DMA buffer file descriptors (dmabuf fd) for
// zero-copy integration with RGA and downstream pipeline stages.
//
// Usage:
//   MppDecoder decoder({.codec_type = CodecType::kH264, .width = 1920, .height = 1080});
//   decoder.Init();
//
//   // Feed compressed packets (e.g. from FFmpeg demuxer)
//   DecodedFrame frame;
//   if (decoder.DecodeNextFrame(packet_data, packet_size, frame)) {
//       // frame.fd is a dmabuf fd containing NV12 data
//       // frame.width, frame.height, frame.format describe the layout
//   }
//
//   decoder.Close();
//
// =============================================================================

#pragma once

#include "MPPKit/mpp_codec.h"

namespace arcforge {
namespace mppkit {

// Decoded frame output — carries a dmabuf fd for zero-copy downstream.
struct DecodedFrame {
    int fd = -1;        // dmabuf file descriptor (NV12 data)
    int width = 0;      // Decoded frame width in pixels
    int height = 0;     // Decoded frame height in pixels
    int format = 0;     // MPP pixel format (MPP_FMT_YUV420SP = NV12)
};

class MppDecoder : public MppCodec {
public:
    explicit MppDecoder(DecoderConfig config);
    ~MppDecoder() override;

    MppDecoder(MppDecoder&&) noexcept;
    MppDecoder& operator=(MppDecoder&&) noexcept;

    bool Init() override;
    void Close() override;
    bool IsOpen() const override;

    // Decode the next frame from a compressed bitstream packet.
    //
    // packet_data / packet_size: compressed H.264/H.265 NAL unit data.
    //   The data must remain valid until this call returns.
    // out: receives the decoded frame's dmabuf fd and dimensions.
    //
    // Returns true if a decoded frame is available.
    // Returns false on error or if more data is needed (decoder is buffering).
    //
    // The returned fd is valid until the next DecodeNextFrame() call or Close().
    bool DecodeNextFrame(const uint8_t* packet_data, size_t packet_size,
                         DecodedFrame& out);

private:
    struct DecoderImpl;
    std::unique_ptr<DecoderImpl> decoder_impl_;
};

}  // namespace mppkit
}  // namespace arcforge
