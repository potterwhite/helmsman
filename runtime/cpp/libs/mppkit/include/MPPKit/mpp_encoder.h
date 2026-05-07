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
// MPPKit/mpp_encoder.h — Hardware video encoder using the RK3588 VPU
//
// This class encodes raw video frames in NV12 pixel format into compressed
// H.264 or H.265 bitstream using the hardware Video Processing Unit (VPU)
// on the RK3588 SoC.
//
// Why NV12?
//   NV12 (also called YUV420SP) is the native pixel format for the VPU.
//   It stores luma (Y) plane first, then interleaved chroma (UV) plane.
//   Frame size = width * height * 3 / 2 bytes.
//   If your input is BGR (OpenCV), use RGAKit to convert BGR→NV12 first.
//
// Usage:
//   EncoderConfig config;
//   config.width = 1920;
//   config.height = 1080;
//   config.fps = 30;
//   config.bitrate = 4'000'000;
//   config.output_path = "/tmp/output.h264";
//
//   MppEncoder encoder(std::move(config));
//   encoder.Init();
//
//   // For each frame:
//   encoder.EncodeFrame(nv12_data_ptr);
//
//   // At the end:
//   encoder.Flush();   // flush any buffered frames (B-frames, etc.)
//   encoder.Close();   // or let the destructor handle it
//
// =============================================================================

#pragma once

#include "MPPKit/mpp_codec.h"

namespace arcforge {
namespace mppkit {

class MppEncoder : public MppCodec {
public:
    explicit MppEncoder(EncoderConfig config);
    ~MppEncoder() override;

    MppEncoder(MppEncoder&&) noexcept;
    MppEncoder& operator=(MppEncoder&&) noexcept;

    // --- MppCodec interface ---
    bool Init() override;
    void Close() override;
    bool IsOpen() const override;

    // --- Encoder-specific operations ---

    // Encode a single NV12 frame and write the compressed data to the output file.
    // - nv12_data: pointer to NV12 pixel data (width * height * 1.5 bytes total)
    // - Returns true on success.
    bool EncodeFrame(const uint8_t* nv12_data);

    // Flush the encoder's internal buffer.
    // Call this before Close() to ensure all pending frames are written.
    void Flush();

private:
    struct EncoderImpl;
    std::unique_ptr<EncoderImpl> encoder_impl_;
};

}  // namespace mppkit
}  // namespace arcforge
