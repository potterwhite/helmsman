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
// MPPKit/mpp_codec.h — Abstract base class for MPP hardware codecs
//
// This file defines the common interface for all MPP (Media Process Platform)
// hardware codecs on Rockchip RK3588. The VPU (Video Processing Unit) supports
// hardware-accelerated video encoding and decoding for H.264 (AVC) and
// H.265 (HEVC), among others.
//
// Architecture:
//
//   MppCodec  (abstract base — this file)
//     ├── MppEncoder  (encodes raw NV12 frames → compressed H.264/H.265 bitstream)
//     └── MppDecoder  (decodes compressed bitstream → raw NV12 frames)
//
// How to use:
//
//   1. Create a codec via the factory functions:
//        auto encoder = mppkit::CreateEncoder(config);
//   2. Initialize:
//        encoder->Init();
//   3. Cast to the concrete type for operation-specific methods:
//        auto* enc = static_cast<mppkit::MppEncoder*>(encoder.get());
//        enc->EncodeFrame(nv12_data);
//   4. Clean up explicitly or let the destructor handle it:
//        encoder->Close();
//
// How to extend:
//
//   To add a new codec type (e.g. AV1):
//     1. Add kAV1 to CodecType enum
//     2. Create MppAv1Encoder/MppAv1Decoder subclass
//     3. Handle kAV1 in CreateEncoder()/CreateDecoder() factory
//
//   To add a new operation (e.g. transcoder):
//     1. Create a new class inheriting from MppCodec
//     2. Implement Init() and Close()
//     3. Add a new factory function
//
// =============================================================================

#pragma once

#include "MPPKit/pch.h"

namespace helmsman {
namespace mppkit {

// ---------------------------------------------------------------------------
// CodecType — which video compression standard to use
//
// MPP hardware supports multiple codecs. This enum selects which one.
// Add new entries here as Rockchip adds hardware support (e.g. AV1 in future SoCs).
// ---------------------------------------------------------------------------
enum class CodecType {
    kH264,  // H.264 / AVC — most widely compatible, good quality/speed balance
    kH265,  // H.265 / HEVC — better compression ratio, newer hardware
};

// ---------------------------------------------------------------------------
// EncoderConfig — parameters for creating a hardware encoder
//
// All fields have sensible defaults. Only output_path is required.
// ---------------------------------------------------------------------------
struct EncoderConfig {
    CodecType codec_type = CodecType::kH264;  // Which codec to use
    int width = 1920;                          // Frame width in pixels
    int height = 1080;                         // Frame height in pixels
    int fps = 30;                              // Target frames per second
    int bitrate = 4'000'000;                   // Target bitrate in bits per second (4 Mbps)
    std::string output_path;                   // File path for the output bitstream (.h264 / .h265)
};

// ---------------------------------------------------------------------------
// DecoderConfig — parameters for creating a hardware decoder
// ---------------------------------------------------------------------------
struct DecoderConfig {
    CodecType codec_type = CodecType::kH264;  // Which codec the bitstream uses
    int width = 1920;                          // Expected frame width
    int height = 1080;                         // Expected frame height
    std::string input_path;                    // File path for the input bitstream
};

// ---------------------------------------------------------------------------
// MppCodec — abstract base class for all MPP hardware codecs
//
// Lifecycle:  CreateEncoder() / CreateDecoder() → Init() → use → Close()
//
// Non-copyable (owns hardware resources that cannot be duplicated).
// Movable (transfers ownership of the hardware context).
//
// Memory management: uses std::unique_ptr exclusively. No shared_ptr.
// ---------------------------------------------------------------------------
class MppCodec {
public:
    virtual ~MppCodec() = default;

    // No copies — hardware resources cannot be duplicated.
    MppCodec(const MppCodec&) = delete;
    MppCodec& operator=(const MppCodec&) = delete;

    // Move is allowed — transfers ownership of the hardware context.
    MppCodec(MppCodec&&) noexcept = default;
    MppCodec& operator=(MppCodec&&) noexcept = default;

    // Initialize the codec hardware context. Must be called before any
    // encode/decode operation. Returns true on success.
    // On failure, check stderr/log for MPP error details.
    virtual bool Init() = 0;

    // Release all hardware resources. Safe to call multiple times.
    // Also called automatically by the destructor if not already closed.
    virtual void Close() = 0;

    // Returns true if Init() succeeded and Close() has not been called yet.
    virtual bool IsOpen() const = 0;

protected:
    // Only subclasses can construct the base class.
    MppCodec() = default;
};

// ---------------------------------------------------------------------------
// Factory Functions — the intended way to create codec instances
//
// These return std::unique_ptr<MppCodec> for clear ownership semantics.
// The caller owns the returned object exclusively.
//
// Returns nullptr if:
//   - The codec type is not supported by the hardware
//   - Internal resource allocation fails
// ---------------------------------------------------------------------------

// Create a hardware encoder for the given codec type.
std::unique_ptr<MppCodec> CreateEncoder(EncoderConfig config);

// Create a hardware decoder for the given codec type.
std::unique_ptr<MppCodec> CreateDecoder(DecoderConfig config);

}  // namespace mppkit
}  // namespace helmsman
