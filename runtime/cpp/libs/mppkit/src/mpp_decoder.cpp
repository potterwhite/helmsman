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
// mpp_decoder.cpp — Placeholder implementation of the MppDecoder class
//
// This is a stub for future use. The decoder follows a similar pattern to the
// encoder, but with the data flow reversed:
//
//   Encoder: NV12 frame → VPU → H.264/H.265 packet → file
//   Decoder: H.264/H.265 packet from file → VPU → NV12 frame → user buffer
//
// The full implementation will be added when the decoding path is needed
// (e.g. for video playback, transcoding, or frame-by-frame analysis).
//
// Key differences from encoding:
//   - Uses mpp_init(ctx, MPP_CTX_DEC, ...) instead of MPP_CTX_ENC
//   - Uses decode_put_packet() + decode_get_frame() instead of encode_*
//   - No MppEncCfg — decoder configures itself from the bitstream headers
//   - May need to handle info_change events (resolution change in stream)
//
// =============================================================================

#include "MPPKit/mpp_decoder.h"

#include <cstdio>

extern "C" {
#include "rk_mpi.h"
}

namespace arcforge {
namespace mppkit {

struct MppDecoder::DecoderImpl {
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    bool is_open = false;
};

MppDecoder::MppDecoder(DecoderConfig /*config*/)
    : decoder_impl_(std::make_unique<DecoderImpl>()) {
    // TODO: store config for Init() to use
}

MppDecoder::~MppDecoder() {
    Close();
}

MppDecoder::MppDecoder(MppDecoder&&) noexcept = default;
MppDecoder& MppDecoder::operator=(MppDecoder&&) noexcept = default;

bool MppDecoder::Init() {
    if (decoder_impl_->is_open) {
        return true;
    }

    MPP_RET ret = mpp_create(&decoder_impl_->ctx, &decoder_impl_->mpi);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppDecoder::Init: mpp_create failed, ret=%d\n", ret);
        return false;
    }

    ret = mpp_init(decoder_impl_->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppDecoder::Init: mpp_init failed, ret=%d\n", ret);
        mpp_destroy(decoder_impl_->ctx);
        decoder_impl_->ctx = nullptr;
        return false;
    }

    decoder_impl_->is_open = true;
    return true;
}

bool MppDecoder::DecodeNextFrame(uint8_t* /*nv12_out*/) {
    if (!decoder_impl_->is_open) {
        return false;
    }

    // TODO: implement the decode loop:
    //   1. Read a chunk from the input file
    //   2. Create MppPacket from the chunk
    //   3. decode_put_packet() — submit to VPU
    //   4. decode_get_frame() — retrieve decoded frame
    //   5. Copy frame data from DMA buffer to nv12_out
    //   6. Release MppFrame and MppPacket

    fprintf(stderr, "[MPPKit] MppDecoder::DecodeNextFrame: not implemented yet\n");
    return false;
}

void MppDecoder::Close() {
    if (!decoder_impl_) {
        return;
    }

    if (decoder_impl_->ctx) {
        mpp_destroy(decoder_impl_->ctx);
        decoder_impl_->ctx = nullptr;
        decoder_impl_->mpi = nullptr;
    }

    decoder_impl_->is_open = false;
}

bool MppDecoder::IsOpen() const {
    return decoder_impl_ && decoder_impl_->is_open;
}

}  // namespace mppkit
}  // namespace arcforge
