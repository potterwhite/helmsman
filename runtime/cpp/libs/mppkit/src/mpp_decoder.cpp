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
// mpp_decoder.cpp — Hardware video decoder implementation
//
// Decoding flow (per frame):
//
//   1. Create MppPacket from the compressed bitstream data
//   2. mpi->decode_put_packet() — submit packet to VPU
//   3. mpi->decode_get_frame()  — retrieve decoded frame (blocks until ready)
//   4. Extract the DMA buffer fd from the decoded MppFrame
//   5. Release the MppFrame and MppPacket
//
// Key differences from encoding:
//   - Uses MPP_CTX_DEC instead of MPP_CTX_ENC
//   - No MppEncCfg — decoder auto-configures from bitstream headers
//   - Output buffer is allocated by MPP (not by us) — we just read the fd
//   - May need to handle info_change events (resolution change mid-stream)
//
// =============================================================================

#include "MPPKit/mpp_decoder.h"

#include <cstring>
#include <cstdio>

extern "C" {
#include "rk_mpi.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_buffer.h"
}

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~(static_cast<RK_U32>(a) - 1))
#endif

namespace helmsman {
namespace mppkit {

struct MppDecoder::DecoderImpl {
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    bool is_open = false;

    DecoderConfig config;
    MppCodingType coding_type = MPP_VIDEO_CodingAVC;
    RK_U32 width = 0;
    RK_U32 height = 0;
    RK_U32 hor_stride = 0;
    RK_U32 ver_stride = 0;
};

MppDecoder::MppDecoder(DecoderConfig config)
    : decoder_impl_(std::make_unique<DecoderImpl>()) {
    decoder_impl_->config = config;
    decoder_impl_->width = static_cast<RK_U32>(config.width);
    decoder_impl_->height = static_cast<RK_U32>(config.height);
    decoder_impl_->hor_stride = MPP_ALIGN(decoder_impl_->width, 16);
    decoder_impl_->ver_stride = MPP_ALIGN(decoder_impl_->height, 16);

    switch (config.codec_type) {
        case CodecType::kH265:
            decoder_impl_->coding_type = MPP_VIDEO_CodingHEVC;
            break;
        case CodecType::kH264:
        default:
            decoder_impl_->coding_type = MPP_VIDEO_CodingAVC;
            break;
    }
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

    ret = mpp_init(decoder_impl_->ctx, MPP_CTX_DEC, decoder_impl_->coding_type);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppDecoder::Init: mpp_init failed, ret=%d\n", ret);
        mpp_destroy(decoder_impl_->ctx);
        decoder_impl_->ctx = nullptr;
        return false;
    }

    decoder_impl_->is_open = true;
    return true;
}

bool MppDecoder::DecodeNextFrame(const uint8_t* packet_data, size_t packet_size,
                                  DecodedFrame& out) {
    if (!decoder_impl_->is_open || !packet_data || packet_size == 0) {
        return false;
    }

    out = {};
    MPP_RET ret;

    // Standard MPP decode: submit packet, drain frames on buffer full.
    // When decode_put_packet returns -1012 (MPP_ERR_BUFFER_FULL), drain
    // a decoded frame to free internal buffer space, then retry.
    // If the drain produces a valid frame, return it immediately.
    constexpr int kMaxRetries = 16;
    bool packet_accepted = false;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        MppPacket packet = nullptr;
        ret = mpp_packet_init(&packet, const_cast<uint8_t*>(packet_data), packet_size);
        if (ret != MPP_OK) {
            fprintf(stderr, "[MPPKit] DecodeNextFrame: mpp_packet_init failed, ret=%d\n", ret);
            return false;
        }

        ret = decoder_impl_->mpi->decode_put_packet(decoder_impl_->ctx, packet);
        mpp_packet_deinit(&packet);

        if (ret == MPP_OK) {
            packet_accepted = true;
            break;
        }

        // Buffer full — drain a frame. If it's a real frame, return it.
        MppFrame frame = nullptr;
        ret = decoder_impl_->mpi->decode_get_frame(decoder_impl_->ctx, &frame);
        if (ret == MPP_OK && frame) {
            if (mpp_frame_get_info_change(frame)) {
                decoder_impl_->width = mpp_frame_get_width(frame);
                decoder_impl_->height = mpp_frame_get_height(frame);
                decoder_impl_->hor_stride = mpp_frame_get_hor_stride(frame);
                decoder_impl_->ver_stride = mpp_frame_get_ver_stride(frame);
                decoder_impl_->mpi->control(decoder_impl_->ctx,
                                            MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
                mpp_frame_deinit(&frame);
            } else {
                // Got a real decoded frame — return it.
                MppBuffer buf = mpp_frame_get_buffer(frame);
                if (buf) {
                    out.fd = mpp_buffer_get_fd(buf);
                    out.width = static_cast<int>(mpp_frame_get_width(frame));
                    out.height = static_cast<int>(mpp_frame_get_height(frame));
                    out.format = static_cast<int>(mpp_frame_get_fmt(frame));
                }
                mpp_frame_deinit(&frame);
                if (out.fd >= 0) {
                    return true;
                }
            }
        }
    }

    if (!packet_accepted) {
        return false;
    }

    // Retrieve the decoded frame. Loop to handle info_change events
    // (MPP returns the real frame immediately after acknowledgement).
    constexpr int kMaxGetRetries = 16;
    for (int i = 0; i < kMaxGetRetries; ++i) {
        MppFrame frame = nullptr;
        ret = decoder_impl_->mpi->decode_get_frame(decoder_impl_->ctx, &frame);
        if (ret != MPP_OK || !frame) {
            return false;  // Need more data
        }

        if (mpp_frame_get_info_change(frame)) {
            decoder_impl_->width = mpp_frame_get_width(frame);
            decoder_impl_->height = mpp_frame_get_height(frame);
            decoder_impl_->hor_stride = mpp_frame_get_hor_stride(frame);
            decoder_impl_->ver_stride = mpp_frame_get_ver_stride(frame);
            decoder_impl_->mpi->control(decoder_impl_->ctx,
                                        MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            mpp_frame_deinit(&frame);
            continue;
        }

        MppBuffer buf = mpp_frame_get_buffer(frame);
        if (buf) {
            out.fd = mpp_buffer_get_fd(buf);
            out.width = static_cast<int>(mpp_frame_get_width(frame));
            out.height = static_cast<int>(mpp_frame_get_height(frame));
            out.format = static_cast<int>(mpp_frame_get_fmt(frame));
        }
        mpp_frame_deinit(&frame);
        return out.fd >= 0;
    }

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
}  // namespace helmsman
