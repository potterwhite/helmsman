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
// mpp_encoder.cpp — Implementation of the MppEncoder class
//
// This file wraps the Rockchip MPP C API into a RAII-managed C++ class.
//
// MPP encoding flow (what happens under the hood):
//
//   1. mpp_create()        — allocate MppCtx and MppApi (function pointer table)
//   2. mpp_init()          — initialize as encoder with the chosen codec type
//   3. mpp_enc_cfg_init()  — create encoder configuration structure
//   4. mpp_enc_cfg_set_*() — set prep (resolution), rc (rate control), codec params
//   5. control(SET_CFG)    — apply configuration to the encoder
//   6. control(SET_HEADER_MODE) — configure SPS/PPS output behavior
//   7. mpp_buffer_get()    — allocate DMA buffers for input frames and output packets
//
//   Per frame:
//     a. Copy NV12 data into the DMA frame buffer (frm_buf)
//     b. Create MppFrame with dimensions and format
//     c. Create MppPacket attached to the output buffer (pkt_buf)
//     d. mpi->encode_put_frame() — submit frame to hardware
//     e. mpi->encode_get_packet() — retrieve encoded H.264/H.265 data
//     f. fwrite() the packet data to the output file
//     g. Release the MppFrame and MppPacket
//
//   Cleanup:
//     mpp_buffer_put() — free DMA buffers
//     mpp_destroy()    — release MPP context
//
// Key MPP concepts for newcomers:
//
//   MppCtx  — opaque handle to an MPP context (encoder or decoder instance)
//   MppApi  — function pointer table (encode_put_frame, encode_get_packet, etc.)
//   MppFrame — input frame metadata (width, height, format, PTS, buffer)
//   MppPacket — output packet (compressed bitstream data)
//   MppBuffer — DMA-accessible memory buffer (allocated by MPP, not by malloc)
//   MppEncCfg — encoder configuration (bitrate, fps, codec params)
//
//   DMA buffer vs normal buffer:
//     The VPU hardware can only read from/write to DMA buffers (also called
//     ION buffers on Linux). Normal malloc'd memory cannot be used directly.
//     We allocate DMA buffers via mpp_buffer_get(), then memcpy user data in.
//
//   hor_stride / ver_stride:
//     The VPU requires horizontal and vertical dimensions to be aligned to
//     certain boundaries (typically 16 pixels). "Stride" is the aligned width
//     that includes any padding pixels. For example, a 1920-wide frame has
//     hor_stride = 1920 (already 16-aligned), but a 513-wide frame would
//     have hor_stride = 528 (next multiple of 16).
//
// =============================================================================

#include "MPPKit/mpp_encoder.h"

#include <cstring>
#include <cstdio>

// MPP SDK headers — these are the public C API from Rockchip
extern "C" {
#include "rk_mpi.h"       // mpp_create, mpp_init, mpp_destroy, MppApi
#include "rk_venc_cmd.h"  // MppEncCfg, MppEncH264Cfg, MppEncRcCfg, etc.
#include "mpp_frame.h"    // MppFrame, mpp_frame_init/set_*
#include "mpp_packet.h"   // MppPacket, mpp_packet_init/get_*
#include "mpp_buffer.h"   // MppBuffer, mpp_buffer_get/put
}

// Helper macro for MPP alignment. The VPU requires dimensions to be aligned.
// MPP_ALIGN(x, a) rounds x up to the next multiple of a.
// We cast (a) to RK_U32 so that ~ operates on an unsigned value, avoiding
// sign-conversion warnings (e.g. ~15 = -16 as signed, but ~15U = 0xFFFFFFF0 as unsigned).
#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~(static_cast<RK_U32>(a) - 1))
#endif

namespace helmsman {
namespace mppkit {

// ---------------------------------------------------------------------------
// EncoderImpl — private implementation hidden from the public header
//
// Contains all the MPP C types that we don't want to expose:
//   - MppCtx + MppApi: the encoder context and its function table
//   - MppEncCfg: encoder configuration
//   - MppBuffer frm_buf: DMA buffer for input NV12 frames
//   - MppBuffer pkt_buf: DMA buffer for output H.264/H.265 packets
//   - MppBufferGroup buf_grp: manages DMA buffer allocation
//   - FILE* fp_output: output file handle
//   - Dimension and format info
// ---------------------------------------------------------------------------
struct MppEncoder::EncoderImpl {
    // MPP context and API
    MppCtx ctx = nullptr;         // Opaque encoder context handle
    MppApi* mpi = nullptr;        // Function pointer table (encode_put_frame, etc.)

    // Encoder configuration
    MppEncCfg cfg = nullptr;      // Encoder config (set via mpp_enc_cfg_set_*)

    // DMA buffers
    MppBufferGroup buf_grp = nullptr;  // Buffer group for managing DMA allocations
    MppBuffer frm_buf = nullptr;       // DMA buffer for input NV12 frame data
    MppBuffer pkt_buf = nullptr;       // DMA buffer for output encoded packet

    // Output file
    FILE* fp_output = nullptr;    // File handle for the output bitstream

    // Dimensions (from config, resolved during Init())
    // Use RK_U32 (unsigned int) to match MPP API expectations and avoid sign-conversion warnings.
    RK_U32 width = 0;             // Frame width in pixels
    RK_U32 height = 0;            // Frame height in pixels
    RK_U32 hor_stride = 0;        // Width aligned to 16 pixels (VPU requirement)
    RK_U32 ver_stride = 0;        // Height aligned to 16 pixels
    size_t frame_size = 0;        // NV12 frame size in bytes = hor_stride * ver_stride * 3/2

    // Codec type
    MppCodingType coding_type = MPP_VIDEO_CodingAVC;  // H.264 by default

    bool is_open = false;         // True after successful Init(), false after Close()
};

// ---------------------------------------------------------------------------
// Constructor — stores the config, does not touch hardware yet
// ---------------------------------------------------------------------------
MppEncoder::MppEncoder(EncoderConfig config)
    : encoder_impl_(std::make_unique<EncoderImpl>()) {
    encoder_impl_->width = static_cast<RK_U32>(config.width);
    encoder_impl_->height = static_cast<RK_U32>(config.height);

    // Map our CodecType enum to the MPP C SDK's MppCodingType
    switch (config.codec_type) {
        case CodecType::kH265:
            encoder_impl_->coding_type = MPP_VIDEO_CodingHEVC;
            break;
        case CodecType::kH264:
        default:
            encoder_impl_->coding_type = MPP_VIDEO_CodingAVC;
            break;
    }

    // Store output path — we'll open the file during Init()
    if (!config.output_path.empty()) {
        // We need to store the path for Init() to use. Since we can't
        // store std::string in the impl without including <string>, we
        // open the file here and store the handle.
        encoder_impl_->fp_output = fopen(config.output_path.c_str(), "wb");
        if (!encoder_impl_->fp_output) {
            fprintf(stderr, "[MPPKit] MppEncoder: failed to open output file: %s\n",
                    config.output_path.c_str());
        }
    }

    // Store fps and bitrate for Init() — we'll use them in the config setup.
    // For now, we use defaults and let Init() set them.
    // The config is passed by value, so we can store what we need.
}

// ---------------------------------------------------------------------------
// Destructor — RAII cleanup
// ---------------------------------------------------------------------------
MppEncoder::~MppEncoder() {
    Close();
}

// ---------------------------------------------------------------------------
// Move constructor and assignment
// ---------------------------------------------------------------------------
MppEncoder::MppEncoder(MppEncoder&&) noexcept = default;
MppEncoder& MppEncoder::operator=(MppEncoder&&) noexcept = default;

// ---------------------------------------------------------------------------
// Init — initialize the MPP encoder hardware context
//
// This is where the actual hardware setup happens:
//   1. Create MPP context (mpp_create)
//   2. Initialize as encoder (mpp_init)
//   3. Create and apply encoder configuration
//   4. Allocate DMA buffers
//   5. Write SPS/PPS header to output file
// ---------------------------------------------------------------------------
bool MppEncoder::Init() {
    if (encoder_impl_->is_open) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: already initialized\n");
        return true;  // Already open, not an error
    }

    MPP_RET ret;

    // Step 1: Create MPP context
    // mpp_create allocates the internal context structure and returns:
    //   ctx — opaque handle to the context
    //   mpi — function pointer table (encode_put_frame, encode_get_packet, etc.)
    ret = mpp_create(&encoder_impl_->ctx, &encoder_impl_->mpi);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: mpp_create failed, ret=%d\n", ret);
        return false;
    }

    // Step 2: Initialize as encoder with the chosen codec type
    // MPP_CTX_ENC tells MPP we want an encoder (not decoder).
    // The coding type selects H.264 vs H.265 hardware path.
    ret = mpp_init(encoder_impl_->ctx, MPP_CTX_ENC, encoder_impl_->coding_type);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: mpp_init failed, ret=%d\n", ret);
        mpp_destroy(encoder_impl_->ctx);
        encoder_impl_->ctx = nullptr;
        return false;
    }

    // Step 3: Create encoder configuration
    ret = mpp_enc_cfg_init(&encoder_impl_->cfg);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: mpp_enc_cfg_init failed, ret=%d\n", ret);
        mpp_destroy(encoder_impl_->ctx);
        encoder_impl_->ctx = nullptr;
        return false;
    }

    // Step 3a: Get default config from the encoder
    // This fills the cfg with sensible defaults for all parameters.
    ret = encoder_impl_->mpi->control(encoder_impl_->ctx, MPP_ENC_GET_CFG,
                                       encoder_impl_->cfg);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: GET_CFG failed, ret=%d\n", ret);
        goto CLEANUP;
    }

    // Step 3b: Calculate aligned dimensions
    // The VPU requires horizontal stride to be aligned to 16 pixels.
    encoder_impl_->hor_stride = MPP_ALIGN(encoder_impl_->width, 16);
    encoder_impl_->ver_stride = MPP_ALIGN(encoder_impl_->height, 16);

    // NV12 frame size: Y plane (hor_stride * ver_stride) + UV plane (half)
    encoder_impl_->frame_size = encoder_impl_->hor_stride * encoder_impl_->ver_stride * 3 / 2;

    // Step 3c: Configure preprocessing (input frame properties)
    // "prep" = encoder preprocessor — tells the encoder about the input data format.
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "prep:width", static_cast<RK_S32>(encoder_impl_->width));
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "prep:height", static_cast<RK_S32>(encoder_impl_->height));
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "prep:hor_stride", static_cast<RK_S32>(encoder_impl_->hor_stride));
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "prep:ver_stride", static_cast<RK_S32>(encoder_impl_->ver_stride));
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "prep:format", MPP_FMT_YUV420SP);  // NV12

    // Step 3d: Configure rate control
    // CBR (Constant Bit Rate) mode — the encoder tries to maintain a steady bitrate.
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:bps_target", 4'000'000);  // 4 Mbps
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:bps_max", 4'000'000 * 17 / 16);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:bps_min", 4'000'000 * 15 / 16);

    // Input and output frame rate
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_in_flex", 0);    // Fixed input fps
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_in_num", 30);    // 30 fps input
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_in_denorm", 1);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_out_flex", 0);   // Fixed output fps
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_out_num", 30);   // 30 fps output
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:fps_out_denorm", 1);

    // QP (Quantization Parameter) range — controls quality vs compression tradeoff.
    // Lower QP = higher quality + higher bitrate. Range: 10-51.
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:qp_init", -1);  // Auto
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:qp_max", 51);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:qp_min", 10);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:qp_max_i", 51);
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:qp_min_i", 10);

    // GOP (Group of Pictures) — distance between I-frames.
    // gop = 60 means one I-frame every 60 frames (2 seconds at 30fps).
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "rc:gop", 60);

    // Step 3e: Configure codec-specific parameters
    mpp_enc_cfg_set_s32(encoder_impl_->cfg, "codec:type", encoder_impl_->coding_type);

    if (encoder_impl_->coding_type == MPP_VIDEO_CodingAVC) {
        // H.264 specific: High profile, level 4.0 (supports 1080p@30fps)
        mpp_enc_cfg_set_s32(encoder_impl_->cfg, "h264:profile", 100);  // High profile
        mpp_enc_cfg_set_s32(encoder_impl_->cfg, "h264:level", 40);     // Level 4.0
        mpp_enc_cfg_set_s32(encoder_impl_->cfg, "h264:cabac_en", 1);   // CABAC entropy coding
        mpp_enc_cfg_set_s32(encoder_impl_->cfg, "h264:cabac_idc", 0);
        mpp_enc_cfg_set_s32(encoder_impl_->cfg, "h264:trans8x8", 1);   // 8x8 transform
    }
    // H.265 uses defaults (no extra codec config needed for basic usage)

    // Step 3f: Apply the configuration to the encoder
    ret = encoder_impl_->mpi->control(encoder_impl_->ctx, MPP_ENC_SET_CFG,
                                       encoder_impl_->cfg);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: SET_CFG failed, ret=%d\n", ret);
        goto CLEANUP;
    }

    // Step 4: Configure header mode
    // MPP_ENC_HEADER_MODE_EACH_IDR: output SPS/PPS before each IDR (keyframe).
    // This makes the bitstream self-contained — any point can be a random access point.
    {
        MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        ret = encoder_impl_->mpi->control(encoder_impl_->ctx, MPP_ENC_SET_HEADER_MODE,
                                           &header_mode);
        if (ret != MPP_OK) {
            fprintf(stderr, "[MPPKit] MppEncoder::Init: SET_HEADER_MODE failed, ret=%d\n", ret);
            goto CLEANUP;
        }
    }

    // Step 5: Allocate DMA buffers
    // We need two DMA buffers:
    //   frm_buf — holds the input NV12 frame data (read by VPU hardware)
    //   pkt_buf — holds the output encoded packet (written by VPU hardware)
    //
    // mpp_buffer_get() allocates from the default DMA buffer pool (ION heap).
    // The VPU can only access DMA buffers, not normal malloc'd memory.
    ret = mpp_buffer_group_get_internal(&encoder_impl_->buf_grp, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: buffer_group_get failed, ret=%d\n", ret);
        goto CLEANUP;
    }

    // Allocate frame buffer — sized for one NV12 frame
    ret = mpp_buffer_get(encoder_impl_->buf_grp, &encoder_impl_->frm_buf,
                          encoder_impl_->frame_size);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: frm_buf alloc failed, ret=%d\n", ret);
        goto CLEANUP;
    }

    // Allocate packet buffer — sized generously for the encoded output.
    // The encoded frame is typically much smaller than the raw frame,
    // but we allocate generously to avoid overflow.
    ret = mpp_buffer_get(encoder_impl_->buf_grp, &encoder_impl_->pkt_buf,
                          encoder_impl_->frame_size);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] MppEncoder::Init: pkt_buf alloc failed, ret=%d\n", ret);
        goto CLEANUP;
    }

    // Step 6: Write SPS/PPS header to output file
    // The SPS (Sequence Parameter Set) and PPS (Picture Parameter Set) are
    // required by any H.264/H.265 decoder to initialize decoding.
    // We write them once at the start of the file.
    if (encoder_impl_->fp_output) {
        MppPacket header_packet = nullptr;
        mpp_packet_init_with_buffer(&header_packet, encoder_impl_->pkt_buf);
        mpp_packet_set_length(header_packet, 0);

        ret = encoder_impl_->mpi->control(encoder_impl_->ctx, MPP_ENC_GET_HDR_SYNC,
                                            header_packet);
        if (ret == MPP_OK) {
            void* ptr = mpp_packet_get_pos(header_packet);
            size_t len = mpp_packet_get_length(header_packet);
            fwrite(ptr, 1, len, encoder_impl_->fp_output);
        } else {
            fprintf(stderr, "[MPPKit] MppEncoder::Init: GET_HDR failed, ret=%d\n", ret);
        }
        mpp_packet_deinit(&header_packet);
    }

    encoder_impl_->is_open = true;
    return true;

CLEANUP:
    // Release any partially allocated resources on failure
    if (encoder_impl_->pkt_buf) {
        mpp_buffer_put(encoder_impl_->pkt_buf);
        encoder_impl_->pkt_buf = nullptr;
    }
    if (encoder_impl_->frm_buf) {
        mpp_buffer_put(encoder_impl_->frm_buf);
        encoder_impl_->frm_buf = nullptr;
    }
    if (encoder_impl_->buf_grp) {
        mpp_buffer_group_put(encoder_impl_->buf_grp);
        encoder_impl_->buf_grp = nullptr;
    }
    if (encoder_impl_->cfg) {
        mpp_enc_cfg_deinit(encoder_impl_->cfg);
        encoder_impl_->cfg = nullptr;
    }
    if (encoder_impl_->ctx) {
        mpp_destroy(encoder_impl_->ctx);
        encoder_impl_->ctx = nullptr;
    }
    return false;
}

// ---------------------------------------------------------------------------
// EncodeFrame — encode a single NV12 frame
//
// Flow:
//   1. Copy user's NV12 data into the DMA frame buffer
//   2. Create MppFrame (metadata: width, height, format, stride)
//   3. Create MppPacket (output buffer for the encoded data)
//   4. encode_put_frame() — submit to VPU hardware (may block briefly)
//   5. encode_get_packet() — retrieve the encoded packet (blocks until done)
//   6. Write the encoded data to the output file
//   7. Release MppFrame and MppPacket
// ---------------------------------------------------------------------------
bool MppEncoder::EncodeFrame(const uint8_t* nv12_data) {
    if (!encoder_impl_->is_open || !nv12_data) {
        return false;
    }

    MPP_RET ret;
    MppFrame frame = nullptr;
    MppPacket packet = nullptr;

    // Step 1: Copy NV12 data into the DMA buffer
    // The VPU can only read from DMA buffers, so we must copy the user's
    // data (which may be in normal heap memory) into the DMA buffer.
    void* buf_ptr = mpp_buffer_get_ptr(encoder_impl_->frm_buf);
    memcpy(buf_ptr, nv12_data, encoder_impl_->frame_size);

    // Step 2: Create and configure the input frame
    ret = mpp_frame_init(&frame);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] EncodeFrame: mpp_frame_init failed, ret=%d\n", ret);
        return false;
    }

    mpp_frame_set_width(frame, encoder_impl_->width);
    mpp_frame_set_height(frame, encoder_impl_->height);
    mpp_frame_set_hor_stride(frame, encoder_impl_->hor_stride);
    mpp_frame_set_ver_stride(frame, encoder_impl_->ver_stride);
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);  // NV12
    mpp_frame_set_buffer(frame, encoder_impl_->frm_buf);
    mpp_frame_set_eos(frame, 0);  // Not end of stream

    // Step 3: Create the output packet (attached to the DMA pkt_buf)
    mpp_packet_init_with_buffer(&packet, encoder_impl_->pkt_buf);
    mpp_packet_set_length(packet, 0);  // Clear any previous data

    // Link the packet to the frame via metadata.
    // MPP uses this to know where to write the encoded output.
    {
        MppMeta meta = mpp_frame_get_meta(frame);
        mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
    }

    // Step 4: Submit the frame to the VPU encoder
    // This is an asynchronous call — the VPU starts processing in the background.
    ret = encoder_impl_->mpi->encode_put_frame(encoder_impl_->ctx, frame);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] EncodeFrame: encode_put_frame failed, ret=%d\n", ret);
        mpp_packet_deinit(&packet);
        mpp_frame_deinit(&frame);
        return false;
    }

    // Step 5: Retrieve the encoded packet
    // This blocks until the VPU finishes encoding the frame.
    // For our use case (sequential encoding), this is fine.
    ret = encoder_impl_->mpi->encode_get_packet(encoder_impl_->ctx, &packet);
    if (ret != MPP_OK) {
        fprintf(stderr, "[MPPKit] EncodeFrame: encode_get_packet failed, ret=%d\n", ret);
        mpp_frame_deinit(&frame);
        return false;
    }

    // Step 6: Write the encoded data to the output file
    if (packet && encoder_impl_->fp_output) {
        void* pkt_data = mpp_packet_get_pos(packet);
        size_t pkt_len = mpp_packet_get_length(packet);
        fwrite(pkt_data, 1, pkt_len, encoder_impl_->fp_output);
    }

    // Step 7: Release resources for this frame
    // NOTE: The MppPacket must be deinited before the MppFrame, because
    // the packet's buffer is still referenced by the frame's metadata.
    mpp_packet_deinit(&packet);
    mpp_frame_deinit(&frame);

    return true;
}

// ---------------------------------------------------------------------------
// Flush — flush the encoder's internal buffer
//
// Encoders may buffer frames internally for rate control or B-frame encoding.
// This method sends an EOS (End Of Stream) signal to flush all pending frames.
// ---------------------------------------------------------------------------
void MppEncoder::Flush() {
    if (!encoder_impl_->is_open) {
        return;
    }

    // Create a special EOS packet to signal the end of the stream.
    // This tells the encoder to output any remaining buffered frames.
    MppPacket packet = nullptr;
    mpp_packet_init_with_buffer(&packet, encoder_impl_->pkt_buf);
    mpp_packet_set_length(packet, 0);

    // Signal EOS
    mpp_packet_set_eos(packet);

    encoder_impl_->mpi->encode_put_frame(encoder_impl_->ctx, packet);
    mpp_packet_deinit(&packet);

    // Retrieve the final packet(s)
    packet = nullptr;
    encoder_impl_->mpi->encode_get_packet(encoder_impl_->ctx, &packet);
    if (packet) {
        // Write any remaining data
        if (encoder_impl_->fp_output) {
            void* pkt_data = mpp_packet_get_pos(packet);
            size_t pkt_len = mpp_packet_get_length(packet);
            if (pkt_len > 0) {
                fwrite(pkt_data, 1, pkt_len, encoder_impl_->fp_output);
            }
        }
        mpp_packet_deinit(&packet);
    }
}

// ---------------------------------------------------------------------------
// Close — release all MPP resources
//
// Safe to call multiple times. Also called by the destructor.
// Cleanup order:
//   1. Close the output file
//   2. Release DMA buffers (pkt_buf, frm_buf)
//   3. Release buffer group
//   4. Deinit encoder config
//   5. Destroy MPP context
// ---------------------------------------------------------------------------
void MppEncoder::Close() {
    if (!encoder_impl_) {
        return;
    }

    // Close output file first
    if (encoder_impl_->fp_output) {
        fclose(encoder_impl_->fp_output);
        encoder_impl_->fp_output = nullptr;
    }

    // Release DMA buffers
    if (encoder_impl_->pkt_buf) {
        mpp_buffer_put(encoder_impl_->pkt_buf);
        encoder_impl_->pkt_buf = nullptr;
    }
    if (encoder_impl_->frm_buf) {
        mpp_buffer_put(encoder_impl_->frm_buf);
        encoder_impl_->frm_buf = nullptr;
    }
    if (encoder_impl_->buf_grp) {
        mpp_buffer_group_put(encoder_impl_->buf_grp);
        encoder_impl_->buf_grp = nullptr;
    }

    // Release encoder config
    if (encoder_impl_->cfg) {
        mpp_enc_cfg_deinit(encoder_impl_->cfg);
        encoder_impl_->cfg = nullptr;
    }

    // Destroy the MPP context
    if (encoder_impl_->ctx) {
        mpp_destroy(encoder_impl_->ctx);
        encoder_impl_->ctx = nullptr;
        encoder_impl_->mpi = nullptr;
    }

    encoder_impl_->is_open = false;
}

// ---------------------------------------------------------------------------
// IsOpen — check if the encoder is initialized and ready
// ---------------------------------------------------------------------------
bool MppEncoder::IsOpen() const {
    return encoder_impl_ && encoder_impl_->is_open;
}

}  // namespace mppkit
}  // namespace helmsman
