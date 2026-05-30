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
// ffmpeg-input-source.h — FFmpeg-based input source (Phase 2, internal)
//
// Uses FFmpeg's avformat API to demux mp4 containers and return raw
// compressed packets (H.264/H.265 NAL units) without decoding.
//
// Deferred to Phase 2 — code kept for future use.
//
// =============================================================================

#pragma once

#include "pipeline/stages/frontend/01-input-source/base-input-source.h"

#include <cstdint>
#include <vector>

// Forward declarations for FFmpeg types (avoids exposing FFmpeg headers)
struct AVFormatContext;
struct AVPacket;
struct AVBSFContext;

class FfmpegInputSource : public BaseInputSource {
public:
    FfmpegInputSource();
    ~FfmpegInputSource() override;

    // Non-copyable, movable
    FfmpegInputSource(const FfmpegInputSource&) = delete;
    FfmpegInputSource& operator=(const FfmpegInputSource&) = delete;
    FfmpegInputSource(FfmpegInputSource&&) noexcept;
    FfmpegInputSource& operator=(FfmpegInputSource&&) noexcept;

    bool open(const std::string& uri) override;
    bool ReadRaw(RawPacket& pkt) override;
    int width() const override;
    int height() const override;
    double fps() const override;
    void close() override;

    // Codec ID from the video stream (AV_CODEC_ID_H264 or AV_CODEC_ID_HEVC).
    // Only valid after open() returns true.
    int CodecId() const;

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVPacket* av_packet_ = nullptr;
    AVBSFContext* bsf_ctx_ = nullptr;  // h264/hevc_mp4toannexb bitstream filter
    int video_stream_idx_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int codec_id_ = 0;  // AVCodecID value

    // BSF output packet — av_bsf_receive_packet writes here.
    AVPacket* bsf_packet_ = nullptr;

    // Merged BSF output buffer. When the BSF splits one AVCC input into
    // multiple Annex B NAL units (SPS + PPS + IDR), we merge them into a
    // single contiguous buffer so MPP receives all parameter sets together
    // with the frame data.
    std::vector<uint8_t> merged_buf_;

    bool bsf_eof_ = false;  // true after BSF flush (end of stream)
};
