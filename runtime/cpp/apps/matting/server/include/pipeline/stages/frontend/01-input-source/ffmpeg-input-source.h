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

#include "pipeline/stages/frontend/input-source/i-input-source.h"

// Forward declarations for FFmpeg types (avoids exposing FFmpeg headers)
struct AVFormatContext;
struct AVPacket;

class _FFmpegInputSource : public _IInputSource {
public:
    _FFmpegInputSource();
    ~_FFmpegInputSource() override;

    // Non-copyable, movable
    _FFmpegInputSource(const _FFmpegInputSource&) = delete;
    _FFmpegInputSource& operator=(const _FFmpegInputSource&) = delete;
    _FFmpegInputSource(_FFmpegInputSource&&) noexcept;
    _FFmpegInputSource& operator=(_FFmpegInputSource&&) noexcept;

    bool open(const std::string& uri) override;
    bool readRaw(RawPacket& pkt) override;
    int width() const override;
    int height() const override;
    double fps() const override;
    void close() override;

    // Codec ID from the video stream (AV_CODEC_ID_H264 or AV_CODEC_ID_HEVC).
    // Only valid after open() returns true.
    int codecId() const;

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVPacket* av_packet_ = nullptr;
    int video_stream_idx_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int codec_id_ = 0;  // AVCodecID value
};
