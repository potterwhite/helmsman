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
// ffmpeg-input-source.cpp — FFmpeg demuxer (Phase 2, internal)
//
// Flow:
//   open() → avformat_open_input + avformat_find_stream_info
//   readRaw() → av_read_frame → extract video packet → RawPacket
//   close() → avformat_close_input
//
// =============================================================================

#include "pipeline/stages/frontend/input-source/ffmpeg-input-source.h"

#include <cstdio>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

_FFmpegInputSource::_FFmpegInputSource() = default;

_FFmpegInputSource::~_FFmpegInputSource() {
    close();
}

_FFmpegInputSource::_FFmpegInputSource(_FFmpegInputSource&& other) noexcept
    : fmt_ctx_(other.fmt_ctx_),
      av_packet_(other.av_packet_),
      video_stream_idx_(other.video_stream_idx_),
      width_(other.width_),
      height_(other.height_),
      fps_(other.fps_) {
    other.fmt_ctx_ = nullptr;
    other.av_packet_ = nullptr;
    other.video_stream_idx_ = -1;
}

_FFmpegInputSource& _FFmpegInputSource::operator=(_FFmpegInputSource&& other) noexcept {
    if (this != &other) {
        close();
        fmt_ctx_ = other.fmt_ctx_;
        av_packet_ = other.av_packet_;
        video_stream_idx_ = other.video_stream_idx_;
        width_ = other.width_;
        height_ = other.height_;
        fps_ = other.fps_;
        other.fmt_ctx_ = nullptr;
        other.av_packet_ = nullptr;
        other.video_stream_idx_ = -1;
    }
    return *this;
}

bool _FFmpegInputSource::open(const std::string& uri) {
    if (fmt_ctx_) {
        fprintf(stderr, "[FFmpegInputSource] already open\n");
        return false;
    }

    int ret = avformat_open_input(&fmt_ctx_, uri.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char err_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, err_buf, sizeof(err_buf));
        fprintf(stderr, "[FFmpegInputSource] avformat_open_input failed: %s\n", err_buf);
        fmt_ctx_ = nullptr;
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[FFmpegInputSource] avformat_find_stream_info failed\n");
        close();
        return false;
    }

    video_stream_idx_ = -1;
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = static_cast<int>(i);
            break;
        }
    }

    if (video_stream_idx_ < 0) {
        fprintf(stderr, "[FFmpegInputSource] no video stream found\n");
        close();
        return false;
    }

    AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
    width_ = stream->codecpar->width;
    height_ = stream->codecpar->height;

    if (stream->avg_frame_rate.den > 0 && stream->avg_frame_rate.num > 0) {
        fps_ = static_cast<double>(stream->avg_frame_rate.num) /
               static_cast<double>(stream->avg_frame_rate.den);
    } else if (stream->r_frame_rate.den > 0 && stream->r_frame_rate.num > 0) {
        fps_ = static_cast<double>(stream->r_frame_rate.num) /
               static_cast<double>(stream->r_frame_rate.den);
    } else {
        fps_ = 30.0;
    }

    av_packet_ = av_packet_alloc();
    if (!av_packet_) {
        fprintf(stderr, "[FFmpegInputSource] av_packet_alloc failed\n");
        close();
        return false;
    }

    return true;
}

bool _FFmpegInputSource::readRaw(RawPacket& pkt) {
    if (!fmt_ctx_ || !av_packet_) {
        pkt = {};
        pkt.is_eof = true;
        return false;
    }

    pkt = {};

    while (true) {
        int ret = av_read_frame(fmt_ctx_, av_packet_);
        if (ret < 0) {
            pkt.is_eof = true;
            return false;
        }

        if (av_packet_->stream_index != video_stream_idx_) {
            av_packet_unref(av_packet_);
            continue;
        }

        pkt.data = av_packet_->data;
        pkt.size = static_cast<size_t>(av_packet_->size);
        pkt.pts = av_packet_->pts;
        pkt.is_eof = false;
        return true;
    }
}

int _FFmpegInputSource::width() const { return width_; }
int _FFmpegInputSource::height() const { return height_; }
double _FFmpegInputSource::fps() const { return fps_; }

void _FFmpegInputSource::close() {
    if (av_packet_) {
        av_packet_free(&av_packet_);
        av_packet_ = nullptr;
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_idx_ = -1;
    width_ = 0;
    height_ = 0;
    fps_ = 0.0;
}
