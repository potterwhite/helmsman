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
// ffmpeg-input-source.cpp — FFmpeg demuxer with Annex B bitstream filter
//
// MP4 containers store H.264/H.265 in AVCC/HVCC format (length-prefixed NAL
// units). Hardware decoders (MPP) expect Annex B format (start-code-prefixed).
// This module uses FFmpeg's h264/hevc_mp4toannexb bitstream filter to convert
// the stream on the fly.
//
// Flow:
//   open()  → avformat_open_input + find stream + init BSF
//   ReadRaw() → av_read_frame → av_bsf_send_packet → av_bsf_receive_packet
//   close() → av_bsf_free + avformat_close_input
//
// =============================================================================

#include "pipeline/stages/frontend/stages/01-input-source/ffmpeg-input-source.h"

#include <cstdio>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

FfmpegInputSource::FfmpegInputSource() = default;

FfmpegInputSource::~FfmpegInputSource() {
    close();
}

FfmpegInputSource::FfmpegInputSource(FfmpegInputSource&& other) noexcept
    : fmt_ctx_(other.fmt_ctx_),
      av_packet_(other.av_packet_),
      bsf_ctx_(other.bsf_ctx_),
      video_stream_idx_(other.video_stream_idx_),
      width_(other.width_),
      height_(other.height_),
      fps_(other.fps_),
      codec_id_(other.codec_id_),
      bsf_packet_(other.bsf_packet_),
      merged_buf_(std::move(other.merged_buf_)) {
    other.fmt_ctx_ = nullptr;
    other.av_packet_ = nullptr;
    other.bsf_ctx_ = nullptr;
    other.bsf_packet_ = nullptr;
    other.video_stream_idx_ = -1;
    other.bsf_eof_ = false;
}

FfmpegInputSource& FfmpegInputSource::operator=(FfmpegInputSource&& other) noexcept {
    if (this != &other) {
        close();
        fmt_ctx_ = other.fmt_ctx_;
        av_packet_ = other.av_packet_;
        bsf_ctx_ = other.bsf_ctx_;
        video_stream_idx_ = other.video_stream_idx_;
        width_ = other.width_;
        height_ = other.height_;
        fps_ = other.fps_;
        codec_id_ = other.codec_id_;
        bsf_packet_ = other.bsf_packet_;
        merged_buf_ = std::move(other.merged_buf_);
        other.fmt_ctx_ = nullptr;
        other.av_packet_ = nullptr;
        other.bsf_ctx_ = nullptr;
        other.bsf_packet_ = nullptr;
        other.video_stream_idx_ = -1;
        other.bsf_eof_ = false;
    }
    return *this;
}

bool FfmpegInputSource::open(const std::string& uri) {
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
    codec_id_ = static_cast<int>(stream->codecpar->codec_id);

    // --- Initialize bitstream filter (AVCC/HVCC → Annex B) ---
    const char* bsf_name = nullptr;
    if (stream->codecpar->codec_id == AV_CODEC_ID_H264) {
        bsf_name = "h264_mp4toannexb";
    } else if (stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
        bsf_name = "hevc_mp4toannexb";
    } else {
        fprintf(stderr, "[FFmpegInputSource] unsupported codec %d\n",
                stream->codecpar->codec_id);
        close();
        return false;
    }

    const AVBitStreamFilter* bsf_filter = av_bsf_get_by_name(bsf_name);
    if (!bsf_filter) {
        fprintf(stderr, "[FFmpegInputSource] BSF '%s' not found\n", bsf_name);
        close();
        return false;
    }

    ret = av_bsf_alloc(bsf_filter, &bsf_ctx_);
    if (ret < 0) {
        fprintf(stderr, "[FFmpegInputSource] av_bsf_alloc failed\n");
        close();
        return false;
    }

    ret = avcodec_parameters_copy(bsf_ctx_->par_in, stream->codecpar);
    if (ret < 0) {
        fprintf(stderr, "[FFmpegInputSource] avcodec_parameters_copy failed\n");
        close();
        return false;
    }
    bsf_ctx_->time_base_in = stream->time_base;

    ret = av_bsf_init(bsf_ctx_);
    if (ret < 0) {
        fprintf(stderr, "[FFmpegInputSource] av_bsf_init failed\n");
        close();
        return false;
    }

    bsf_packet_ = av_packet_alloc();
    if (!bsf_packet_) {
        fprintf(stderr, "[FFmpegInputSource] av_packet_alloc (bsf) failed\n");
        close();
        return false;
    }

    // --- Allocate read packet ---
    av_packet_ = av_packet_alloc();
    if (!av_packet_) {
        fprintf(stderr, "[FFmpegInputSource] av_packet_alloc failed\n");
        close();
        return false;
    }

    if (stream->avg_frame_rate.den > 0 && stream->avg_frame_rate.num > 0) {
        fps_ = static_cast<double>(stream->avg_frame_rate.num) /
               static_cast<double>(stream->avg_frame_rate.den);
    } else if (stream->r_frame_rate.den > 0 && stream->r_frame_rate.num > 0) {
        fps_ = static_cast<double>(stream->r_frame_rate.num) /
               static_cast<double>(stream->r_frame_rate.den);
    } else {
        fps_ = 30.0;
    }

    fprintf(stderr, "[FFmpegInputSource] opened: %dx%d @ %.2f fps, BSF=%s\n",
            width_, height_, fps_, bsf_name);
    return true;
}

bool FfmpegInputSource::ReadRaw(RawPacket& pkt) {
    if (!fmt_ctx_ || !av_packet_ || !bsf_ctx_) {
        pkt = {};
        pkt.is_eof = true;
        return false;
    }

    // Already flushed — no more data.
    if (bsf_eof_) {
        pkt = {};
        pkt.is_eof = true;
        return false;
    }

    // Read next compressed packet from the container and convert via BSF.
    // All BSF output NAL units from one input packet are merged into a single
    // contiguous buffer, so MPP receives complete [SPS+PPS+frame] data.
    while (true) {
        int ret = av_read_frame(fmt_ctx_, av_packet_);
        if (ret < 0) {
            // EOF — flush the BSF to drain any remaining buffered packets.
            av_bsf_send_packet(bsf_ctx_, nullptr);
            merged_buf_.clear();
            while (av_bsf_receive_packet(bsf_ctx_, bsf_packet_) == 0) {
                merged_buf_.insert(merged_buf_.end(), bsf_packet_->data,
                                   bsf_packet_->data + bsf_packet_->size);
                av_packet_unref(bsf_packet_);
            }
            bsf_eof_ = true;
            if (!merged_buf_.empty()) {
                pkt.data = merged_buf_.data();
                pkt.size = merged_buf_.size();
                pkt.pts = 0;
                pkt.is_eof = false;
                return true;
            }
            pkt = {};
            pkt.is_eof = true;
            return false;
        }

        if (av_packet_->stream_index != video_stream_idx_) {
            av_packet_unref(av_packet_);
            continue;
        }

        // Feed the AVCC packet into the BSF.
        ret = av_bsf_send_packet(bsf_ctx_, av_packet_);
        if (ret < 0) {
            char err_buf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, err_buf, sizeof(err_buf));
            fprintf(stderr, "[FFmpegInputSource] av_bsf_send_packet failed: %s\n", err_buf);
            av_packet_unref(av_packet_);
            pkt = {};
            pkt.is_eof = true;
            return false;
        }

        // Merge all Annex B output NAL units into one contiguous buffer.
        merged_buf_.clear();
        while ((ret = av_bsf_receive_packet(bsf_ctx_, bsf_packet_)) == 0) {
            merged_buf_.insert(merged_buf_.end(), bsf_packet_->data,
                               bsf_packet_->data + bsf_packet_->size);
            av_packet_unref(bsf_packet_);
        }

        if (merged_buf_.empty()) {
            av_packet_unref(av_packet_);
            continue;
        }

        pkt.data = merged_buf_.data();
        pkt.size = merged_buf_.size();
        pkt.pts = av_packet_->pts;
        pkt.is_eof = false;
        av_packet_unref(av_packet_);
        return true;
    }
}

int FfmpegInputSource::width() const { return width_; }
int FfmpegInputSource::height() const { return height_; }
double FfmpegInputSource::fps() const { return fps_; }
int FfmpegInputSource::CodecId() const { return codec_id_; }

void FfmpegInputSource::close() {
    merged_buf_.clear();
    bsf_eof_ = false;
    if (bsf_packet_) {
        av_packet_free(&bsf_packet_);
        bsf_packet_ = nullptr;
    }
    if (bsf_ctx_) {
        av_bsf_free(&bsf_ctx_);
        bsf_ctx_ = nullptr;
    }
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
    codec_id_ = 0;
}
