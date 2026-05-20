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
// frontend.cpp — Frontend router for the matting pipeline
//
// Two paths:
//   MPP path:    _IInputSource::readRaw → _IFrameDecoder::decode → NV12 fd
//                → RGA NV12→BGR → cpu_frame (cv::Mat)
//   OpenCV path: cv::VideoCapture::read → cv::Mat (software decode)
//
// =============================================================================

#include "pipeline/stages/frontend/frontend.h"
#include "pipeline/stages/frontend/decoder/mpp-frame-decoder.h"
#include "pipeline/stages/frontend/input-source/ffmpeg-input-source.h"
#include "MPPKit/mpp_codec.h"

#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------------------
// MPP hardware path constructor
// ---------------------------------------------------------------------------
Frontend::Frontend(std::unique_ptr<_IInputSource> source,
                   std::unique_ptr<_IFrameDecoder> decoder)
    : use_hardware_(true),
      source_(std::move(source)),
      decoder_(std::move(decoder)) {
    // Create RGA NV12→BGR converter for hardware path
    nv12_to_bgr_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaCvtColor>(
        helmsman::rgakit::RgaPixelFormat::kNv12,
        helmsman::rgakit::RgaPixelFormat::kBgr888,
        helmsman::rgakit::RgaCscMode::kYuvToRgbBt601Limit);
}

// ---------------------------------------------------------------------------
// OpenCV shortcut path constructor
// ---------------------------------------------------------------------------
Frontend::Frontend(const std::string& video_path)
    : use_hardware_(false) {
    if (!cv_cap_.open(video_path)) {
        fprintf(stderr, "[Frontend] failed to open video: %s\n",
                video_path.c_str());
    }
}

// ---------------------------------------------------------------------------
// Config-based constructor — creates internal source + decoder
// ---------------------------------------------------------------------------
Frontend::Frontend(const std::string& input_path, bool use_hardware_decoder) {
    if (use_hardware_decoder) {
        use_hardware_ = true;

        // Create and open FFmpeg input source
        auto source = std::make_unique<_FFmpegInputSource>();
        if (!source->open(input_path)) {
            throw std::runtime_error("Failed to open video with FFmpeg: " + input_path);
        }

        // Detect codec from stream
        helmsman::mppkit::CodecType codec = helmsman::mppkit::CodecType::kH264;
        if (source->codecId() == 173) {
            codec = helmsman::mppkit::CodecType::kH265;
        }

        // Create and init MPP hardware decoder
        helmsman::mppkit::DecoderConfig decoder_cfg;
        decoder_cfg.codec_type = codec;
        decoder_cfg.width = source->width();
        decoder_cfg.height = source->height();

        auto decoder = std::make_unique<_MppFrameDecoder>(decoder_cfg);
        if (!decoder->init()) {
            throw std::runtime_error("Failed to init MPP decoder");
        }

        source_ = std::move(source);
        decoder_ = std::move(decoder);

        // Create RGA NV12→BGR converter
        nv12_to_bgr_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaCvtColor>(
            helmsman::rgakit::RgaPixelFormat::kNv12,
            helmsman::rgakit::RgaPixelFormat::kBgr888,
            helmsman::rgakit::RgaCscMode::kYuvToRgbBt601Limit);
    } else {
        use_hardware_ = false;
        if (!cv_cap_.open(input_path)) {
            throw std::runtime_error("Failed to open video: " + input_path);
        }
    }
}

Frontend::~Frontend() = default;

// ---------------------------------------------------------------------------
// readFrame — read the next decoded frame
// ---------------------------------------------------------------------------
bool Frontend::readFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame) {
    cpu_frame.release();
    hw_frame = HardwareFrame{};

    if (use_hardware_) {
        if (!source_ || !decoder_) {
            return false;
        }

        RawPacket pkt;
        if (!source_->readRaw(pkt) || pkt.is_eof) {
            return false;
        }

        if (!decoder_->decode(pkt.data, pkt.size, hw_frame)) {
            return false;
        }

        // Convert NV12 → BGR via RGA hardware
        if (hw_frame.fd >= 0 && nv12_to_bgr_) {
            // Allocate BGR buffer on first frame (or dimension change)
            if (bgr_buf_.empty() ||
                bgr_buf_.cols != hw_frame.width ||
                bgr_buf_.rows != hw_frame.height) {
                bgr_buf_ = cv::Mat(hw_frame.height, hw_frame.width, CV_8UC3);
            }

            auto src = helmsman::rgakit::ImageDescriptor::FromFd(
                hw_frame.fd, hw_frame.width, hw_frame.height,
                helmsman::rgakit::RgaPixelFormat::kNv12);
            auto dst = helmsman::rgakit::ImageDescriptor(
                bgr_buf_.data, bgr_buf_.cols, bgr_buf_.rows,
                helmsman::rgakit::RgaPixelFormat::kBgr888);

            if (nv12_to_bgr_->Execute(src, dst)) {
                cpu_frame = bgr_buf_;
                return true;
            }
            fprintf(stderr, "[Frontend] RGA NV12→BGR conversion failed\n");
            return false;
        }
        return false;
    }

    if (!cv_cap_.isOpened()) {
        return false;
    }

    return cv_cap_.read(cpu_frame);
}

// ---------------------------------------------------------------------------
// preprocess — convert a BGR frame into TensorData
// ---------------------------------------------------------------------------
TensorData Frontend::preprocess(const cv::Mat& frame,
                                size_t model_w, size_t model_h) {
    return preprocessor_.preprocess(frame, model_w, model_h);
}

// ---------------------------------------------------------------------------
// Source properties
// ---------------------------------------------------------------------------
int Frontend::width() const {
    if (use_hardware_ && source_) {
        return source_->width();
    }
    if (cv_cap_.isOpened()) {
        return static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    }
    return 0;
}

int Frontend::height() const {
    if (use_hardware_ && source_) {
        return source_->height();
    }
    if (cv_cap_.isOpened()) {
        return static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    }
    return 0;
}

double Frontend::fps() const {
    if (use_hardware_ && source_) {
        return source_->fps();
    }
    if (cv_cap_.isOpened()) {
        return cv_cap_.get(cv::CAP_PROP_FPS);
    }
    return 0.0;
}
