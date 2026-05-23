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
// rockchip-frontend.cpp — Rockchip hardware-decode Frontend subclass
//
// =============================================================================

#include "pipeline/stages/frontend/00-base/rockchip-frontend.h"

#include "MPPKit/mpp_codec.h"
#include "pipeline/stages/frontend/01-input-source/ffmpeg-input-source.h"
#include "pipeline/stages/frontend/02-decoder/mpp-frame-decoder.h"
#include "pipeline/stages/frontend/03-color-convert/rga-nv12-to-bgr.h"

#include <cassert>
#include <stdexcept>

RockchipFrontend::RockchipFrontend(const std::string& input_path, bool use_pipeline)
    : FrontendBase(true),
      pipeline_([this]() -> std::optional<ReadResult> { return _ReadFrame(); },
                use_pipeline) {
    // Create and open FFmpeg input source (concrete type for CodecId access)
    auto ffmpeg_source = std::make_unique<FfmpegInputSource>();
    if (!ffmpeg_source->open(input_path)) {
        throw std::runtime_error("Failed to open video with FFmpeg: " + input_path);
    }

    // Detect codec from stream
    helmsman::mppkit::CodecType codec = helmsman::mppkit::CodecType::kH264;
    if (ffmpeg_source->CodecId() == 173) {
        codec = helmsman::mppkit::CodecType::kH265;
    }

    // Create and init hardware decoder (concrete type for init() access)
    helmsman::mppkit::DecoderConfig decoder_cfg;
    decoder_cfg.codec_type = codec;
    decoder_cfg.width = ffmpeg_source->width();
    decoder_cfg.height = ffmpeg_source->height();

    auto mpp_decoder = std::make_unique<MppFrameDecoder>(decoder_cfg);
    if (!mpp_decoder->init()) {
        throw std::runtime_error("Failed to init hardware decoder");
    }

    // Set base class source properties now that the source is open.
    SetSourceProperties(ffmpeg_source->width(), ffmpeg_source->height(), ffmpeg_source->fps());
    assert(width() > 0 && height() > 0);

    // Move to base class pointers (after all concrete-type operations are done)
    source_ = std::move(ffmpeg_source);
    decoder_ = std::move(mpp_decoder);
    color_converter_ = std::make_unique<RgaNv12ToBgr>();
}

std::optional<FrameResult> RockchipFrontend::ProcessOneFrame(int model_w, int model_h) {
    return pipeline_.ProcessOneFrame(model_w, model_h);
}

void RockchipFrontend::Stop() {
    pipeline_.Stop();
}

const helmsman::utils::timing::StageAccumulator& RockchipFrontend::preprocess_acc() const {
    return pipeline_.preprocess_acc();
}

std::optional<ReadResult> RockchipFrontend::_ReadFrame() {
    if (!source_ || !decoder_ || !color_converter_) {
        return std::nullopt;
    }

    // Feed packets until the decoder produces a frame or we hit EOF.
    // Hardware decoders may need multiple packets before the first
    // frame appears (e.g. SPS/PPS in H.264), so a single failed
    // decode does not mean end-of-stream.
    RawPacket pkt;
    ReadResult result;
    while (true) {
        if (!source_->ReadRaw(pkt) || pkt.is_eof) {
            return std::nullopt;
        }

        if (decoder_->decode(pkt.data, pkt.size, result.hw_frame)) {
            break;  // Got a decoded frame
        }
        // decode returned false — decoder needs more data, keep feeding
    }

    // Convert hardware frame to BGR via color converter
    if (!color_converter_->convert(result.hw_frame, result.frame)) {
        return std::nullopt;
    }

    return result;
}
