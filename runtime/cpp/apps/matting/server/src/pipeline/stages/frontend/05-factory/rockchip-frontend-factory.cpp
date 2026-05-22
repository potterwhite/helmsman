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
// rockchip-frontend-factory.cpp — Rockchip platform Frontend factory (internal)
//
// =============================================================================

#include "pipeline/stages/frontend/05-factory/rockchip-frontend-factory.h"
#include "pipeline/stages/frontend/frontend.h"
#include "pipeline/stages/frontend/01-input-source/ffmpeg-input-source.h"
#include "pipeline/stages/frontend/02-decoder/mpp-frame-decoder.h"
#include "pipeline/stages/frontend/03-color-convert/rga-nv12-to-bgr.h"
#include "MPPKit/mpp_codec.h"

#include <stdexcept>

std::unique_ptr<Frontend> RockchipFrontendFactory::create(const std::string& input_path) {
	// Create and open FFmpeg input source
	auto source = std::make_unique<FfmpegInputSource>();
	if (!source->open(input_path)) {
		throw std::runtime_error("Failed to open video with FFmpeg: " + input_path);
	}

	// Detect codec from stream
	helmsman::mppkit::CodecType codec = helmsman::mppkit::CodecType::kH264;
	if (source->CodecId() == 173) {
		codec = helmsman::mppkit::CodecType::kH265;
	}

	// Create and init hardware decoder
	helmsman::mppkit::DecoderConfig decoder_cfg;
	decoder_cfg.codec_type = codec;
	decoder_cfg.width = source->width();
	decoder_cfg.height = source->height();

	auto decoder = std::make_unique<MppFrameDecoder>(decoder_cfg);
	if (!decoder->init()) {
		throw std::runtime_error("Failed to init hardware decoder");
	}

	// Assemble Frontend with DI constructor
	return std::make_unique<Frontend>(
	    std::move(source), std::move(decoder),
	    std::make_unique<RgaNv12ToBgr>());
}

std::unique_ptr<Frontend> CreateFrontend(const std::string& input_path, bool use_hardware) {
	if (use_hardware) {
		RockchipFrontendFactory factory;
		return factory.create(input_path);
	}
	return std::make_unique<Frontend>(input_path);
}
