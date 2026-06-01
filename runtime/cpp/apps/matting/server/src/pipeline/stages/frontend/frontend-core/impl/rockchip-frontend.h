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
// rockchip-frontend.h — Rockchip hardware-decode Frontend subclass
//
// Uses FFmpegInputSource + MppFrameDecoder + RgaNv12ToBgr for hardware decode.
//
// =============================================================================

#pragma once

#include <memory>
#include "pipeline/stages/frontend/frontend-core/frontend-base.h"
#include "pipeline/stages/frontend/stages/01-input-source/base-input-source.h"
#include "pipeline/stages/frontend/stages/02-decoder/base-frame-decoder.h"
#include "pipeline/stages/frontend/stages/03-color-convert/base-color-converter.h"

class RockchipFrontend : public FrontendBase {
public:
    // Constructs the hardware decode pipeline: FFmpeg demux -> MPP decode -> RGA color convert.
    // Throws std::runtime_error on failure.
    explicit RockchipFrontend(const std::string& input_path, bool multithread_enabled = false);

protected:
    void _OpenSource(const std::string& input_path) override;

    // Stage 01: FFmpeg read raw packet
    bool _ReadInputSource01(RawPacket& pkt, ReadResult& result) override;

    // Stage 02: MPP hardware decode
    bool _DecodeFrame02(const RawPacket& pkt, ReadResult& result) override;

    // Stage 03: RGA NV12 → BGR color convert
    bool _ConvertToBgr03(ReadResult& result) override;

private:
    // Hardware decode components
    std::unique_ptr<BaseInputSource> source_;
    std::unique_ptr<BaseFrameDecoder> decoder_;
    std::unique_ptr<BaseColorConverter> color_converter_;
};
