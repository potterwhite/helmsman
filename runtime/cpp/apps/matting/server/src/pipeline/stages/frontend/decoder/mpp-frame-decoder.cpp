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
// mpp-frame-decoder.cpp — RK3588 VPU hardware frame decoder (internal)
//
// =============================================================================

#include "pipeline/stages/frontend/decoder/mpp-frame-decoder.h"

#include <cstdio>
#include "MPPKit/mpp_decoder.h"

_MppFrameDecoder::_MppFrameDecoder(arcforge::mppkit::DecoderConfig config)
    : config_(std::move(config)) {}

_MppFrameDecoder::~_MppFrameDecoder() = default;

_MppFrameDecoder::_MppFrameDecoder(_MppFrameDecoder&&) noexcept = default;
_MppFrameDecoder& _MppFrameDecoder::operator=(_MppFrameDecoder&&) noexcept = default;

bool _MppFrameDecoder::init() {
    if (decoder_) {
        fprintf(stderr, "[MppFrameDecoder] already initialized\n");
        return false;
    }

    decoder_ = std::make_unique<arcforge::mppkit::MppDecoder>(config_);
    if (!decoder_->Init()) {
        fprintf(stderr, "[MppFrameDecoder] MppDecoder::Init failed\n");
        decoder_.reset();
        return false;
    }

    return true;
}

bool _MppFrameDecoder::decode(const uint8_t* data, size_t size,
                              HardwareFrame& out) {
    if (!decoder_ || !decoder_->IsOpen()) {
        out = {};
        return false;
    }

    arcforge::mppkit::DecodedFrame decoded;
    if (!decoder_->DecodeNextFrame(data, size, decoded)) {
        out = {};
        return false;
    }

    out.fd = decoded.fd;
    out.width = decoded.width;
    out.height = decoded.height;
    out.format = decoded.format;
    return true;
}
