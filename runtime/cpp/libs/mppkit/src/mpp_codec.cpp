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
// mpp_codec.cpp — Factory functions for creating MPP codec instances
//
// This file implements the CreateEncoder() and CreateDecoder() factory
// functions. Each function instantiates the appropriate concrete class
// (MppEncoder or MppDecoder) based on the CodecType in the config.
//
// Adding a new codec type:
//   1. Add kNewCodec to the CodecType enum in mpp_codec.h
//   2. Create MppNewEncoder / MppNewDecoder subclass
//   3. Add a case in the switch below
//
// =============================================================================

#include "MPPKit/mpp_codec.h"
#include "MPPKit/mpp_decoder.h"
#include "MPPKit/mpp_encoder.h"

#include <cstdio>

namespace helmsman {
namespace mppkit {

std::unique_ptr<MppCodec> CreateEncoder(EncoderConfig config) {
    // Validate config
    if (config.width <= 0 || config.height <= 0) {
        fprintf(stderr, "[MPPKit] CreateEncoder: invalid dimensions %dx%d\n",
                config.width, config.height);
        return nullptr;
    }
    if (config.output_path.empty()) {
        fprintf(stderr, "[MPPKit] CreateEncoder: output_path is empty\n");
        return nullptr;
    }

    // Dispatch based on codec type.
    // Currently only H.264 and H.265 are supported.
    // To add AV1: create MppAv1Encoder and add a case here.
    switch (config.codec_type) {
        case CodecType::kH264:
        case CodecType::kH265: {
            // Both H.264 and H.265 share the same MppEncoder class.
            // The codec type is passed through the config and used during
            // mpp_init() to select the correct hardware codec.
            auto encoder = std::make_unique<MppEncoder>(std::move(config));
            return encoder;
        }
        default:
            fprintf(stderr, "[MPPKit] CreateEncoder: unsupported codec type\n");
            return nullptr;
    }
}

std::unique_ptr<MppCodec> CreateDecoder(DecoderConfig config) {
    if (config.width <= 0 || config.height <= 0) {
        fprintf(stderr, "[MPPKit] CreateDecoder: invalid dimensions %dx%d\n",
                config.width, config.height);
        return nullptr;
    }

    switch (config.codec_type) {
        case CodecType::kH264:
        case CodecType::kH265: {
            auto decoder = std::make_unique<MppDecoder>(std::move(config));
            return decoder;
        }
        default:
            fprintf(stderr, "[MPPKit] CreateDecoder: unsupported codec type\n");
            return nullptr;
    }
}

}  // namespace mppkit
}  // namespace helmsman
