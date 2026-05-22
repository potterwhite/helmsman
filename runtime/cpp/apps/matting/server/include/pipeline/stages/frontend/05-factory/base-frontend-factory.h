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
// base-frontend-factory.h — Frontend factory interface
//
// Each SoC vendor provides its own implementation (e.g., RockchipFrontendFactory)
// that wires the appropriate input source, decoder, and color converter.
//
// =============================================================================

#pragma once

#include <memory>
#include <string>

class Frontend;

class BaseFrontendFactory {
public:
    virtual ~BaseFrontendFactory() = default;

    // Create a hardware-decode Frontend for the given input path.
    // Throws std::runtime_error on failure.
    virtual std::unique_ptr<Frontend> create(const std::string& input_path) = 0;
};

// Creates a Frontend for the given input path.
// If use_hardware is true, uses the platform-specific hardware decode factory
// selected at build time. Otherwise, uses OpenCV software decode.
// Throws std::runtime_error on failure.
std::unique_ptr<Frontend> create_frontend(const std::string& input_path,
                                          bool use_hardware);
