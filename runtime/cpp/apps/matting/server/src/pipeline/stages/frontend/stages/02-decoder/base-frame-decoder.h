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
// base-frame-decoder.h — Abstract frame decoder interface (internal to Frontend)
//
// A FrameDecoder takes compressed bitstream data and produces decoded frames.
//
// =============================================================================

#pragma once

// Hardware frame output — a DMA buffer fd for zero-copy downstream.
struct HardwareFrame {
    int fd = -1;        // DMA buffer file descriptor
    int width = 0;      // Frame width in pixels
    int height = 0;     // Frame height in pixels
    int format = 0;     // Pixel format (implementation-specific)
};

// Abstract frame decoder interface (internal — do not use directly).
class BaseFrameDecoder {
public:
    virtual ~BaseFrameDecoder() = default;

    // Decode a compressed packet into a hardware frame (DMA buffer fd).
    // Returns true if a decoded frame is available.
    // The fd is valid until the next decode() call or destruction.
    virtual bool decode(const uint8_t* data, size_t size, HardwareFrame& out) = 0;
};
