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
// base-input-source.h — Abstract input source interface (internal to Frontend)
//
// An InputSource is responsible for reading raw compressed data from a signal
// source (mp4 file, camera, IPC stream). It does NOT decode — that is the
// FrameDecoder's job.
//
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// A raw compressed packet from the input source.
struct RawPacket {
    const uint8_t* data = nullptr;  // Compressed bitstream data (H.264/H.265 NAL)
    size_t size = 0;                // Data size in bytes
    int64_t pts = 0;                // Presentation timestamp (microseconds)
    bool is_eof = false;            // True if no more packets
};

// Abstract input source interface (internal — do not use directly).
class BaseInputSource {
public:
    virtual ~BaseInputSource() = default;

    // Open the source. Returns true on success.
    virtual bool open(const std::string& uri) = 0;

    // Read the next raw compressed packet.
    // Returns true if a packet is available (check pkt.is_eof for EOF).
    // The packet data pointer is valid until the next ReadRaw() or close() call.
    virtual bool ReadRaw(RawPacket& pkt) = 0;

    // Source properties (available after open()).
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;

    // Release resources.
    virtual void close() = 0;
};
