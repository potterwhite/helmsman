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
// DmaKit/dma_buffer.h — DMA buffer allocator for zero-copy pipelines
//
// Allocates DMA buffers via /dev/dma_heap/system (Linux DMA-Heap API).
// These buffers can be shared between processes via fd passing (SCM_RIGHTS)
// and used directly by hardware accelerators (RGA, MPP, GPU) without copying.
//
// Usage:
//   auto buf = DmaBuffer::Allocate(1920 * 1080 * 3);  // BGR 1080p
//   if (!buf) { /* handle error */ }
//   int fd = buf->fd();          // for RGA wrapbuffer_fd() or IPC
//   void* ptr = buf->map();      // for CPU read/write
//   // ... use ptr to write pixel data ...
//   buf->unmap();                // optional, destructor also unmaps
//
// Lifecycle:
//   Allocate() → fd valid immediately, map() on demand → ~DmaBuffer cleans up
//
// Thread safety: Each DmaBuffer instance is NOT thread-safe (same as cv::Mat).
// Different threads should use different DmaBuffer instances.
//
// =============================================================================

#pragma once

#include "DmaKit/pch.h"

namespace helmsman {
namespace dmakit {

class DmaBuffer {
public:
    // Allocate a DMA buffer of `bytes` size from /dev/dma_heap/system.
    // Returns nullptr on failure (fd open failed, ioctl failed, etc.).
    static std::unique_ptr<DmaBuffer> Allocate(size_t bytes);

    ~DmaBuffer();

    // Non-copyable (owns fd + potentially mmap'd memory).
    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    // Movable.
    DmaBuffer(DmaBuffer&& other) noexcept;
    DmaBuffer& operator=(DmaBuffer&& other) noexcept;

    // File descriptor — pass to RGA (wrapbuffer_fd) or send via SCM_RIGHTS.
    // Valid from construction until destruction.
    int fd() const { return fd_; }

    // Map the buffer into virtual address space for CPU access.
    // Returns nullptr on failure. Idempotent (second call returns same ptr).
    void* map();

    // Unmap the buffer. Safe to call multiple times. Also called by destructor.
    void unmap();

    // Buffer size in bytes.
    size_t size() const { return size_; }

    // Whether the buffer is currently mapped.
    bool is_mapped() const { return mapped_ != nullptr; }

private:
    DmaBuffer(int fd, size_t size);

    int fd_ = -1;
    void* mapped_ = nullptr;
    size_t size_ = 0;
};

}  // namespace dmakit
}  // namespace helmsman
