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
// dma_buffer.cpp — DMA buffer allocator implementation
//
// Uses the Linux DMA-Heap API (/dev/dma_heap/system):
//   1. Open /dev/dma_heap/system
//   2. ioctl(DMA_HEAP_IOCTL_ALLOC) → returns a dmabuf fd
//   3. mmap() on demand for CPU access
//
// The fd can be passed directly to RGA (wrapbuffer_fd) or MPP (mpp_buffer).
// Different processes can share the same physical memory via fd passing.
// =============================================================================

#include "DmaKit/dma_buffer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace arcforge {
namespace dmakit {

static constexpr const char* kDmaHeapPath = "/dev/dma_heap/system";

DmaBuffer::DmaBuffer(int fd, size_t size) : fd_(fd), size_(size) {}

DmaBuffer::~DmaBuffer() {
    unmap();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

DmaBuffer::DmaBuffer(DmaBuffer&& other) noexcept
    : fd_(other.fd_), mapped_(other.mapped_), size_(other.size_) {
    other.fd_ = -1;
    other.mapped_ = nullptr;
    other.size_ = 0;
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer&& other) noexcept {
    if (this != &other) {
        unmap();
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        mapped_ = other.mapped_;
        size_ = other.size_;
        other.fd_ = -1;
        other.mapped_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

std::unique_ptr<DmaBuffer> DmaBuffer::Allocate(size_t bytes) {
    // 1. Open the DMA heap device.
    int heap_fd = ::open(kDmaHeapPath, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        fprintf(stderr, "[DmaKit] Failed to open %s: %s\n",
                kDmaHeapPath, strerror(errno));
        return nullptr;
    }

    // 2. Allocate via ioctl.
    struct dma_heap_allocation_data alloc = {};
    alloc.len = bytes;
    alloc.fd_flags = O_RDWR | O_CLOEXEC;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        fprintf(stderr, "[DmaKit] DMA_HEAP_IOCTL_ALLOC failed (len=%zu): %s\n",
                bytes, strerror(errno));
        ::close(heap_fd);
        return nullptr;
    }

    // alloc.fd is the dmabuf file descriptor.
    ::close(heap_fd);

    return std::unique_ptr<DmaBuffer>(new DmaBuffer(static_cast<int>(alloc.fd), bytes));
}

void* DmaBuffer::map() {
    if (mapped_) return mapped_;

    if (fd_ < 0) {
        fprintf(stderr, "[DmaBuffer] map() called on invalid fd\n");
        return nullptr;
    }

    mapped_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        fprintf(stderr, "[DmaBuffer] mmap failed (fd=%d, size=%zu): %s\n",
                fd_, size_, strerror(errno));
        mapped_ = nullptr;
        return nullptr;
    }

    return mapped_;
}

void DmaBuffer::unmap() {
    if (mapped_) {
        ::munmap(mapped_, size_);
        mapped_ = nullptr;
    }
}

}  // namespace dmakit
}  // namespace arcforge
