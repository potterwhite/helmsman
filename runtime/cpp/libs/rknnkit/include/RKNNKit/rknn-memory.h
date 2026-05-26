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

#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>
#include "rknn_api.h"

namespace helmsman::rknnkit {

// ============================================================================
// RKNNMemory — stateless utility class for RKNN zero-copy buffer management.
//
// **Default cacheability is CACHEABLE** (matches plain rknn_create_mem()
// behaviour). Selectively switch specific tensor indices to non-cacheable by
// passing them via `non_cacheable_indices`. Indices outside that set stay
// cacheable. Empty set ⇒ all cacheable, equivalent to the legacy path.
//
// Typical usage (RVM 5-input / 6-output model):
//
//     // All inputs cacheable except r1i~r4i (indices 1..4):
//     input_mems_ = RKNNMemory::AllocAll(ctx, input_attrs, {1, 2, 3, 4});
//
//     // All outputs cacheable except r1o~r4o (indices 2..5):
//     output_mems_ = RKNNMemory::AllocAll(ctx, output_attrs, {2, 3, 4, 5});
//
//     // Cleanup:
//     RKNNMemory::FreeAll(ctx, input_mems_);
//     RKNNMemory::FreeAll(ctx, output_mems_);
//
//     // Manual cache sync (only meaningful for cacheable buffers when
//     // RKNN_FLAG_DISABLE_FLUSH_*_MEM_CACHE is set at rknn_init):
//     RKNNMemory::Sync(ctx, input_mems_[0], RKNN_MEMORY_SYNC_TO_DEVICE);
//
// All methods are static. Class is non-instantiable.
// Thread safety: methods are stateless; ctx thread-safety is caller's duty.
// ============================================================================
class RKNNMemory {
   public:
	RKNNMemory() = delete;
	RKNNMemory(const RKNNMemory&) = delete;
	RKNNMemory& operator=(const RKNNMemory&) = delete;

	// Allocate one zero-copy DMA buffer per tensor attr.
	// Indices listed in `non_cacheable_indices` use RKNN_FLAG_MEMORY_NON_CACHEABLE;
	// all other indices use RKNN_FLAG_MEMORY_CACHEABLE (the default).
	// Throws std::runtime_error on allocation failure.
	static std::vector<rknn_tensor_mem*> AllocAll(
	    rknn_context ctx,
	    const std::vector<rknn_tensor_attr>& attrs,
	    std::initializer_list<uint32_t> non_cacheable_indices = {});

	// Release every non-null mem in the vector via rknn_destroy_mem() and
	// clear the vector. Safe to call on an already-empty vector.
	static void FreeAll(rknn_context ctx, std::vector<rknn_tensor_mem*>& mems);

	// Thin wrapper over rknn_mem_sync. Only meaningful when the buffer is
	// CACHEABLE and the runtime's auto flush/invalidate was disabled via
	// RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE / _OUTPUT_MEM_CACHE.
	// No-op-safe to call on non-cacheable buffers (SDK ignores).
	static int Sync(rknn_context ctx, rknn_tensor_mem* mem, rknn_mem_sync_mode mode);

   private:
	~RKNNMemory() = delete;
};

}  // namespace helmsman::rknnkit
