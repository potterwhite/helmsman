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

#include "RKNNKit/rknn-memory.h"
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace helmsman::rknnkit {

std::vector<rknn_tensor_mem*> RKNNMemory::AllocAll(
    rknn_context ctx,
    const std::vector<rknn_tensor_attr>& attrs,
    std::initializer_list<uint32_t> non_cacheable_indices) {
	std::unordered_set<uint32_t> non_cache(non_cacheable_indices);
	std::vector<rknn_tensor_mem*> mems(attrs.size(), nullptr);

	for (uint32_t i = 0; i < attrs.size(); ++i) {
		rknn_mem_alloc_flags flag = (non_cache.count(i) > 0)
		                                ? RKNN_FLAG_MEMORY_NON_CACHEABLE
		                                : RKNN_FLAG_MEMORY_CACHEABLE;
		mems[i] = rknn_create_mem2(ctx, static_cast<uint32_t>(attrs[i].size), flag);
		if (!mems[i]) {
			// Release anything already allocated to avoid leaks on partial failure.
			for (uint32_t j = 0; j < i; ++j) {
				if (mems[j]) {
					rknn_destroy_mem(ctx, mems[j]);
				}
			}
			throw std::runtime_error("RKNNMemory::AllocAll: rknn_create_mem2 failed for index " +
			                         std::to_string(i) +
			                         " (size=" + std::to_string(attrs[i].size) + ")");
		}
	}
	return mems;
}

void RKNNMemory::FreeAll(rknn_context ctx, std::vector<rknn_tensor_mem*>& mems) {
	for (auto* mem : mems) {
		if (mem) {
			rknn_destroy_mem(ctx, mem);
		}
	}
	mems.clear();
}

int RKNNMemory::Sync(rknn_context ctx, rknn_tensor_mem* mem, rknn_mem_sync_mode mode) {
	if (!mem) {
		return -1;
	}
	return rknn_mem_sync(ctx, mem, mode);
}

}  // namespace helmsman::rknnkit
