/*
 * Copyright (c) 2026 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

// ---------------------------------------------------------------------------
// Utils/simd/fp16-convert.h — SIMD-accelerated FP32 <-> FP16 conversion
//
// Provides batch conversion between float32 and float16 arrays using
// hardware SIMD instructions where available:
//   - ARM NEON (aarch64): FCVTN / FCVTN2 — 4 or 8 elements per instruction
//   - Scalar fallback:    for non-ARM platforms or tail elements
//
// Usage:
//   #include "Utils/simd/fp16-convert.h"
//
//   std::vector<float> src = ...;
//   __fp16* dst = reinterpret_cast<__fp16*>(dma_buffer);
//   helmsman::utils::simd::fp32_to_fp16(src.data(), dst, src.size());
//
//   // Reverse direction:
//   __fp16* out_fp16 = ...;
//   std::vector<float> out_fp32(count);
//   helmsman::utils::simd::fp16_to_fp32(out_fp16, out_fp32.data(), count);
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)
#define HELMSMAN_SIMD_NEON_FP16 1
#include <arm_neon.h>
#else
#define HELMSMAN_SIMD_NEON_FP16 0
#endif

namespace helmsman::utils::simd {

// ============================================================================
// fp32_to_fp16 — batch convert float32 → float16
//
// On aarch64 with NEON: processes 4 elements per iteration via FCVTN.
// On other platforms: scalar fallback with static_cast.
//
// src    — input float32 array
// dst    — output float16 array (must have room for count elements)
// count  — number of elements to convert
// ============================================================================
inline void fp32_to_fp16(const float* src, __fp16* dst, std::size_t count) {
	std::size_t i = 0;

#if HELMSMAN_SIMD_NEON_FP16
	// Process 4 floats at a time: vcvt_f16_f32 converts float32x4 → float16x4
	for (; i + 4 <= count; i += 4) {
		float32x4_t v = vld1q_f32(src + i);
		float16x4_t h = vcvt_f16_f32(v);
		vst1_f16(reinterpret_cast<float16_t*>(dst + i), h);
	}
#endif

	// Scalar tail (or full path on non-ARM)
	for (; i < count; ++i) {
		dst[i] = static_cast<__fp16>(src[i]);
	}
}

// ============================================================================
// fp16_to_fp32 — batch convert float16 → float32
//
// On aarch64 with NEON: processes 4 elements per iteration via FCVTL.
// On other platforms: scalar fallback with static_cast.
//
// src    — input float16 array
// dst    — output float32 array (must have room for count elements)
// count  — number of elements to convert
// ============================================================================
inline void fp16_to_fp32(const __fp16* src, float* dst, std::size_t count) {
	std::size_t i = 0;

#if HELMSMAN_SIMD_NEON_FP16
	// Process 4 fp16 at a time: vcvt_f32_f16 converts float16x4 → float32x4
	for (; i + 4 <= count; i += 4) {
		float16x4_t h = vld1_f16(reinterpret_cast<const float16_t*>(src + i));
		float32x4_t v = vcvt_f32_f16(h);
		vst1q_f32(dst + i, v);
	}
#endif

	// Scalar tail (or full path on non-ARM)
	for (; i < count; ++i) {
		dst[i] = static_cast<float>(src[i]);
	}
}

}  // namespace helmsman::utils::simd
