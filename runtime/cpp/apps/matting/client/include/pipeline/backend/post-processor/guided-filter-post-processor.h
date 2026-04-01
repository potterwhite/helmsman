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

#include "pipeline/backend/post-processor/i-post-processor.h"

/**
 * @brief Guided Filter post-processor.
 *
 * Uses the original high-resolution image as a guide to snap the blurry
 * AI alpha mask edges onto real pixel boundaries (hair-level detail).
 *
 * Algorithm: cv::ximgproc::guidedFilter
 *   guide = grayscale original image (physical edge information)
 *   src   = coarse alpha mask from NPU inference (upscaled, blurry)
 *
 * Tuning:
 *   radius  — larger = more smoothing, smaller = sharper edges
 *   epsilon — larger = more averaging, smaller = more edge-preserving
 *
 * Typical starting point: radius=16, epsilon=1e-4
 */
class GuidedFilterPostProcessor : public IPostProcessor {
   public:
    /**
     * @param radius   Filter radius in pixels (default: 16)
     * @param epsilon  Regularisation coefficient (default: 1e-4)
     */
    explicit GuidedFilterPostProcessor(int radius = 16, double epsilon = 1e-4);
    ~GuidedFilterPostProcessor() override = default;

    cv::Mat process(const cv::Mat& alpha_f32, const cv::Mat& guide_bgr) const override;

   private:
    int    radius_;
    double epsilon_;
};
