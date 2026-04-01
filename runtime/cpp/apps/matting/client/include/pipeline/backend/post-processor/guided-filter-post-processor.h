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
 * Self-contained implementation of He et al. Guided Image Filtering (TPAMI 2013)
 * using only opencv_core + opencv_imgproc (boxFilter). No contrib/ximgproc needed.
 *
 * Pipeline:
 *   1. (optional) threshold alpha_f32 to harden soft edges before GF.
 *      This is critical when the AI mask has a wide soft transition band
 *      (e.g. 30-50px after upscaling from 512×512 → 1080P).
 *      threshold=0 disables this step.
 *   2. Run Guided Filter: snaps edges to physical pixel boundaries in guide.
 *
 * Tuning:
 *   radius    — search radius; larger = wider snap range, more halo
 *   epsilon   — regularisation; keep small (1e-4~1e-6) for edge-preserving
 *   threshold — pre-GF binarisation cutoff in [0,1]; 0 = disabled
 */
class GuidedFilterPostProcessor : public IPostProcessor {
   public:
    /**
     * @param radius     Filter radius in pixels (default: 4)
     * @param epsilon    Regularisation coefficient (default: 1e-4)
     * @param threshold  Pre-GF hard threshold in [0,1]; 0.0 = disabled (default: 0.0)
     */
    explicit GuidedFilterPostProcessor(int radius = 4, double epsilon = 1e-4,
                                       float threshold = 0.0f);
    ~GuidedFilterPostProcessor() override = default;

    cv::Mat process(const cv::Mat& alpha_f32, const cv::Mat& guide_bgr) const override;

   private:
    int    radius_;
    double epsilon_;
    float  threshold_;
};
