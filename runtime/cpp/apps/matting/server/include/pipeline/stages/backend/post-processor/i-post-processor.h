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

#include <opencv2/opencv.hpp>

/**
 * @brief Contract for any alpha-mask post-processor.
 *
 * A post-processor takes:
 *   - alpha_f32 : CV_32FC1 alpha mask at original resolution, values in [0, 1]
 *   - guide_bgr : CV_8UC3 original BGR image at the same resolution
 *
 * It returns a refined CV_32FC1 alpha mask, also in [0, 1].
 *
 * Attach/detach processors via MattingBackend::setPostProcessor().
 * If no processor is attached the raw alpha is used as-is.
 */
class IPostProcessor {
   public:
    virtual ~IPostProcessor() = default;

    /**
     * @param alpha_f32  Input alpha mask  (CV_32FC1, [0,1], original resolution)
     * @param guide_bgr  Original BGR image (CV_8UC3, same size as alpha_f32)
     * @return           Refined alpha mask (CV_32FC1, [0,1], same size)
     */
    virtual cv::Mat process(const cv::Mat& alpha_f32, const cv::Mat& guide_bgr) const = 0;
};
