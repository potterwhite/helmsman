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

#include "pipeline/backend/post-processor/guided-filter-post-processor.h"

#include <stdexcept>
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------------------------
// Guided Image Filtering — self-contained implementation.
//
// Algorithm: He et al., "Guided Image Filtering", IEEE TPAMI 2013.
// Uses only opencv_core + opencv_imgproc (no contrib / ximgproc needed).
//
// q_i = a_k * I_i + b_k   for all i in window w_k
// where a_k, b_k are computed per window to minimise
//   E(a_k, b_k) = sum[(a_k*I_i + b_k - p_i)^2 + eps*a_k^2]
//
// guide (I): grayscale original image, float32 [0,1]
// src   (p): coarse alpha mask,        float32 [0,1]
// ---------------------------------------------------------------------------
static cv::Mat guided_filter(const cv::Mat& guide,   // CV_32FC1
                             const cv::Mat& src,     // CV_32FC1
                             int radius,
                             double epsilon) {
    const cv::Size ksize(2 * radius + 1, 2 * radius + 1);

    // Helper: fast mean via box filter
    auto mean_f = [&](const cv::Mat& m) {
        cv::Mat out;
        cv::boxFilter(m, out, CV_32F, ksize, cv::Point(-1, -1), true, cv::BORDER_REFLECT_101);
        return out;
    };

    cv::Mat mean_I  = mean_f(guide);
    cv::Mat mean_p  = mean_f(src);
    cv::Mat mean_Ip = mean_f(guide.mul(src));
    cv::Mat mean_II = mean_f(guide.mul(guide));

    // cov(I,p) = E[I*p] - E[I]*E[p]
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    // var(I)   = E[I*I] - E[I]^2
    cv::Mat var_I  = mean_II - mean_I.mul(mean_I);

    // a = cov(I,p) / (var(I) + eps)
    cv::Mat a = cov_Ip / (var_I + static_cast<float>(epsilon));
    // b = E[p] - a * E[I]
    cv::Mat b = mean_p - a.mul(mean_I);

    // Smooth a and b over local windows
    cv::Mat mean_a = mean_f(a);
    cv::Mat mean_b = mean_f(b);

    // q = mean_a * I + mean_b
    cv::Mat q = mean_a.mul(guide) + mean_b;

    return q;
}

// ---------------------------------------------------------------------------

GuidedFilterPostProcessor::GuidedFilterPostProcessor(int radius, double epsilon, float threshold, int erode_iters, int src_blur_ksize)
    : radius_(radius), epsilon_(epsilon), threshold_(threshold), erode_iters_(erode_iters), src_blur_ksize_(src_blur_ksize) {}

cv::Mat GuidedFilterPostProcessor::process(const cv::Mat& alpha_f32,
                                           const cv::Mat& guide_bgr) const {
    if (alpha_f32.empty()) {
        throw std::invalid_argument("GuidedFilterPostProcessor: alpha_f32 is empty");
    }
    if (guide_bgr.empty()) {
        throw std::invalid_argument("GuidedFilterPostProcessor: guide_bgr is empty");
    }
    if (alpha_f32.size() != guide_bgr.size()) {
        throw std::invalid_argument(
            "GuidedFilterPostProcessor: alpha_f32 and guide_bgr must have the same size");
    }

    // Step 1: Optional threshold — harden the soft alpha edge before GF.
    // Without this, a wide soft transition band (~30-50px after upscaling from 512px)
    // prevents GF from knowing which physical edge to snap to.
    cv::Mat src;
    if (threshold_ > 0.0f) {
        cv::threshold(alpha_f32, src, static_cast<double>(threshold_), 1.0,
                      cv::THRESH_BINARY);
        src.convertTo(src, CV_32F);
    } else {
        src = alpha_f32;
    }

    // Step 2: Optional morphological erosion — pull mask boundary inward.
    // The AI mask silhouette tends to sit ~3-5px outside the true physical
    // boundary. Eroding before GF compensates this offset so GF snaps to
    // the correct edge rather than an already-offset one.
    if (erode_iters_ > 0) {
        cv::Mat src_8u;
        src.convertTo(src_8u, CV_8U, 255.0);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(src_8u, src_8u, kernel, cv::Point(-1, -1), erode_iters_);
        src_8u.convertTo(src, CV_32F, 1.0 / 255.0);
    }

    // Step 3: Optional Gaussian blur on src — soften pixel-level jagged edges
    // on the binary mask before GF, so GF snaps to a smooth boundary rather
    // than inheriting the staircase pattern from the discrete binary edge.
    if (src_blur_ksize_ > 0) {
        int ksize = (src_blur_ksize_ % 2 == 0) ? src_blur_ksize_ + 1 : src_blur_ksize_;
        cv::GaussianBlur(src, src, cv::Size(ksize, ksize), 0);
    }

    // Step 4: Convert BGR guide to grayscale float32 [0, 1]
    cv::Mat guide_gray_8u;
    cv::cvtColor(guide_bgr, guide_gray_8u, cv::COLOR_BGR2GRAY);
    cv::Mat guide_f32;
    guide_gray_8u.convertTo(guide_f32, CV_32F, 1.0 / 255.0);

    // Step 5: Guided filter — snap edges to physical pixel boundaries
    cv::Mat result = guided_filter(guide_f32, src, radius_, epsilon_);

    // Clamp back to [0, 1]
    cv::Mat clamped;
    cv::min(cv::max(result, 0.0f), 1.0f, clamped);

    return clamped;
}

