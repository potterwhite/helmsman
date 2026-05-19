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
// i-preprocessor.h — Abstract preprocessor interface (internal to Frontend)
//
// A Preprocessor converts a decoded frame (cv::Mat BGR) into a TensorData
// structure ready for the inference engine.
//
// =============================================================================

#pragma once

#include <opencv2/core/mat.hpp>
#include "common/data_structure.h"

// Abstract preprocessor interface (internal — do not use directly).
class _IPreprocessor {
public:
    virtual ~_IPreprocessor() = default;

    // Preprocess a BGR frame into a TensorData for inference.
    virtual TensorData preprocess(const cv::Mat& bgr_frame,
                                  size_t model_width,
                                  size_t model_height) = 0;
};
