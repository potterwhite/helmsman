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

#include <memory>
#include <string>
#include <opencv2/videoio.hpp>
#include "pipeline/stages/inference-engine/base/inference-engine.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend.h"
#include "pipeline/recurrent-state-manager.h"
#include "input/input-source.h"

class RVMMode {
public:
    int run(InferenceEngine* engine,
           std::unique_ptr<InputSource> input_source,
           const std::string& model_path,
           const std::string& output_bin_path,
           const std::string& background_path,
           bool timing_enabled);

    int runSinglePicture(InferenceEngine* engine,
                         const std::string& input_image_path,
                         const std::string& model_path,
                         const std::string& output_bin_path,
                         const std::string& background_path,
                         bool timing_enabled);

private:
    void initRecurrentStates(size_t model_input_height, size_t model_input_width);
    bool openVideoWriter(cv::VideoWriter& writer, const std::string& path,
                         int width, int height, double fps);
    cv::Mat loadOrCreateBackground(int width, int height);
    cv::Mat inferOneFrame(InferenceEngine* engine, const TensorData& src);
    void compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
                           const cv::Mat& alpha_8u, const cv::Mat& bg_bgr);

    ImageFrontend frontend_;
    ImageFrontend prefetch_frontend_;
    MattingBackend backend_;
    RecurrentStateManager state_mgr_;
    std::string background_path_;
    std::string output_bin_path_;
};