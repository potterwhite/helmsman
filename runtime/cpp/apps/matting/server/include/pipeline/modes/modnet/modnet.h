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

#include <chrono>
#include <string>
#include <vector>
#include <opencv2/videoio.hpp>
#include "common/types.h"
#include "pipeline/stages/inference-engine/engine-core/inference-engine.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend/frontend-core/frontend.h"
#include "DRMKit/drm_display.h"

class MODNetMode {
public:
    void SetEngine(InferenceEngine* engine);
    void SetFrontend(FrontEnd* frontend);
    void SetBackend(BackEnd* backend);
    void SetAppConfig(const AppConfig& config);

    int Run();

private:
    void _InitOutputSink(int src_width, int src_height, double fps);
    void _Display(const cv::Mat& result, int output_w, int output_h);

    InferenceEngine* engine_ = nullptr;  // Non-owning; owned by Pipeline
    FrontEnd* frontend_ = nullptr;  // Non-owning; owned by Pipeline
    BackEnd* backend_ = nullptr;  // Non-owning; owned by Pipeline
    AppConfig config_;

    // DRM display (initialized when output_mode == kDrm)
    helmsman::drmkit::DrmDisplay drm_display_;
    std::vector<uint8_t> argb_buf_;  // reusable buffer for BGR→XRGB conversion
    int drm_panel_w_ = 0;
    int drm_panel_h_ = 0;

    // Video output (initialized when output_mode == kMp4)
    cv::VideoWriter video_writer_;

    // Loop state
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point fps_window_start_;
};
