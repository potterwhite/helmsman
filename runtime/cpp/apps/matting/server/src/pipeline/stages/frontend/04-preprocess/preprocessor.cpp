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
// preprocessor.cpp — Image preprocessor for inference (internal)
//
// =============================================================================

#include "pipeline/stages/frontend/04-preprocess/preprocessor.h"

#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "common/common-define.h"
#include "common/types.h"

Preprocessor::Preprocessor() {
    helmsman::utils::Logger::GetInstance().Info(
        "Preprocessor constructed.", kcurrent_module_name);
}

Preprocessor::~Preprocessor() {
    helmsman::utils::Logger::GetInstance().Info(
        "Preprocessor cleaned up.", kcurrent_module_name);
}

void Preprocessor::SetOutputBinPath(const std::string& path) {
    output_bin_path_ = path;
}

void Preprocessor::SetDumpEnabled(bool enabled) { dump_enabled_ = enabled; }

TensorData Preprocessor::preprocess(const cv::Mat& bgr_frame,
                                     int model_width,
                                     int model_height) {
    return PreprocessCore(bgr_frame.clone(), model_width, model_height);
}

TensorData Preprocessor::PreprocessCore(cv::Mat img,
                                         int model_width,
                                         int model_height) {
    TensorData tensor_data;

    auto& logger = helmsman::utils::Logger::GetInstance();
    auto& file_utils = helmsman::utils::FileUtils::GetInstance();

    // Step 1: BGR → RGB
    img = cvkit_.bgrToRgb(img);
    if (dump_enabled_)
        cvkit_.dumpBinary(img, output_bin_path_ + "/cpp_02_bgrToRgb.bin");

    // Step 2: Ensure 3 channels
    img = cvkit_.ensure3Channel(img);
    if (dump_enabled_)
        cvkit_.dumpBinary(img, output_bin_path_ + "/cpp_03_ensure3Channel.bin");

    int original_w = img.cols;
    int original_h = img.rows;

    logger.Info("Original size: Width=" + std::to_string(img.cols) +
                    ", Height=" + std::to_string(img.rows),
                kcurrent_module_name);

    // Step 3: Resize strategy
    int pad_top = 0;
    int pad_bottom = (32 - (img.rows % 32)) % 32;
    int pad_left = 0;
    int pad_right = (32 - (img.cols % 32)) % 32;

    if (model_width > 0 && model_height > 0) {
        helmsman::utils::timing::ManualTimer t_resize;
        t_resize.start();
        cv::resize(img, img,
                   cv::Size(model_width, model_height),
                   0, 0, cv::INTER_LINEAR);
        acc_resize_.record(t_resize.stop());
        logger.Info("RKNN resize: " + std::to_string(original_w) + "x" +
                        std::to_string(original_h) + " -> " +
                        std::to_string(model_width) + "x" +
                        std::to_string(model_height),
                    kcurrent_module_name);
        img.convertTo(img, CV_32FC3);
    } else {
        pad_bottom = (32 - (img.rows % 32)) % 32;
        pad_right = (32 - (img.cols % 32)) % 32;

        cv::Mat padded_u8;
        cv::copyMakeBorder(img, padded_u8, pad_top, pad_bottom,
                           pad_left, pad_right, cv::BORDER_REPLICATE);
        logger.Info("Replicate pad: " + std::to_string(img.cols) + "x" +
                        std::to_string(img.rows) + " -> " +
                        std::to_string(padded_u8.cols) + "x" +
                        std::to_string(padded_u8.rows) +
                        " (pad_right=" + std::to_string(pad_right) +
                        ", pad_bottom=" + std::to_string(pad_bottom) + ")",
                    kcurrent_module_name);
        padded_u8.convertTo(img, CV_32FC3);
    }

    if (dump_enabled_)
        cvkit_.dumpBinary(img, output_bin_path_ + "/cpp_05_resized.bin");

    // Step 4: Copy HWC continuous memory
    cv::Mat continuous_img;
    if (img.isContinuous()) {
        continuous_img = img;
    } else {
        continuous_img = img.clone();
    }

    CV_Assert(continuous_img.isContinuous());

    size_t total = continuous_img.total() *
                   static_cast<size_t>(continuous_img.channels());
    float* ptr = continuous_img.ptr<float>(0);

    tensor_data.data.assign(ptr, ptr + total);

    if (dump_enabled_)
        file_utils.dumpBinary(tensor_data.data,
                              output_bin_path_ + "/cpp_06-07_hwc_direct.bin");

    // Step 5: Construct NHWC shape
    tensor_data.shape = {1, static_cast<int64_t>(img.rows),
                         static_cast<int64_t>(img.cols), 3};

    tensor_data.orig_width = original_w;
    tensor_data.orig_height = original_h;
    tensor_data.pad_top = pad_top;
    tensor_data.pad_bottom = pad_bottom;
    tensor_data.pad_left = pad_left;
    tensor_data.pad_right = pad_right;

    return tensor_data;
}
