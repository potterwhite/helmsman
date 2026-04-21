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

#include "pipeline/modes/rvm/rvm.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "common/common-define.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"
#include "Utils/timing/timer.h"

using arcforge::utils::timing::ManualTimer;
using arcforge::utils::timing::ScopedTimer;
using arcforge::utils::timing::StageAccumulator;

extern std::atomic<bool> g_stop_signal_received;

constexpr float kDownsampleRatio = 0.25f;
inline constexpr std::string_view kRvmModuleName = "RVMMode";

template <typename T>
class SingleSlotChannel {
public:
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_empty_.wait(lk, [this] { return !has_item_ || closed_; });
        if (closed_) return false;
        item_     = std::move(item);
        has_item_ = true;
        cv_full_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_full_.wait(lk, [this] { return has_item_ || closed_; });
        if (!has_item_) return std::nullopt;
        T out      = std::move(*item_);
        item_      = std::nullopt;
        has_item_  = false;
        cv_empty_.notify_one();
        return out;
    }

    void close() {
        std::unique_lock<std::mutex> lk(mtx_);
        closed_ = true;
        cv_full_.notify_all();
        cv_empty_.notify_all();
    }

private:
    std::mutex              mtx_;
    std::condition_variable cv_empty_;
    std::condition_variable cv_full_;
    std::optional<T>        item_;
    bool                    has_item_ = false;
    bool                    closed_   = false;
};

void RVMMode::initRecurrentStates(size_t model_input_height, size_t model_input_width) {
    const int64_t internal_h =
        static_cast<int64_t>(static_cast<float>(model_input_height) * kDownsampleRatio);
    const int64_t internal_w =
        static_cast<int64_t>(static_cast<float>(model_input_width) * kDownsampleRatio);

    auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    state_mgr_.init(
        {
            {1, 16, ceil_div(internal_h, 2),  ceil_div(internal_w, 2)},
            {1, 20, ceil_div(internal_h, 4),  ceil_div(internal_w, 4)},
            {1, 40, ceil_div(internal_h, 8),  ceil_div(internal_w, 8)},
            {1, 64, ceil_div(internal_h, 16), ceil_div(internal_w, 16)},
        },
        {"r1i", "r2i", "r3i", "r4i"});
}

bool RVMMode::openVideoWriter(cv::VideoWriter& writer, const std::string& path,
                           int width, int height, double fps) {
    auto& logger = arcforge::embedded::utils::Logger::GetInstance();
    writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        logger.Warning("Failed to open VideoWriter: " + path + ". Composited video will NOT be saved.",
                     kRvmModuleName);
        return false;
    }
    logger.Info("VideoWriter opened: " + path + " (" + std::to_string(width) + "x" +
                    std::to_string(height) + " @ " + std::to_string(fps) + " fps)",
                kRvmModuleName);
    return true;
}

cv::Mat RVMMode::loadOrCreateBackground(int width, int height) {
    cv::Mat bg;
    if (!background_path_.empty()) {
        bg = cv::imread(background_path_, cv::IMREAD_COLOR);
        if (!bg.empty()) {
            cv::resize(bg, bg, cv::Size(width, height));
            return bg;
        }
    }
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 100, 0));
}

cv::Mat RVMMode::inferOneFrame(InferenceEngine* engine, const TensorData& src) {
    auto& logger = arcforge::embedded::utils::Logger::GetInstance();

    std::vector<TensorData> inputs = {src};
    state_mgr_.inject(inputs);

#if defined(INFERENCE_BACKEND_ONNX)
    TensorData dsr;
    dsr.name  = "downsample_ratio";
    dsr.shape = {1};
    dsr.data  = {kDownsampleRatio};
    inputs.push_back(std::move(dsr));
#endif

    std::vector<TensorData> outputs;
    auto t0 = std::chrono::high_resolution_clock::now();
    engine->infer(inputs, outputs);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur = t1 - t0;
    logger.Info("infer() cost: " + std::to_string(dur.count()) + " ms.", kRvmModuleName);

    state_mgr_.update(outputs);
    return backend_.postprocess(outputs);
}

void RVMMode::compositeAndWrite(cv::VideoWriter& writer, const cv::Mat& frame,
                                 const cv::Mat& alpha_8u, const cv::Mat& bg_bgr) {
    if (!writer.isOpened() || alpha_8u.empty()) {
        return;
    }

    cv::Mat alpha_1ch;
    if (alpha_8u.channels() == 1) {
        alpha_1ch = alpha_8u;
    } else {
        cv::cvtColor(alpha_8u, alpha_1ch, cv::COLOR_BGR2GRAY);
    }

    if (alpha_1ch.size() != frame.size()) {
        cv::resize(alpha_1ch, alpha_1ch, frame.size(), 0, 0, cv::INTER_LINEAR);
    }

    cv::Mat alpha_f32;
    alpha_1ch.convertTo(alpha_f32, CV_32FC1, 1.0 / 255.0);

    cv::Mat alpha_3ch;
    cv::cvtColor(alpha_f32, alpha_3ch, cv::COLOR_GRAY2BGR);

    cv::Mat fg_f32, bg_f32, composed_f32, composed_8u;
    frame.convertTo(fg_f32, CV_32FC3, 1.0 / 255.0);
    bg_bgr.convertTo(bg_f32, CV_32FC3, 1.0 / 255.0);

    composed_f32 = alpha_3ch.mul(fg_f32) + (cv::Scalar(1.0, 1.0, 1.0) - alpha_3ch).mul(bg_f32);
    composed_f32.convertTo(composed_8u, CV_8UC3, 255.0);

    writer.write(composed_8u);
}

int RVMMode::run(InferenceEngine* engine,
                  std::unique_ptr<InputSource> input_source,
                  const std::string& model_path,
                  const std::string& output_bin_path,
                  const std::string& background_path,
                  bool timing_enabled) {
    auto& logger = arcforge::embedded::utils::Logger::GetInstance();

    output_bin_path_ = output_bin_path;
    background_path_ = background_path;

    ScopedTimer run_rvm_timer("runRVM total", timing_enabled, logger, kRvmModuleName);

    // 1. Load model
    {
        ScopedTimer t("runRVM: model load", timing_enabled, logger, kRvmModuleName);
        engine->setOutputBinPath(output_bin_path);
        engine->load(model_path);
    }

    const size_t model_input_height = engine->getInputHeight() > 0 ? engine->getInputHeight() : 288;
    const size_t model_input_width  = engine->getInputWidth()  > 0 ? engine->getInputWidth()  : 512;

    // 2. Recurrent state manager
    initRecurrentStates(model_input_height, model_input_width);

    // 3. Frontend setup
    frontend_.setOutputBinPath(output_bin_path);

    // 4. Backend setup
    backend_.setOutputPath(output_bin_path);
    backend_.setBackgroundPath(background_path);

    // 5. Video output
    const int    src_width  = input_source->width();
    const int    src_height = input_source->height();
    const double src_fps    = input_source->fps();
    const double output_fps = (src_fps > 0) ? src_fps : 30.0;
    const std::string output_video_path = output_bin_path + "/output_composited.mp4";

    cv::VideoWriter video_writer;
    openVideoWriter(video_writer, output_video_path, src_width, src_height, output_fps);

    cv::Mat bg_bgr = loadOrCreateBackground(src_width, src_height);

    // 6. Persistent prefetch worker
    SingleSlotChannel<cv::Mat>    raw_ch;
    SingleSlotChannel<TensorData> tensor_ch;

    StageAccumulator preprocess_acc("worker::preprocess");
    StageAccumulator infer_acc("main::infer+composite");

    std::thread prefetch_worker([this, model_input_width, model_input_height,
                                 &raw_ch, &tensor_ch, &preprocess_acc]() {
        while (true) {
            auto frame_opt = raw_ch.pop();
            if (!frame_opt) break;
            ManualTimer t;
            t.start();
            auto tensor = prefetch_frontend_.preprocess(*frame_opt, model_input_width, model_input_height);
            preprocess_acc.record(t.stop());
            tensor_ch.push(std::move(tensor));
        }
        tensor_ch.close();
    });

    cv::Mat current_frame;
    if (!input_source->read(current_frame)) {
        logger.Info("No frames to process.", kRvmModuleName);
        raw_ch.close();
        prefetch_worker.join();
        return 0;
    }
    raw_ch.push(current_frame);

    int frame_count = 0;

    // 7. Main loop
    while (true) {
        if (g_stop_signal_received.load()) {
            logger.Info("Stop signal received. Finishing video at frame " +
                            std::to_string(frame_count) + ".",
                        kRvmModuleName);
            raw_ch.close();
            break;
        }

        auto tensor_opt = tensor_ch.pop();
        if (!tensor_opt) break;

        logger.Info("=== RVM Frame " + std::to_string(frame_count + 1) + " ===",
                    kRvmModuleName);

        cv::Mat next_frame;
        bool    has_next = input_source->read(next_frame);
        if (has_next) {
            raw_ch.push(next_frame);
        } else {
            raw_ch.close();
        }

        ManualTimer infer_t;
        infer_t.start();
        cv::Mat alpha_8u = inferOneFrame(engine, *tensor_opt);
        compositeAndWrite(video_writer, current_frame, alpha_8u, bg_bgr);
        infer_acc.record(infer_t.stop());

        current_frame = std::move(next_frame);
        frame_count++;

        if (!has_next) break;
    }

    prefetch_worker.join();

    preprocess_acc.report(timing_enabled, logger, kRvmModuleName);
    infer_acc.report(timing_enabled, logger, kRvmModuleName);

    if (video_writer.isOpened()) {
        video_writer.release();
        logger.Info("Video compositing complete: " + std::to_string(frame_count) +
                        " frames written to " + output_video_path,
                    kRvmModuleName);
    }
    logger.Info("RVM video pipeline finished. Total frames: " + std::to_string(frame_count),
                kRvmModuleName);
    return 0;
}

int RVMMode::runSinglePicture(InferenceEngine* engine,
                              const std::string& input_image_path,
                              const std::string& model_path,
                              const std::string& output_bin_path,
                              const std::string& background_path,
                              bool timing_enabled) {
    (void)timing_enabled;
    auto& logger = arcforge::embedded::utils::Logger::GetInstance();

    output_bin_path_ = output_bin_path;
    background_path_ = background_path;

    MattingBackend backend;

    engine->setOutputBinPath(output_bin_path);
    engine->load(model_path);

    const size_t model_input_height = engine->getInputHeight() > 0 ? engine->getInputHeight() : 288;
    const size_t model_input_width  = engine->getInputWidth()  > 0 ? engine->getInputWidth()  : 512;

    initRecurrentStates(model_input_height, model_input_width);

    frontend_.setOutputBinPath(output_bin_path);
    TensorData src = frontend_.preprocess(input_image_path, model_input_width, model_input_height);

    backend.setOutputPath(output_bin_path);
    backend.setBackgroundPath(background_path);
    backend.setForegroundImagePath(input_image_path);
    backend.setPostProcessor(std::make_shared<GuidedFilterPostProcessor>(2, 1e-4, 0.2f, 1));

    constexpr int kNumTestFrames = 5;
    for (int frame = 0; frame < kNumTestFrames; ++frame) {
        logger.Info("=== RVM Frame " + std::to_string(frame + 1) + "/" +
                        std::to_string(kNumTestFrames) + " ===",
                    kRvmModuleName);

        std::vector<TensorData> inputs   = {src};
        state_mgr_.inject(inputs);

#if defined(INFERENCE_BACKEND_ONNX)
        TensorData dsr;
        dsr.name  = "downsample_ratio";
        dsr.shape = {1};
        dsr.data  = {kDownsampleRatio};
        inputs.push_back(std::move(dsr));
#endif

        std::vector<TensorData> outputs;
        auto t0 = std::chrono::high_resolution_clock::now();
        engine->infer(inputs, outputs);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dur = t1 - t0;
        logger.Info("[RVM Frame " + std::to_string(frame + 1) +
                        "] infer() cost: " + std::to_string(dur.count()) + " ms.",
                    kRvmModuleName);

        state_mgr_.update(outputs);

        if (frame == kNumTestFrames - 1) {
            backend.postprocess(outputs);
        }
    }

    return 0;
}