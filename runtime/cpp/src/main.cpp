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


#include <atomic>
#include <chrono>
#include <csignal>  // For signal handling
#include <iostream>
#include <sstream>  // For std::ostringstream
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

// 假设 kcurrent_lib_name 在此编译单元中可用
// 这里我们定义一个临时的，实际项目中可能来自别处
const std::string_view kcurrent_app_name = PROJECT_NAME;

// --- 全局用于信号处理 ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);  // 修改后的代码

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	// 使用 Logger 而不是 cerr
	std::cout << oss.str() << std::endl;
}

bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}


namespace helmsman {
namespace preprocess {

/**
 * @brief Load image from disk using OpenCV
 */
cv::Mat loadImage(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    return img;
}

/**
 * @brief Ensure image is 3-channel RGB
 */
cv::Mat ensureRGB3Channel(const cv::Mat& input)
{
    cv::Mat img = input;

    // Convert grayscale to 3-channel
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    }
    // Drop alpha channel if exists
    else if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    }

    // Convert BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    return img;
}

/**
 * @brief Normalize image to range [-1, 1]
 * Formula: (pixel - 127.5) / 127.5
 */
cv::Mat normalizeToMinusOneToOne(const cv::Mat& input)
{
    cv::Mat floatImg;
    input.convertTo(floatImg, CV_32FC3);

    floatImg = (floatImg - 127.5f) / 127.5f;

    return floatImg;
}

/**
 * @brief Dump raw float data to binary file
 */
void dumpBinary(const cv::Mat& mat, const std::string& outputPath)
{
    if (!mat.isContinuous()) {
        throw std::runtime_error("Mat is not continuous in memory");
    }

    std::ofstream ofs(outputPath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output file: " + outputPath);
    }

    ofs.write(
        reinterpret_cast<const char*>(mat.ptr<float>(0)),
        static_cast<std::streamsize>(mat.total() * mat.elemSize())
    );

    ofs.close();
}

} // namespace preprocess
} // namespace helmsman

// // ------------------------------------------------------------

// int main(int argc, char** argv)
// {
//     if (argc != 3) {
//         std::cerr << "Usage: preprocess_normalize <input_image> <output_bin>\n";
//         return 1;
//     }

//     const std::string imagePath = argv[1];
//     const std::string outputBinPath = argv[2];

//     try {
//         cv::Mat img = helmsman::preprocess::loadImage(imagePath);
//         img = helmsman::preprocess::ensureRGB3Channel(img);
//         img = helmsman::preprocess::normalizeToMinusOneToOne(img);
//         helmsman::preprocess::dumpBinary(img, outputBinPath);
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }


int main(int argc, char* argv[]) {

	// 1. 设置日志级别 (例如，只记录 Warning 及以上)
	if (isRelease() == true) {
		// logger.setLevel(kinfo);
	} else {
		// logger.setLevel(kdebug);
	}

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

    if (argc != 3) {
        std::cerr << "Usage: preprocess_normalize <input_image> <output_bin>\n";
        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string outputBinPath = argv[2];

    try {
        cv::Mat img = helmsman::preprocess::loadImage(imagePath);
        img = helmsman::preprocess::ensureRGB3Channel(img);
        img = helmsman::preprocess::normalizeToMinusOneToOne(img);
        helmsman::preprocess::dumpBinary(img, outputBinPath);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

	std::cout << "hello " << kcurrent_app_name << std::endl;

	return 0;
}