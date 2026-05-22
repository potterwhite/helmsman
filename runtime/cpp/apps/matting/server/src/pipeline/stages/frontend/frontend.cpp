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
// frontend.cpp — Frontend for the matting pipeline
//
// Two paths:
//   Hardware path: BaseInputSource::ReadRaw → BaseFrameDecoder::decode
//                → BaseColorConverter::convert → cpu_frame (cv::Mat)
//   OpenCV path:   cv::VideoCapture::read → cv::Mat (software decode)
//
// =============================================================================

#include "pipeline/stages/frontend/frontend.h"

#include <cstdio>

// ---------------------------------------------------------------------------
// deconstructor
// ---------------------------------------------------------------------------
Frontend::~Frontend() = default;

// ---------------------------------------------------------------------------
// Hardware decode path constructor (DI)
// ---------------------------------------------------------------------------
Frontend::Frontend(std::unique_ptr<BaseInputSource> source, std::unique_ptr<BaseFrameDecoder> decoder,
                   std::unique_ptr<BaseColorConverter> converter)
    : use_hardware_(true), source_(std::move(source)), decoder_(std::move(decoder)),
      color_converter_(std::move(converter)) {
}

// ---------------------------------------------------------------------------
// OpenCV shortcut path constructor
// ---------------------------------------------------------------------------
Frontend::Frontend(const std::string& video_path) : use_hardware_(false) {
	if (!cv_cap_.open(video_path)) {
		fprintf(stderr, "[Frontend] failed to open video: %s\n", video_path.c_str());
	}
}

// ---------------------------------------------------------------------------
// ReadFrame — read the next decoded frame
// ---------------------------------------------------------------------------
bool Frontend::ReadFrame(cv::Mat& cpu_frame, HardwareFrame& hw_frame) {
	cpu_frame.release();
	hw_frame = HardwareFrame{};

	if (use_hardware_) {
		if (!source_ || !decoder_ || !color_converter_) {
			return false;
		}

		// Feed packets until the decoder produces a frame or we hit EOF.
		// Hardware decoders may need multiple packets before the first
		// frame appears (e.g. SPS/PPS in H.264), so a single failed
		// decode does not mean end-of-stream.
		RawPacket pkt;
		while (true) {
			if (!source_->ReadRaw(pkt) || pkt.is_eof) {
				return false;
			}

			if (decoder_->decode(pkt.data, pkt.size, hw_frame)) {
				break;  // Got a decoded frame
			}
			// decode returned false — decoder needs more data, keep feeding
		}

		// Convert hardware frame to BGR via color converter
		return color_converter_->convert(hw_frame, cpu_frame);
	}

	if (!cv_cap_.isOpened()) {
		return false;
	}

	return cv_cap_.read(cpu_frame);
}

// ---------------------------------------------------------------------------
// preprocess — convert a BGR frame into TensorData
// ---------------------------------------------------------------------------
TensorData Frontend::preprocess(const cv::Mat& frame, size_t model_w, size_t model_h) {
	return preprocessor_.preprocess(frame, model_w, model_h);
}

// ---------------------------------------------------------------------------
// Source properties
// ---------------------------------------------------------------------------
int Frontend::width() const {
	if (use_hardware_ && source_) {
		return source_->width();
	}
	if (cv_cap_.isOpened()) {
		return static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
	}
	return 0;
}

int Frontend::height() const {
	if (use_hardware_ && source_) {
		return source_->height();
	}
	if (cv_cap_.isOpened()) {
		return static_cast<int>(cv_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
	}
	return 0;
}

double Frontend::fps() const {
	if (use_hardware_ && source_) {
		return source_->fps();
	}
	if (cv_cap_.isOpened()) {
		return cv_cap_.get(cv::CAP_PROP_FPS);
	}
	return 0.0;
}
