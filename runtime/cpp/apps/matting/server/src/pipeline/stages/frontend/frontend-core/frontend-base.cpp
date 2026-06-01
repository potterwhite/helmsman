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
// frontend-base.cpp — FrontendBase implementation (Template Method pattern)
//
// Owns the multithread infrastructure. Subclasses override stage methods
// (ReadInputSource01, DecodeFrame02, ConvertToBgr03) to supply decoded frames
// via the platform-specific decode path. Stage 04 (preprocess) is non-virtual
// and owned by the base class.
//
// =============================================================================

#include "pipeline/stages/frontend/frontend-core/frontend-base.h"

// ---------------------------------------------------------------------------
// Protected constructor
// ---------------------------------------------------------------------------
FrontendBase::FrontendBase(bool use_hardware, bool multithread_enabled)
    : use_hardware_(use_hardware), multithread_enabled_(multithread_enabled) {
	if (multithread_enabled_) {
		raw_ch_ = std::make_unique<SingleSlotChannel<cv::Mat>>();
		tensor_ch_ = std::make_unique<SingleSlotChannel<TensorData>>();
	}
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FrontendBase::~FrontendBase() {
	Stop();
}

// ---------------------------------------------------------------------------
// SetSourceProperties — called by subclasses after opening the source
// ---------------------------------------------------------------------------
void FrontendBase::SetSourceProperties(int width, int height, double fps) {
	width_ = width;
	height_ = height;
	fps_ = fps;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
bool FrontendBase::IsHardwarePath() const {
	return use_hardware_;
}
int FrontendBase::width() const {
	return width_;
}
int FrontendBase::height() const {
	return height_;
}
double FrontendBase::fps() const {
	return fps_;
}

const helmsman::utils::timing::StageAccumulator& FrontendBase::read_input_acc() const {
	return acc_lv03_02_frontend_read_;
}

const helmsman::utils::timing::StageAccumulator& FrontendBase::decode_acc() const {
	return acc_lv03_03_frontend_decode_;
}

const helmsman::utils::timing::StageAccumulator& FrontendBase::color_convert_acc() const {
	return acc_lv03_04_frontend_color_convert_;
}

const helmsman::utils::timing::StageAccumulator& FrontendBase::preprocess_acc() const {
	return acc_lv03_05_frontend_preprocess_;
}

const helmsman::utils::timing::StageAccumulator& FrontendBase::resize_acc() const {
	return preprocessor_.resize_acc();
}

// ---------------------------------------------------------------------------
// Stage 01: default no-op (subclasses override)
// ---------------------------------------------------------------------------
bool FrontendBase::_ReadInputSource01(RawPacket& /*pkt*/, ReadResult& /*result*/) {
	return false;
}

// ---------------------------------------------------------------------------
// Stage 02: default no-op (subclasses override).
// Skips if result.frame already populated (NoHwFrontend Stage 01 is atomic).
// ---------------------------------------------------------------------------
bool FrontendBase::_DecodeFrame02(const RawPacket& /*pkt*/, ReadResult& result) {
	return !result.frame.empty();
}

// ---------------------------------------------------------------------------
// Stage 03: default no-op (subclasses override).
// Skips if result.frame already populated (NoHwFrontend Stage 01 is atomic).
// ---------------------------------------------------------------------------
bool FrontendBase::_ConvertToBgr03(ReadResult& result) {
	return !result.frame.empty();
}

// ---------------------------------------------------------------------------
// Stage 04: preprocess BGR frame into TensorData (non-virtual)
// ---------------------------------------------------------------------------
TensorData FrontendBase::_PreprocessForInference04(const cv::Mat& frame, int model_w, int model_h) {
	helmsman::utils::timing::ManualTimer t;
	t.start();
	auto tensor = preprocessor_.preprocess(frame, model_w, model_h);
	acc_lv03_05_frontend_preprocess_.record(t.stop());
	return tensor;
}

// ---------------------------------------------------------------------------
// _MultithreadWorkerLoop — worker thread entry point (stage 04 only)
// ---------------------------------------------------------------------------
void FrontendBase::_MultithreadWorkerLoop(int model_w, int model_h) {
	while (true) {
		auto frame_opt = raw_ch_->pop();
		if (!frame_opt)
			break;

		auto tensor = _PreprocessForInference04(*frame_opt, model_w, model_h);
		tensor_ch_->push(std::move(tensor));
	}
	tensor_ch_->close();
}

// ---------------------------------------------------------------------------
// _ProcessSync — single-thread mode: 4 stages on calling thread
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontendBase::_ProcessSync(int model_w, int model_h) {
	ReadResult read_result;
	RawPacket pkt;

	// Stages 01-03: read + decode + color convert (with retry loop for hardware decoders)
	while (true) {
		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			if (!_ReadInputSource01(pkt, read_result))
				return std::nullopt;
			acc_lv03_02_frontend_read_.record(t.stop());
		}
		if (pkt.is_eof)
			return std::nullopt;

		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			bool decoded = _DecodeFrame02(pkt, read_result);
			acc_lv03_03_frontend_decode_.record(t.stop());
			if (!decoded)
				continue;  // decoder needs more data
		}

		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			if (!_ConvertToBgr03(read_result))
				return std::nullopt;
			acc_lv03_04_frontend_color_convert_.record(t.stop());
		}

		// Stage 04: preprocess
		FrameResult result;
		result.frame = std::move(read_result.frame);
		result.hw_frame = read_result.hw_frame;
		result.tensor = _PreprocessForInference04(result.frame, model_w, model_h);
		return result;
	}
}

// ---------------------------------------------------------------------------
// _ProcessMultithread — dual-buffer mode: stages 01-03 on main thread,
//                       stage 04 on worker thread
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontendBase::_ProcessMultithread(int model_w, int model_h) {
	if (mt_eof_)
		return std::nullopt;

	auto read_stages_01_03 = [this](ReadResult& rr) -> bool {
		RawPacket pkt;
		while (true) {
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				if (!_ReadInputSource01(pkt, rr))
					return false;
				acc_lv03_02_frontend_read_.record(t.stop());
			}
			if (pkt.is_eof)
				return false;
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				bool decoded = _DecodeFrame02(pkt, rr);
				acc_lv03_03_frontend_decode_.record(t.stop());
				if (!decoded)
					continue;
			}
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				bool ok = _ConvertToBgr03(rr);
				acc_lv03_04_frontend_color_convert_.record(t.stop());
				return ok;
			}
		}
	};

	if (!mt_started_) {
		// Phase 1: bootstrap
		mt_started_ = true;

		// Stages 01-03: read frame 1
		ReadResult read1;
		if (!read_stages_01_03(read1))
			return std::nullopt;

		cv::Mat frame_1 = std::move(read1.frame);
		stored_hw_frame_ = read1.hw_frame;

		// Launch worker for stage 04
		prefetch_worker_ =
		    std::thread(&FrontendBase::_MultithreadWorkerLoop, this, model_w, model_h);
		raw_ch_->push(frame_1);

		// Stage 04: pop tensor from worker
		auto tensor_1 = tensor_ch_->pop();
		if (!tensor_1) {
			mt_eof_ = true;
			return std::nullopt;
		}

		// Stages 01-03: read frame 2 (overlaps with stage 04 for frame 1)
		ReadResult read2;
		if (read_stages_01_03(read2)) {
			next_frame_ = std::move(read2.frame);
			next_hw_frame_ = read2.hw_frame;
			raw_ch_->push(next_frame_);
		} else {
			raw_ch_->close();
		}

		FrameResult result;
		result.frame = std::move(frame_1);
		result.hw_frame = stored_hw_frame_;
		result.tensor = std::move(*tensor_1);
		return result;
	}

	// Phase 2: subsequent calls

	// Stage 04: pop tensor from worker
	auto tensor = tensor_ch_->pop();
	if (!tensor) {
		mt_eof_ = true;
		return std::nullopt;
	}

	// Save current buffered frame to return
	cv::Mat return_frame = std::move(next_frame_);
	HardwareFrame return_hw_frame = next_hw_frame_;

	// Stages 01-03: read next frame
	ReadResult read_next;
	if (read_stages_01_03(read_next)) {
		next_frame_ = std::move(read_next.frame);
		next_hw_frame_ = read_next.hw_frame;
		raw_ch_->push(next_frame_);
	} else {
		raw_ch_->close();
	}

	FrameResult result;
	result.frame = std::move(return_frame);
	result.hw_frame = return_hw_frame;
	result.tensor = std::move(*tensor);
	return result;
}

// ---------------------------------------------------------------------------
// Stop — signal prefetch worker to stop
// ---------------------------------------------------------------------------
void FrontendBase::Stop() {
	if (raw_ch_)
		raw_ch_->close();
	if (prefetch_worker_.joinable())
		prefetch_worker_.join();
}

// ---------------------------------------------------------------------------
// ProcessOneFrame — dispatcher: sync or multithread
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontendBase::ProcessOneFrame(int model_w, int model_h) {
	if (!multithread_enabled_)
		return _ProcessSync(model_w, model_h);
	return _ProcessMultithread(model_w, model_h);
}
