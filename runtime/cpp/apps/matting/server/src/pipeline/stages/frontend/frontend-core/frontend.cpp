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
// frontend.cpp — FrontEnd implementation (Template Method pattern)
//
// Owns the multithread infrastructure. Subclasses override stage methods
// (ReadInputSource01, DecodeFrame02, ConvertToBgr03) to supply decoded frames
// via the platform-specific decode path. Stage 04 (preprocess) is non-virtual
// and owned by the base class.
//
// =============================================================================

#include "pipeline/stages/frontend/frontend-core/frontend.h"

// ---------------------------------------------------------------------------
// Protected constructor
// ---------------------------------------------------------------------------
FrontEnd::FrontEnd(bool use_hardware, bool multithread_enabled)
    : use_hardware_(use_hardware), multithread_enabled_(multithread_enabled) {
	if (multithread_enabled_) {
		raw_ch_ = std::make_unique<SingleSlotChannel<cv::Mat>>();
		tensor_ch_ = std::make_unique<SingleSlotChannel<TensorData>>();
	}
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FrontEnd::~FrontEnd() {
	Stop();
}

// ---------------------------------------------------------------------------
// SetSourceProperties — called by subclasses after opening the source
// ---------------------------------------------------------------------------
void FrontEnd::SetSourceProperties(int width, int height, double fps) {
	width_ = width;
	height_ = height;
	fps_ = fps;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
bool FrontEnd::IsHardwarePath() const {
	return use_hardware_;
}
int FrontEnd::width() const {
	return width_;
}
int FrontEnd::height() const {
	return height_;
}
double FrontEnd::fps() const {
	return fps_;
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::read_input_acc() const {
	return acc_lv03_02_frontend_read_;
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::decode_acc() const {
	return acc_lv03_03_frontend_decode_;
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::color_convert_acc() const {
	return acc_lv03_04_frontend_color_convert_;
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::preprocess_acc() const {
	return acc_lv03_05_frontend_preprocess_;
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::resize_acc() const {
	return preprocessor_.resize_acc();
}

const helmsman::utils::timing::StageAccumulator& FrontEnd::total_acc() const {
	return acc_total_;
}

void FrontEnd::ReportAccumulatedTimers(bool timing_enabled, helmsman::utils::Logger& logger,
                                           std::string_view module) const {
	read_input_acc().report(timing_enabled, logger, module);
	decode_acc().report(timing_enabled, logger, module);
	color_convert_acc().report(timing_enabled, logger, module);
	preprocess_acc().report(timing_enabled, logger, module);
	resize_acc().report(timing_enabled, logger, module);

	logger.Info("", module);    // blank line for separation
	acc_total_.report(timing_enabled, logger, module);
    logger.Info("", module);    // blank line for separation
}

// ---------------------------------------------------------------------------
// Stage 01: default no-op (subclasses override)
// ---------------------------------------------------------------------------
bool FrontEnd::_ReadInputSource01(RawPacket& /*pkt*/, ReadResult& /*result*/) {
	return false;
}

// ---------------------------------------------------------------------------
// Stage 02: default no-op (subclasses override).
// Skips if result.frame already populated (NoHwFrontend Stage 01 is atomic).
// ---------------------------------------------------------------------------
bool FrontEnd::_DecodeFrame02(const RawPacket& /*pkt*/, ReadResult& result) {
	return !result.frame.empty();
}

// ---------------------------------------------------------------------------
// Stage 03: default no-op (subclasses override).
// Skips if result.frame already populated (NoHwFrontend Stage 01 is atomic).
// ---------------------------------------------------------------------------
bool FrontEnd::_ConvertToBgr03(ReadResult& result) {
	return !result.frame.empty();
}

// ---------------------------------------------------------------------------
// Stage 04: preprocess BGR frame into TensorData (non-virtual)
// ---------------------------------------------------------------------------
TensorData FrontEnd::_PreprocessForInference04(const cv::Mat& frame, int model_w, int model_h) {
	helmsman::utils::timing::ManualTimer t;
	t.start();
	auto tensor = preprocessor_.preprocess(frame, model_w, model_h);
	acc_lv03_05_frontend_preprocess_.record(t.stop());
	return tensor;
}

// ---------------------------------------------------------------------------
// _MultithreadWorkerLoop — worker thread entry point (stage 04 only)
// ---------------------------------------------------------------------------
void FrontEnd::_MultithreadWorkerLoop(int model_w, int model_h) {
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
std::optional<FrameResult> FrontEnd::_ProcessSync(int model_w, int model_h) {
	ReadResult read_result;
	RawPacket pkt;
	FrameResult result;
	double read_total = 0, decode_total = 0, color_total = 0;

	// Stages 01-03: read + decode + color convert (with retry loop for hardware decoders)
	while (true) {
		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			if (!_ReadInputSource01(pkt, read_result))
				return std::nullopt;
			double ms = t.stop();
			acc_lv03_02_frontend_read_.record(ms);
			read_total += ms;
		}
		if (pkt.is_eof)
			return std::nullopt;

		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			bool decoded = _DecodeFrame02(pkt, read_result);
			double ms = t.stop();
			acc_lv03_03_frontend_decode_.record(ms);
			decode_total += ms;
			if (!decoded)
				continue;  // decoder needs more data
		}

		{
			helmsman::utils::timing::ManualTimer t;
			t.start();
			if (!_ConvertToBgr03(read_result))
				return std::nullopt;
			double ms = t.stop();
			acc_lv03_04_frontend_color_convert_.record(ms);
			color_total += ms;
		}

		// Stage 04: preprocess
		result.frame = std::move(read_result.frame);
		result.hw_frame = read_result.hw_frame;
		helmsman::utils::timing::ManualTimer t_prep;
		t_prep.start();
		result.tensor = _PreprocessForInference04(result.frame, model_w, model_h);
		result.preprocess_ms = t_prep.stop();

		result.read_ms = read_total;
		result.decode_ms = decode_total;
		result.color_convert_ms = color_total;
		return result;
	}
}

// ---------------------------------------------------------------------------
// _ProcessMultithread — dual-buffer mode: stages 01-03 on main thread,
//                       stage 04 on worker thread
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontEnd::_ProcessMultithread(int model_w, int model_h) {
	if (mt_eof_)
		return std::nullopt;

	auto read_stages_01_03 = [this](ReadResult& rr, StageTiming& timing) -> bool {
		RawPacket pkt;
		while (true) {
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				if (!_ReadInputSource01(pkt, rr))
					return false;
				double ms = t.stop();
				acc_lv03_02_frontend_read_.record(ms);
				timing.read_ms += ms;
			}
			if (pkt.is_eof)
				return false;
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				bool decoded = _DecodeFrame02(pkt, rr);
				double ms = t.stop();
				acc_lv03_03_frontend_decode_.record(ms);
				timing.decode_ms += ms;
				if (!decoded)
					continue;
			}
			{
				helmsman::utils::timing::ManualTimer t;
				t.start();
				bool ok = _ConvertToBgr03(rr);
				double ms = t.stop();
				acc_lv03_04_frontend_color_convert_.record(ms);
				timing.color_ms += ms;
				return ok;
			}
		}
	};

	if (!mt_started_) {
		// Phase 1: bootstrap
		mt_started_ = true;

		// Stages 01-03: read frame 1
		ReadResult read1;
		StageTiming timing1;
		if (!read_stages_01_03(read1, timing1))
			return std::nullopt;

		cv::Mat frame_1 = std::move(read1.frame);
		stored_hw_frame_ = read1.hw_frame;

		// Launch worker for stage 04
		prefetch_worker_ =
		    std::thread(&FrontEnd::_MultithreadWorkerLoop, this, model_w, model_h);
		raw_ch_->push(frame_1);

		// Stage 04: pop tensor from worker
		helmsman::utils::timing::ManualTimer t_pop1;
		t_pop1.start();
		auto tensor_1 = tensor_ch_->pop();
		if (!tensor_1) {
			mt_eof_ = true;
			return std::nullopt;
		}
		double preprocess_1_ms = t_pop1.stop();

		// Stages 01-03: read frame 2 (overlaps with stage 04 for frame 1)
		ReadResult read2;
		StageTiming timing2;
		if (read_stages_01_03(read2, timing2)) {
			next_frame_ = std::move(read2.frame);
			next_hw_frame_ = read2.hw_frame;
			next_timing_ = timing2;
			raw_ch_->push(next_frame_);
		} else {
			raw_ch_->close();
		}

		FrameResult result;
		result.frame = std::move(frame_1);
		result.hw_frame = stored_hw_frame_;
		result.tensor = std::move(*tensor_1);
		result.read_ms = timing1.read_ms;
		result.decode_ms = timing1.decode_ms;
		result.color_convert_ms = timing1.color_ms;
		result.preprocess_ms = preprocess_1_ms;
		return result;
	}

	// Phase 2: subsequent calls

	// Stage 04: pop tensor from worker
	helmsman::utils::timing::ManualTimer t_pop;
	t_pop.start();
	auto tensor = tensor_ch_->pop();
	if (!tensor) {
		mt_eof_ = true;
		return std::nullopt;
	}
	double preprocess_ms = t_pop.stop();

	// Save current buffered frame to return
	cv::Mat return_frame = std::move(next_frame_);
	HardwareFrame return_hw_frame = next_hw_frame_;
	StageTiming return_timing = next_timing_;

	// Stages 01-03: read next frame
	ReadResult read_next;
	StageTiming next_t;
	if (read_stages_01_03(read_next, next_t)) {
		next_frame_ = std::move(read_next.frame);
		next_hw_frame_ = read_next.hw_frame;
		next_timing_ = next_t;
		raw_ch_->push(next_frame_);
	} else {
		raw_ch_->close();
	}

	FrameResult result;
	result.frame = std::move(return_frame);
	result.hw_frame = return_hw_frame;
	result.tensor = std::move(*tensor);
	result.read_ms = return_timing.read_ms;
	result.decode_ms = return_timing.decode_ms;
	result.color_convert_ms = return_timing.color_ms;
	result.preprocess_ms = preprocess_ms;
	return result;
}

// ---------------------------------------------------------------------------
// Stop — signal prefetch worker to stop
// ---------------------------------------------------------------------------
void FrontEnd::Stop() {
	if (raw_ch_)
		raw_ch_->close();
	if (prefetch_worker_.joinable())
		prefetch_worker_.join();
}

// ---------------------------------------------------------------------------
// ProcessOneFrame — dispatcher: sync or multithread
// ---------------------------------------------------------------------------
std::optional<FrameResult> FrontEnd::ProcessOneFrame(int model_w, int model_h) {
	helmsman::utils::timing::ManualTimer t;
	t.start();
	auto result = multithread_enabled_ ? _ProcessMultithread(model_w, model_h)
	                                   : _ProcessSync(model_w, model_h);
	acc_total_.record(t.stop());
	return result;
}
