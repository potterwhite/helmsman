// Copyright (c) 2025 PotterWhite
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

// libs/asr_engine/src/recognizer/impl/recognizer-impl.cpp
#include "ASREngine/recognizer/impl/recognizer-impl.h"
#include "Utils/logger/logger.h"

#include "sherpa-onnx/c-api/cxx-api.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

using namespace sherpa_onnx::cxx;

RecognizerImpl::RecognizerImpl() {
	arcforge::embedded::utils::Logger::GetInstance().Info("RecognizerImpl object constructed.",
	                                                      kcurrent_lib_name);
}

RecognizerImpl::~RecognizerImpl() {
	// std::ostringstream oss;
	// oss << "RecognizerImpl cleaned up.";

	arcforge::embedded::utils::Logger::GetInstance().Info("RecognizerImpl cleaned up.",
	                                                      kcurrent_lib_name);
}

RecognizerImpl::RecognizerImpl(RecognizerImpl&& other) noexcept
    : recognizer_ptr_(std::move(other.recognizer_ptr_)),
      stream_ptr_(std::move(other.stream_ptr_)),
      last_displayed_text_(std::move(other.last_displayed_text_)),
      expected_sample_rate_(other.expected_sample_rate_) {}

RecognizerImpl& RecognizerImpl::operator=(RecognizerImpl&& other) noexcept {
	if (this != &other) {
		recognizer_ptr_ = std::move(other.recognizer_ptr_);
		stream_ptr_ = std::move(other.stream_ptr_);
		last_displayed_text_ = std::move(other.last_displayed_text_);
		expected_sample_rate_ = other.expected_sample_rate_;
	}
	return *this;
}

bool RecognizerImpl::Initialize(const SherpaConfig& sherpa_config) {
	OnlineRecognizerConfig config;

	config.model_config.transducer.encoder = sherpa_config.getFirstEncoderPath();
	config.model_config.transducer.decoder = sherpa_config.getSecondDecoderPath();
	config.model_config.transducer.joiner = sherpa_config.getThirdJoinerPath();
	config.model_config.tokens = sherpa_config.getFourthTokensPath();
	config.model_config.provider = sherpa_config.getFifthProvider();
	config.model_config.num_threads = sherpa_config.getSixthNumThreads();

	if (sherpa_config.getEleventhDebugLevel() == SherpaDebug::ktrue) {
		config.model_config.debug = true;
	} else {
		config.model_config.debug = false;
	}

	config.feat_config.sample_rate = expected_sample_rate_;

	config.rule1_min_trailing_silence = sherpa_config.getSeventhRule1MinTrailingSilence();
	config.rule2_min_trailing_silence = sherpa_config.getEighthRule2MinTrailingSilence();
	config.rule3_min_utterance_length = sherpa_config.getNinthRule3MinUtteranceLength();
	config.decoding_method = sherpa_config.getTenthDecodingMethod();
	if (sherpa_config.getTwelfthEndpointDetectionSupport() == SherpaEndPointSupport::kenable) {
		config.enable_endpoint = true;
	} else {
		config.enable_endpoint = false;
	}

	std::ostringstream oss;
	oss << "Initializing Sherpa-ONNX (Impl) with provider: " << sherpa_config.getFifthProvider();
	arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_lib_name);

	try {
		/*********************************************************
		 * I. Create recognizer_ptr_
		 *********************************************************/
		recognizer_ptr_ = std::make_unique<OnlineRecognizer>(OnlineRecognizer::Create(config));

		if (recognizer_ptr_->Get() == nullptr) {
			// std::cerr << "Failed to create OnlineRecognizer (internal pointer is null).";
			arcforge::embedded::utils::Logger::GetInstance().Error(
			    "Failed to create OnlineRecognizer (internal pointer is null).", kcurrent_lib_name);

			recognizer_ptr_.reset();

			return false;
		}

		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "Sherpa-ONNX Recognizer (Impl) created.", kcurrent_lib_name);

		/*********************************************************
		 * II. Create Stream object
		 *********************************************************/
		stream_ptr_ = std::make_unique<OnlineStream>(recognizer_ptr_->CreateStream());
		if (stream_ptr_->Get() == nullptr) {
			// std::cerr << "Failed to create OnlineStream (internal pointer is null).";
			arcforge::embedded::utils::Logger::GetInstance().Error(
			    "Failed to create OnlineStream (internal pointer is null).", kcurrent_lib_name);

			recognizer_ptr_.reset();
			stream_ptr_.reset();

			return false;
		}

		arcforge::embedded::utils::Logger::GetInstance().Info("Sherpa-ONNX Stream (Impl) created.",
		                                                      kcurrent_lib_name);

	} catch (const std::exception& e) {
		std::ostringstream oss_catch;
		oss_catch << "Exception during RecognizerImpl::Initialize: " << e.what();
		arcforge::embedded::utils::Logger::GetInstance().Error(oss_catch.str(), kcurrent_lib_name);

		stream_ptr_.reset();
		return false;
	}
	return true;
}

void RecognizerImpl::ProcessAudioChunk(const std::vector<float>& audio_chunk) {
	if (!stream_ptr_ || !recognizer_ptr_) {
		arcforge::embedded::utils::Logger::GetInstance().Error("ASR (Impl) not initialized.",
		                                                       kcurrent_lib_name);
		return;
	}
	if (audio_chunk.empty()) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Warning: Received empty audio chunk (Impl).", kcurrent_lib_name);
		return;
	}

	stream_ptr_->AcceptWaveform(expected_sample_rate_, audio_chunk.data(),
	                            static_cast<int32_t>(audio_chunk.size()));
	// stream_ptr_->InputFinished();

	while (recognizer_ptr_->IsReady(stream_ptr_.get())) {
		recognizer_ptr_->Decode(stream_ptr_.get());
	}
}

void RecognizerImpl::InputFinished() {
	if (stream_ptr_ && recognizer_ptr_) {
		stream_ptr_->InputFinished();
		while (recognizer_ptr_->IsReady(stream_ptr_.get())) {
			recognizer_ptr_->Decode(stream_ptr_.get());
		}
	}
}

std::string RecognizerImpl::GetCurrentText() {
	if (!recognizer_ptr_ || !stream_ptr_) {
		return "";
	}
	OnlineRecognizerResult result = recognizer_ptr_->GetResult(stream_ptr_.get());
#if 1
	last_displayed_text_ = result.text;

	return last_displayed_text_;
#else
	return result.text;
#endif
}

bool RecognizerImpl::IsEndpoint() const {
	if (!recognizer_ptr_ || !stream_ptr_) {
		return false;
	}
	return recognizer_ptr_->IsEndpoint(stream_ptr_.get());
}

void RecognizerImpl::ResetStream() {
	if (recognizer_ptr_ && stream_ptr_) {

		recognizer_ptr_->Reset(stream_ptr_.get());
		last_displayed_text_.clear();

		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "[ASR Stream Reset (Impl) for new utterance]", kcurrent_lib_name);

	} else {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Cannot reset stream, recognizer or stream not initialized.", kcurrent_lib_name);
	}
}

int RecognizerImpl::GetExpectedSampleRate() const {
	return expected_sample_rate_;
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge