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

// libs/asr_engine/src/recognizer/recognizer.cpp
#include "ASREngine/recognizer/recognizer.h"
#include "ASREngine/recognizer/impl/recognizer-impl.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

Recognizer::Recognizer() : impl_(std::make_unique<RecognizerImpl>()) {}

Recognizer::~Recognizer() {}

Recognizer::Recognizer(Recognizer&& other) noexcept : impl_(std::move(other.impl_)) {}

Recognizer& Recognizer::operator=(Recognizer&& other) noexcept {
	if (this != &other) {
		impl_ = std::move(other.impl_);
	}
	return *this;
}

bool Recognizer::Initialize(const SherpaConfig& config) {
	if (!impl_) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Recognizer::Initialize called on a moved-from or uninitialized object's PIMPL.",
		    kcurrent_lib_name);

		return false;
	}

	return impl_->Initialize(config);
}

void Recognizer::ProcessAudioChunk(const std::vector<float>& audio_chunk) {
	if (impl_) {
		impl_->ProcessAudioChunk(audio_chunk);
	} else {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Recognizer::ProcessAudioChunk called on a null PIMPL.", kcurrent_lib_name);
		// std::runtime_error("Recognizer not initialized or moved-from");
	}
}

void Recognizer::InputFinished() {
	if (impl_) {
		impl_->InputFinished();
	} else {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Recognizer::InputFinished called on a null PIMPL.", kcurrent_lib_name);
	}
}

std::string Recognizer::GetCurrentText() const {
	if (impl_) {
		return impl_->GetCurrentText();
	}

	arcforge::embedded::utils::Logger::GetInstance().Error(
	    "Recognizer::GetCurrentText called on a null PIMPL.", kcurrent_lib_name);
	return "";
}

bool Recognizer::IsEndpoint() const {
	if (impl_) {
		return impl_->IsEndpoint();
	}

	arcforge::embedded::utils::Logger::GetInstance().Error(
	    "Recognizer::IsEndpoint called on a null PIMPL.", kcurrent_lib_name);
	return false;
}

void Recognizer::ResetStream() {
	if (impl_) {
		impl_->ResetStream();
	} else {

		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Recognizer::ResetStream called on a null PIMPL.", kcurrent_lib_name);
	}
}

int Recognizer::GetExpectedSampleRate() const {
	if (impl_) {
		return impl_->GetExpectedSampleRate();
	}

	arcforge::embedded::utils::Logger::GetInstance().Error(
	    "Recognizer::GetExpectedSampleRate called on a null PIMPL.", kcurrent_lib_name);

	return kdefault_sample_rate;
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge