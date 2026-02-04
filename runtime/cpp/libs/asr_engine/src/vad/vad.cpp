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

#include "sherpa-onnx/c-api/cxx-api.h"
#include "ASREngine/vad/vad.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

VAD::VAD() {
	std::cout << "VAD object constructed." << std::endl;
}

VAD::~VAD() {
	std::cout << "VAD cleaned up." << std::endl;
	// vad_ unique_ptr will automatically delete the VoiceActivityDetector object
}

bool VAD::Initialize(const VADConfig& user_config) {
	using namespace sherpa_onnx::cxx;

	VadModelConfig config;
	config.silero_vad.model = user_config.getFirstVadModelPath();
	config.silero_vad.threshold = user_config.getSecondSileroThreshold();
	config.silero_vad.min_silence_duration = user_config.getThirdMinSilenceDuration();
	config.silero_vad.min_speech_duration = user_config.getFourthMinSpeechDuration();
	config.silero_vad.window_size =
	    user_config.getSixthWindowSizeSamples();  // Make sure this aligns with model properties

	config.sample_rate = user_config.getFifthSampleRate();
	if (user_config.getEighthDebugLevel() == SherpaDebug::ktrue) {
		config.debug = true;
	} else {
		config.debug = false;
	}

	expected_sample_rate_ = user_config.getFifthSampleRate();
	window_size_samples_ =
	    user_config.getSixthWindowSizeSamples();  // Store for internal buffering logic

	std::cout << "Initializing Sherpa-ONNX VAD with model: " << user_config.getFirstVadModelPath()
	          << std::endl;

	vad_ = std::make_unique<VoiceActivityDetector>(
	    VoiceActivityDetector::Create(config, user_config.getSeventhSpeechBufferSeconds()));

	if (!vad_ || !vad_->Get()) {  // Check both unique_ptr and underlying pointer
		std::cerr << "Failed to create VoiceActivityDetector. Please check your VAD model path and "
		             "config."
		          << std::endl;
		vad_.reset();  // Ensure unique_ptr is cleared if Get() failed post-creation
		return false;
	}

	std::cout << "Sherpa-ONNX VAD created." << std::endl;
	return true;
}

void VAD::Reset() {
	if (vad_ && vad_->Get()) {
		vad_->Clear();  // Clear any buffered speech segments and internal state
		internal_buffer_.clear();
		std::cout << "[VAD Stream Reset]" << std::endl;
	}
}

bool VAD::AcceptWaveform(const float* samples, int num_samples) {
	if (!vad_ || !vad_->Get()) {
		std::cerr << "VAD not initialized." << std::endl;
		return false;
	}

	if (samples == nullptr || num_samples <= 0) {
		// std::cerr << "Warning: Received empty or null audio chunk for VAD processing." << std::endl;
		return true;  // Not an error, just nothing to process
	}

	// Add new samples to the internal buffer
	internal_buffer_.insert(internal_buffer_.end(), samples, samples + num_samples);

	// Process full windows from the internal buffer
	while (internal_buffer_.size() >= static_cast<size_t>(window_size_samples_)) {
		vad_->AcceptWaveform(internal_buffer_.data(), window_size_samples_);
		// Remove processed samples from the beginning of the buffer
		internal_buffer_.erase(internal_buffer_.begin(),
		                       internal_buffer_.begin() + window_size_samples_);
	}
	return true;
}

void VAD::InputFinished() {
	if (vad_ && vad_->Get()) {
		// Process any remaining samples in the internal buffer
		// Silero VAD expects fixed size windows. If remaining samples are less than window_size,
		// they might be ignored or padded depending on the underlying implementation.
		// For simplicity here, we flush what's possible. The VAD's Flush might handle this.
		if (!internal_buffer_.empty()) {
			// If Silero VAD needs padding for the last chunk, that logic might be complex
			// or it might simply process what it can and the VAD Flush() is the main signal.
			// A common approach is to pad with zeros to make a full window if required by the model.
			// For now, let's assume vad->Flush() is sufficient.
			// Or, if you *must* process them, and `AcceptWaveform` *must* take `window_size_samples_`,
			// you would pad `internal_buffer_` with zeros to `window_size_samples_`.
			// This part is tricky with fixed-window VADs if the library doesn't handle partial final windows.
			// The Sherpa-ONNX examples often just call Flush().

			// If internal_buffer_ contains data, but less than a full window,
			// VAD Flush below is what signals the end of stream to the VAD logic.
		}
		internal_buffer_.clear();  // Clear buffer after attempting to flush
		vad_->Flush();
		std::cout << "[VAD Input Finished and Flushed]" << std::endl;
	}
}

bool VAD::IsSpeechSegmentReady() const {
	if (!vad_ || !vad_->Get()) {
		return false;
	}
	return !vad_->IsEmpty();
}

sherpa_onnx::cxx::SpeechSegment VAD::GetNextSpeechSegment() {
	using namespace sherpa_onnx::cxx;
	if (!IsSpeechSegmentReady()) {
		return SpeechSegment{};  // Return an empty segment
	}
	SpeechSegment segment = vad_->Front();
	vad_->Pop();
	return segment;
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge