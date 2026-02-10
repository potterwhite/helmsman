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
#include "ASREngine/vad/vad-config.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

// --- Implementation of VADConfig's Private Constructor ---
VADConfig::VADConfig(const std::string& model_path, float threshold, float min_silence,
                     float min_speech, int sample_rate, int window_size, float buffer_seconds,
                     SherpaDebug debug_level)  // Param type changed
    : first_vad_model_path_(model_path),
      second_silero_threshold_(threshold),
      third_min_silence_duration_(min_silence),
      fourth_min_speech_duration_(min_speech),
      fifth_sample_rate_(sample_rate),
      sixth_window_size_samples_(window_size),
      seventh_speech_buffer_seconds_(buffer_seconds),
      eighth_debug_level_(debug_level)  // Member type changed
{
	// Constructor body
}

// --- Implementation of VADConfig::Builder Constructor ---
VADConfig::Builder::Builder()
    : b_first_vad_model_path_(""),
      b_second_silero_threshold_(0.5f),
      b_third_min_silence_duration_(0.25f),
      b_fourth_min_speech_duration_(0.1f),
      b_fifth_sample_rate_(16000),
      b_sixth_window_size_samples_(512),
      b_seventh_speech_buffer_seconds_(30.0f),
      b_eighth_debug_level_(SherpaDebug::kfalse)  // Default value changed to enum
{
	// Builder constructor body
}

// --- Implementation of VADConfig::Builder Setters ---
VADConfig::Builder& VADConfig::Builder::setFirstVadModelPath(const std::string& path) {
	b_first_vad_model_path_ = path;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setSecondSileroThreshold(float threshold) {
	b_second_silero_threshold_ = threshold;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setThirdMinSilenceDuration(float duration) {
	b_third_min_silence_duration_ = duration;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setFourthMinSpeechDuration(float duration) {
	b_fourth_min_speech_duration_ = duration;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setFifthSampleRate(int rate) {
	b_fifth_sample_rate_ = rate;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setSixthWindowSizeSamples(int samples) {
	b_sixth_window_size_samples_ = samples;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setSeventhSpeechBufferSeconds(float seconds) {
	b_seventh_speech_buffer_seconds_ = seconds;
	return *this;
}

VADConfig::Builder& VADConfig::Builder::setEighthDebugLevel(
    SherpaDebug level) {  // Param type changed
	b_eighth_debug_level_ = level;
	return *this;
}

// --- Implementation of VADConfig::Builder::build() ---
VADConfig VADConfig::Builder::build() {
	if (b_first_vad_model_path_.empty()) {
		throw std::runtime_error(
		    "VADConfig Build Error: First VAD model path is mandatory and was not set.");
	}
	if (b_fifth_sample_rate_ <= 0) {
		throw std::runtime_error("VADConfig Build Error: Fifth sample rate must be positive.");
	}
	if (b_sixth_window_size_samples_ <= 0) {
		throw std::runtime_error(
		    "VADConfig Build Error: Sixth window size samples must be positive.");
	}
	if (b_seventh_speech_buffer_seconds_ <= 0) {
		throw std::runtime_error(
		    "VADConfig Build Error: Seventh speech buffer seconds must be positive.");
	}
	// No specific validation needed for b_eighth_debug_level_ as it's an enum
	// unless you want to restrict it to specific values beyond what the enum defines.

	return VADConfig(b_first_vad_model_path_, b_second_silero_threshold_,
	                 b_third_min_silence_duration_, b_fourth_min_speech_duration_,
	                 b_fifth_sample_rate_, b_sixth_window_size_samples_,
	                 b_seventh_speech_buffer_seconds_,
	                 b_eighth_debug_level_  // Argument type changed
	);
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge