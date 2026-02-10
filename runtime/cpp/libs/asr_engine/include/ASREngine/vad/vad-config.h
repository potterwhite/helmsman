/*
 * Copyright (c) 2025 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "ASREngine/pch.h"
#include "ASREngine/common/common-types.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

// // Define your custom SherpaDebug enum class here
// enum class SherpaDebug { kfalse = 0x10, ktrue = 0x11 };

class VADConfig {
   private:
	// --- Private Member Variables of VADConfig with ordinal prefixes ---
	std::string first_vad_model_path_;
	float second_silero_threshold_;
	float third_min_silence_duration_;
	float fourth_min_speech_duration_;
	int fifth_sample_rate_;
	int sixth_window_size_samples_;
	float seventh_speech_buffer_seconds_;
	SherpaDebug eighth_debug_level_;  // Changed from bool to SherpaDebug

	// --- Private Constructor (Declaration only) ---
	VADConfig(const std::string& model_path, float threshold, float min_silence, float min_speech,
	          int sample_rate, int window_size, float buffer_seconds,
	          SherpaDebug debug_level);  // Changed param type

   public:
	// --- Public Getters (adjusted names and types) ---
	const std::string& getFirstVadModelPath() const { return first_vad_model_path_; }
	float getSecondSileroThreshold() const { return second_silero_threshold_; }
	float getThirdMinSilenceDuration() const { return third_min_silence_duration_; }
	float getFourthMinSpeechDuration() const { return fourth_min_speech_duration_; }
	int getFifthSampleRate() const { return fifth_sample_rate_; }
	int getSixthWindowSizeSamples() const { return sixth_window_size_samples_; }
	float getSeventhSpeechBufferSeconds() const { return seventh_speech_buffer_seconds_; }
	SherpaDebug getEighthDebugLevel() const { return eighth_debug_level_; }  // Changed return type

	// --- Disable Copying and Assignment ---
	VADConfig(const VADConfig&) = delete;
	VADConfig& operator=(const VADConfig&) = delete;

	// --- Allow Moving ---
	VADConfig(VADConfig&&) = default;
	VADConfig& operator=(VADConfig&&) = default;

	// --- The Builder Class (Nested Public Class) ---
	class Builder {
	   private:
		// Builder's temporary storage with ordinal prefixes
		std::string b_first_vad_model_path_;
		float b_second_silero_threshold_;
		float b_third_min_silence_duration_;
		float b_fourth_min_speech_duration_;
		int b_fifth_sample_rate_;
		int b_sixth_window_size_samples_;
		float b_seventh_speech_buffer_seconds_;
		SherpaDebug b_eighth_debug_level_;  // Changed from bool to SherpaDebug

	   public:
		// --- Builder Constructor (Declaration only) ---
		Builder();

		// --- Builder's Setters (Declarations only, adjusted names and types) ---
		Builder& setFirstVadModelPath(const std::string& path);
		Builder& setSecondSileroThreshold(float threshold);
		Builder& setThirdMinSilenceDuration(float duration);
		Builder& setFourthMinSpeechDuration(float duration);
		Builder& setFifthSampleRate(int rate);
		Builder& setSixthWindowSizeSamples(int samples);
		Builder& setSeventhSpeechBufferSeconds(float seconds);
		Builder& setEighthDebugLevel(SherpaDebug level);  // Changed param type

		// Helper to initialize builder from an existing config (if needed later)
		// Builder& fromConfig(const VADConfig& existingConfig);

		// --- The Build Method (Declaration only) ---
		VADConfig build();
	};  // End of Builder class
};      // End of VADConfig class

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge