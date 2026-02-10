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
#include "ASREngine/vad/vad-config.h"

// Forward declarations from sherpa-onnx if not fully included by sherpa-pch.h
namespace sherpa_onnx {
namespace cxx {
class VoiceActivityDetector;
struct SpeechSegment;
}  // namespace cxx
}  // namespace sherpa_onnx

namespace arcforge {
namespace embedded {
namespace ai_asr {

// class VADImpl;

class VAD {
   public:
	VAD();
	~VAD();

	// Disallow copy and assign
	VAD(const VAD&) = delete;
	VAD& operator=(const VAD&) = delete;

	bool Initialize(const VADConfig& config);
	void Reset();  // To reset VAD state for a new stream/file

	// Process a chunk of audio data
	// Returns true if successful, false on error
	bool AcceptWaveform(const float* samples, int num_samples);

	// Call when all audio data has been sent
	void InputFinished();

	// Check if there are speech segments ready
	bool IsSpeechSegmentReady() const;

	// Get the next available speech segment
	// Returns an empty segment if none is ready
	sherpa_onnx::cxx::SpeechSegment GetNextSpeechSegment();

	int GetSampleRate() const { return expected_sample_rate_; }

   private:
	std::unique_ptr<sherpa_onnx::cxx::VoiceActivityDetector> vad_;
	int expected_sample_rate_ = 0;
	int window_size_samples_ = 0;         // Window size expected by the VAD model
	std::vector<float> internal_buffer_;  // To buffer audio until a full window can be processed
};

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge