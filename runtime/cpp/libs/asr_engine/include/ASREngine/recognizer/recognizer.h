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

#include "ASREngine/common/common-types.h"
#include "ASREngine/pch.h"
#include "ASREngine/recognizer/recognizer-config.h"

// #include <memory>
// #include <string>
// #include <vector>

namespace arcforge {
namespace embedded {
namespace ai_asr {

// forward declaration of the PIMPL implementation class
class RecognizerImpl;

class Recognizer {
   public:
	Recognizer();
	~Recognizer();

	bool Initialize(const SherpaConfig& config);
	/*
	 * @brief Synchronously processes a chunk of audio data.
	 * @param audio_chunk A vector of floats representing the audio data.
	 */
	void ProcessAudioChunk(const std::vector<float>& audio_chunk);
	void InputFinished();
	std::string GetCurrentText() const;
	bool IsEndpoint() const;
	void ResetStream();
	int GetExpectedSampleRate() const;

	Recognizer(const Recognizer&) = delete;
	Recognizer& operator=(const Recognizer&) = delete;
	Recognizer(Recognizer&&) noexcept;
	Recognizer& operator=(Recognizer&&) noexcept;

   private:
	std::unique_ptr<RecognizerImpl> impl_;
};

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge