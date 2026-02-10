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

namespace arcforge {
namespace embedded {
namespace ai_asr {
class WavReader {
   public:
	WavReader();
	~WavReader();

	bool Open(const std::string& filepath, int expected_sample_rate = 16000,
	          int expected_channels = 1);
	void Close();

	size_t ReadSamples(std::vector<float>& out_samples, size_t num_samples_to_read);
	bool IsOpened() const { return is_opened_; }
	bool Eof() const { return eof_ || !wav_file_.is_open(); }

	int GetSampleRate() const { return sample_rate_; }
	int GetChannels() const { return channels_; }
	int GetBitsPerSample() const { return bits_per_sample_; }

   private:
	std::ifstream wav_file_;
	bool is_opened_ = false;
	bool eof_ = false;
	int sample_rate_ = 0;
	int channels_ = 0;
	int bits_per_sample_ = 0;

	std::streamoff data_chunk_pos_ = 0;
	size_t data_chunk_size_ = 0;
	size_t bytes_read_from_data_chunk_ = 0;

	bool ParseWavHeader(int expected_sample_rate, int expected_channels);
};
}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge