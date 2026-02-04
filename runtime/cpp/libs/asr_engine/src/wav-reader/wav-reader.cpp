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

#include "ASREngine/wav-reader/wav-reader.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

// Helper to read little-endian values
template <typename T>
T read_le(std::istream& is) {
	T val;
	is.read(reinterpret_cast<char*>(&val), sizeof(T));
	return val;
}

WavReader::WavReader() {}

WavReader::~WavReader() {
	Close();
}

bool WavReader::Open(const std::string& filepath, int expected_sample_rate, int expected_channels) {
	if (is_opened_) {
		Close();
	}
	wav_file_.open(filepath, std::ios::binary);
	if (!wav_file_) {
		std::ostringstream oss;
		oss << "Error opening WAV file: " << filepath;
		arcforge::embedded::utils::Logger::GetInstance().Error(oss.str(), kcurrent_lib_name);
		return false;
	}

	if (!ParseWavHeader(expected_sample_rate, expected_channels)) {
		wav_file_.close();
		return false;
	}

	wav_file_.seekg(data_chunk_pos_, std::ios::beg);

	if (!wav_file_) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Error seeking to data chunk in WAV file.", kcurrent_lib_name);
		wav_file_.close();
		return false;
	}
	bytes_read_from_data_chunk_ = 0;
	is_opened_ = true;
	eof_ = false;

	std::ostringstream oss;
	oss << "WAV file opened: " << filepath << "\n"
	    << "  Sample Rate: " << sample_rate_ << " Hz (Expected: " << expected_sample_rate << ")\n"
	    << "  Channels: " << channels_ << " (Expected: " << expected_channels << ")\n"
	    << "  Bits/Sample: " << bits_per_sample_ << "\n"
	    << "  Data chunk size: " << data_chunk_size_ << " bytes";
	arcforge::embedded::utils::Logger::GetInstance().MultiLineLog(
	    arcforge::embedded::utils::LoggerLevel::kinfo, oss.str(), kcurrent_lib_name);

	return true;
}

void WavReader::Close() {
	if (wav_file_.is_open()) {
		wav_file_.close();
		arcforge::embedded::utils::Logger::GetInstance().Info("WAV file closed.",
		                                                      kcurrent_lib_name);
	}
	is_opened_ = false;
	eof_ = true;
}

bool WavReader::ParseWavHeader(int expected_sample_rate, int expected_channels) {
	char buffer[4];
	// RIFF chunk
	wav_file_.read(buffer, 4);
	if (!wav_file_ || std::memcmp(buffer, "RIFF", 4) != 0) {
		arcforge::embedded::utils::Logger::GetInstance().Error("Not a RIFF file or read error",
		                                                       kcurrent_lib_name);
		return false;
	}
	read_le<uint32_t>(wav_file_);
	wav_file_.read(buffer, 4);
	if (!wav_file_ || std::memcmp(buffer, "WAVE", 4) != 0) {
		arcforge::embedded::utils::Logger::GetInstance().Error("Not a WAVE file or read error",
		                                                       kcurrent_lib_name);
		return false;
	}

	bool fmt_found = false;
	while (wav_file_.read(buffer, 4)) {
		uint32_t sub_chunk_size = read_le<uint32_t>(wav_file_);
		if (!wav_file_) {
			arcforge::embedded::utils::Logger::GetInstance().Error("Read error after sub-chunk ID",
			                                                       kcurrent_lib_name);
			return false;
		}

		if (std::memcmp(buffer, "fmt ", 4) == 0) {
			fmt_found = true;
			uint16_t audio_format = read_le<uint16_t>(wav_file_);
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error("Read error (audio_format)",
				                                                       kcurrent_lib_name);
				return false;
			}
			channels_ = static_cast<int>(read_le<uint16_t>(wav_file_));
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error("Read error (channels)",
				                                                       kcurrent_lib_name);
				return false;
			}

			uint32_t sr_u32 = read_le<uint32_t>(wav_file_);
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error("Read error (sample_rate)",
				                                                       kcurrent_lib_name);
				return false;
			}
			if (sr_u32 > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "Sample rate value too large for int type.", kcurrent_lib_name);
				return false;
			}
			sample_rate_ = static_cast<int>(sr_u32);

			read_le<uint32_t>(wav_file_);  // Byte rate
			read_le<uint16_t>(wav_file_);  // Block align
			bits_per_sample_ = static_cast<int>(read_le<uint16_t>(wav_file_));
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "Read error (bits_per_sample)", kcurrent_lib_name);
				return false;
			}

			if (audio_format != 1) {
				arcforge::embedded::utils::Logger::GetInstance().Error("Not PCM format in WAV",
				                                                       kcurrent_lib_name);
				return false;
			}
			if (sample_rate_ != expected_sample_rate) {
				std::ostringstream oss;
				oss << "WAV sample rate (" << sample_rate_ << ") differs from expected ("
				    << expected_sample_rate << "). Output might be incorrect.";
				arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(),
				                                                         kcurrent_lib_name);
			}
			if (channels_ <= 0) {
				std::ostringstream oss;
				oss << "Invalid number of channels: " << channels_;
				arcforge::embedded::utils::Logger::GetInstance().Error(oss.str(),
				                                                       kcurrent_lib_name);
				return false;
			}
			if (channels_ != expected_channels && expected_channels != 0) {
				std::ostringstream oss;
				oss << "WAV channels (" << channels_ << ") differs from expected ("
				    << expected_channels << "). Will use first channel if multi-channel.";
				arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(),
				                                                         kcurrent_lib_name);
			}
			if (bits_per_sample_ != 16) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "WAV not 16-bit. Only 16-bit PCM is supported.", kcurrent_lib_name);
				return false;
			}

			if (sub_chunk_size > 16) {  // 16 is sizeof(common PCM fmt sub-chunk data part)
				std::streamoff offset = static_cast<std::streamoff>(sub_chunk_size) - 16;
				wav_file_.seekg(offset, std::ios::cur);
				if (!wav_file_) {
					arcforge::embedded::utils::Logger::GetInstance().Error(
					    "Seek error past fmt chunk", kcurrent_lib_name);
					return false;
				}
			}
			break;
		} else {
			wav_file_.seekg(static_cast<std::streamoff>(sub_chunk_size), std::ios::cur);
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "Seek error past other chunk", kcurrent_lib_name);
				return false;
			}
		}
	}
	if (!fmt_found) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "\"fmt \" chunk not found or read error before finding it.", kcurrent_lib_name);
		return false;
	}

	bool data_found = false;
	while (wav_file_.read(buffer, 4)) {
		uint32_t sub_chunk_size_data_u32 = read_le<uint32_t>(wav_file_);
		if (!wav_file_) {
			arcforge::embedded::utils::Logger::GetInstance().Error(
			    "Read error after data sub-chunk ID", kcurrent_lib_name);
			return false;
		}

		if (std::memcmp(buffer, "data", 4) == 0) {
			data_found = true;

			data_chunk_pos_ = wav_file_.tellg();

			data_chunk_size_ = static_cast<size_t>(sub_chunk_size_data_u32);  // uint32_t to size_t
			break;
		} else {
			wav_file_.seekg(static_cast<std::streamoff>(sub_chunk_size_data_u32), std::ios::cur);
			if (!wav_file_) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "Seek error past other chunk (while searching data)", kcurrent_lib_name);
				return false;
			}
		}
	}
	if (!data_found) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "\"data\" chunk not found or read error before finding it.", kcurrent_lib_name);
		return false;
	}
	return true;
}

size_t WavReader::ReadSamples(std::vector<float>& out_samples, size_t num_samples_to_read) {
	if (!is_opened_ || eof_) {
		out_samples.clear();
		return 0;
	}
	if (bits_per_sample_ == 0 || channels_ == 0) {  //Defensive division by zero
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Error: bits_per_sample or channels is zero.", kcurrent_lib_name);
		out_samples.clear();
		eof_ = true;
		return 0;
	}

	size_t bytes_per_sample_all_channels = static_cast<size_t>(bits_per_sample_ / 8 * channels_);
	if (bytes_per_sample_all_channels == 0) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Error: bytes_per_sample_all_channels is zero.", kcurrent_lib_name);
		out_samples.clear();
		eof_ = true;
		return 0;
	}

	size_t max_samples_remaining_in_chunk = 0;
	if (data_chunk_size_ >= bytes_read_from_data_chunk_) {
		max_samples_remaining_in_chunk =
		    (data_chunk_size_ - bytes_read_from_data_chunk_) / bytes_per_sample_all_channels;
	} else {
		// Should not happen if logic is correct, but defensive.
		arcforge::embedded::utils::Logger::GetInstance().Warning(
		    "Warning: bytes_read_from_data_chunk_ exceeds data_chunk_size_.", kcurrent_lib_name);
	}

	size_t actual_samples_to_read = std::min(num_samples_to_read, max_samples_remaining_in_chunk);

	if (actual_samples_to_read == 0) {
		eof_ = true;
		out_samples.clear();
		return 0;
	}

	out_samples.resize(actual_samples_to_read);
	std::vector<int16_t> temp_s16_buffer(actual_samples_to_read * static_cast<size_t>(channels_));

	size_t bytes_to_read = actual_samples_to_read * bytes_per_sample_all_channels;

	wav_file_.read(reinterpret_cast<char*>(temp_s16_buffer.data()),
	               static_cast<std::streamsize>(bytes_to_read));

	if (!wav_file_ && !wav_file_.eof()) {
		arcforge::embedded::utils::Logger::GetInstance().Error("Error reading WAV data.",
		                                                       kcurrent_lib_name);
		out_samples.clear();
		eof_ = true;
		return 0;
	}

	size_t bytes_just_read = static_cast<size_t>(wav_file_.gcount());
	size_t samples_just_read_per_channel = 0;
	if (bytes_per_sample_all_channels > 0) {
		samples_just_read_per_channel = bytes_just_read / bytes_per_sample_all_channels;
	}

	bytes_read_from_data_chunk_ += bytes_just_read;

	if (samples_just_read_per_channel < actual_samples_to_read) {
		eof_ = true;
		out_samples.resize(samples_just_read_per_channel);
		if (samples_just_read_per_channel == 0 && bytes_just_read > 0) {
			std::ostringstream oss;
			oss << "Warning: Read 0 samples but " << bytes_just_read
			    << " bytes. Possible partial sample read or end of data chunk.";
			arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(), kcurrent_lib_name);
		}
	}

	if (static_cast<size_t>(channels_) == 0) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Error: channels_ is zero before processing samples.", kcurrent_lib_name);
		out_samples.clear();
		return 0;
	}

	for (size_t i = 0; i < samples_just_read_per_channel; ++i) {

		out_samples[i] =
		    static_cast<float>(temp_s16_buffer[i * static_cast<size_t>(channels_)]) / 32768.0f;
	}

	if (bytes_read_from_data_chunk_ >= data_chunk_size_) {
		eof_ = true;
	}
	if (wav_file_.eof()) {
		eof_ = true;
	}

	return samples_just_read_per_channel;
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge