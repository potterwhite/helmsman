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

#include "Network/pch.h"

namespace arcforge {
namespace embedded {
namespace network_socket {


// inline constexpr int killegal_fd_value = -1;

class SherpaConfig {
   private:
	// --- Private Member Variables of SherpaConfig with ordinal prefixes ---
	std::string first_encoder_path_;
	std::string second_decoder_path_;
	std::string third_joiner_path_;
	std::string fourth_tokens_path_;
	std::string fifth_provider_;
	int sixth_num_threads_;
	float seventh_rule1_min_trailing_silence_;
	float eighth_rule2_min_trailing_silence_;
	float ninth_rule3_min_utterance_length_;
	std::string tenth_decoding_method_;
	SherpaDebug eleventh_debug_level_;
	SherpaEndPointSupport twelfth_enable_endpoint_detection_;

	// --- Private Constructor (Declaration only) ---
	SherpaConfig(const std::string& enc_path, const std::string& dec_path,
	             const std::string& join_path, const std::string& tok_path,
	             const std::string& provider, int num_threads, float rule1, float rule2,
	             float rule3, const std::string& dec_method, SherpaDebug debug,
	             SherpaEndPointSupport endpoint_detection);

   public:
	// --- Public Getters (adjusted names) ---
	const std::string& getFirstEncoderPath() const { return first_encoder_path_; }
	const std::string& getSecondDecoderPath() const { return second_decoder_path_; }
	const std::string& getThirdJoinerPath() const { return third_joiner_path_; }
	const std::string& getFourthTokensPath() const { return fourth_tokens_path_; }
	const std::string& getFifthProvider() const { return fifth_provider_; }
	int getSixthNumThreads() const { return sixth_num_threads_; }
	float getSeventhRule1MinTrailingSilence() const { return seventh_rule1_min_trailing_silence_; }
	float getEighthRule2MinTrailingSilence() const { return eighth_rule2_min_trailing_silence_; }
	float getNinthRule3MinUtteranceLength() const { return ninth_rule3_min_utterance_length_; }
	const std::string& getTenthDecodingMethod() const { return tenth_decoding_method_; }
	SherpaDebug getEleventhDebugLevel() const { return eleventh_debug_level_; }
	SherpaEndPointSupport getTwelfthEndpointDetectionSupport() const {
		return twelfth_enable_endpoint_detection_;
	}

	// --- Disable Copying and Assignment ---
	SherpaConfig(const SherpaConfig&) = delete;
	SherpaConfig& operator=(const SherpaConfig&) = delete;

	// --- Allow Moving ---
	SherpaConfig(SherpaConfig&&) = default;
	SherpaConfig& operator=(SherpaConfig&&) = default;

	// --- The Builder Class (Nested Public Class) ---
	class Builder {
	   private:
		// Builder's temporary storage with ordinal prefixes
		std::string b_first_encoder_path_;
		std::string b_second_decoder_path_;
		std::string b_third_joiner_path_;
		std::string b_fourth_tokens_path_;
		std::string b_fifth_provider_;
		int b_sixth_num_threads_;
		float b_seventh_rule1_min_trailing_silence_;
		float b_eighth_rule2_min_trailing_silence_;
		float b_ninth_rule3_min_utterance_length_;
		std::string b_tenth_decoding_method_;
		SherpaDebug b_eleventh_debug_level_;
		SherpaEndPointSupport b_twelfth_enable_endpoint_detection_;

	   public:
		// --- Builder Constructor (Declaration only) ---
		Builder();

		// --- Builder's Setters (Declarations only, adjusted names) ---
		Builder& setFirstEncoderPath(const std::string& path);
		Builder& setSecondDecoderPath(const std::string& path);
		Builder& setThirdJoinerPath(const std::string& path);
		Builder& setFourthTokensPath(const std::string& path);
		Builder& setFifthProvider(const std::string& provider);
		Builder& setSixthNumThreads(int threads);
		Builder& setSeventhRule1MinTrailingSilence(float silence);
		Builder& setEighthRule2MinTrailingSilence(float silence);
		Builder& setNinthRule3MinUtteranceLength(float length);
		Builder& setTenthDecodingMethod(const std::string& method);
		Builder& setEleventhDebugLevel(SherpaDebug level);
		Builder& setTwelfthEndpointDetectionSupport(SherpaEndPointSupport enable);

		// Helper to initialize builder from an existing config
		Builder& fromConfig(const SherpaConfig& existingConfig);

		// --- The Build Method (Declaration only) ---
		SherpaConfig build();
	};  // End of Builder class
};      // End of SherpaConfig class

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge