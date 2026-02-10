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

#include "ASREngine/recognizer/recognizer-config.h"
#include "sherpa-onnx/c-api/cxx-api.h"

namespace arcforge {
namespace embedded {
namespace ai_asr {

// --- Implementation of SherpaConfig's Private Constructor ---
SherpaConfig::SherpaConfig(const std::string& enc_path, const std::string& dec_path,
                           const std::string& join_path, const std::string& tok_path,
                           const std::string& provider, int num_threads, float rule1, float rule2,
                           float rule3, const std::string& dec_method, SherpaDebug debug,
                           SherpaEndPointSupport endpoint_detection)
    : first_encoder_path_(enc_path),                          // Adjusted member name
      second_decoder_path_(dec_path),                         // Adjusted member name
      third_joiner_path_(join_path),                          // Adjusted member name
      fourth_tokens_path_(tok_path),                          // Adjusted member name
      fifth_provider_(provider),                              // Adjusted member name
      sixth_num_threads_(num_threads),                        // Adjusted member name
      seventh_rule1_min_trailing_silence_(rule1),             // Adjusted member name
      eighth_rule2_min_trailing_silence_(rule2),              // Adjusted member name
      ninth_rule3_min_utterance_length_(rule3),               // Adjusted member name
      tenth_decoding_method_(dec_method),                     // Adjusted member name
      eleventh_debug_level_(debug),                           // Adjusted member name
      twelfth_enable_endpoint_detection_(endpoint_detection)  // Adjusted member name
{
	// Constructor body
}

// --- Implementation of SherpaConfig::Builder Constructor ---
SherpaConfig::Builder::Builder()
    : b_first_encoder_path_(""),
      b_second_decoder_path_(""),
      b_third_joiner_path_(""),
      b_fourth_tokens_path_(""),
      b_fifth_provider_("cpu"),
      b_sixth_num_threads_(1),
      b_seventh_rule1_min_trailing_silence_(2.4f),
      b_eighth_rule2_min_trailing_silence_(1.2f),
      b_ninth_rule3_min_utterance_length_(20.0f),
      b_tenth_decoding_method_("greedy_search"),
      b_eleventh_debug_level_(SherpaDebug::kfalse),
      b_twelfth_enable_endpoint_detection_(SherpaEndPointSupport::kdisable) {
	// Builder constructor body
}

// --- Implementation of SherpaConfig::Builder Setters (adjusted names) ---
SherpaConfig::Builder& SherpaConfig::Builder::setFirstEncoderPath(const std::string& path) {
	b_first_encoder_path_ = path;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setSecondDecoderPath(const std::string& path) {
	b_second_decoder_path_ = path;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setThirdJoinerPath(const std::string& path) {
	b_third_joiner_path_ = path;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setFourthTokensPath(const std::string& path) {
	b_fourth_tokens_path_ = path;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setFifthProvider(const std::string& provider) {
	b_fifth_provider_ = provider;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setSixthNumThreads(int threads) {
	// --num-threads=1 to select RKNN_NPU_CORE_AUTO
	// --num-threads=0 to select RKNN_NPU_CORE_0
	// --num-threads=-1 to select RKNN_NPU_CORE_1
	// --num-threads=-2 to select RKNN_NPU_CORE_2
	// --num-threads=-3 to select RKNN_NPU_CORE_0_1
	// --num-threads=-4 to select RKNN_NPU_CORE_0_1_2
	if (threads < -4) {
		threads = 1;
	}
	b_sixth_num_threads_ = threads;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setSeventhRule1MinTrailingSilence(float silence) {
	b_seventh_rule1_min_trailing_silence_ = silence;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setEighthRule2MinTrailingSilence(float silence) {
	b_eighth_rule2_min_trailing_silence_ = silence;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setNinthRule3MinUtteranceLength(float length) {
	b_ninth_rule3_min_utterance_length_ = length;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setTenthDecodingMethod(const std::string& method) {
	b_tenth_decoding_method_ = method;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setEleventhDebugLevel(SherpaDebug level) {
	b_eleventh_debug_level_ = level;
	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::setTwelfthEndpointDetectionSupport(
    SherpaEndPointSupport ep_support) {
	b_twelfth_enable_endpoint_detection_ = ep_support;

	return *this;
}

SherpaConfig::Builder& SherpaConfig::Builder::fromConfig(const SherpaConfig& existingConfig) {
	this->b_first_encoder_path_ = existingConfig.getFirstEncoderPath();
	this->b_second_decoder_path_ = existingConfig.getSecondDecoderPath();
	this->b_third_joiner_path_ = existingConfig.getThirdJoinerPath();
	this->b_fourth_tokens_path_ = existingConfig.getFourthTokensPath();
	this->b_fifth_provider_ = existingConfig.getFifthProvider();
	this->b_sixth_num_threads_ = existingConfig.getSixthNumThreads();
	this->b_seventh_rule1_min_trailing_silence_ =
	    existingConfig.getSeventhRule1MinTrailingSilence();
	this->b_eighth_rule2_min_trailing_silence_ = existingConfig.getEighthRule2MinTrailingSilence();
	this->b_ninth_rule3_min_utterance_length_ = existingConfig.getNinthRule3MinUtteranceLength();
	this->b_tenth_decoding_method_ = existingConfig.getTenthDecodingMethod();
	this->b_eleventh_debug_level_ = existingConfig.getEleventhDebugLevel();
	this->b_twelfth_enable_endpoint_detection_ =
	    existingConfig.getTwelfthEndpointDetectionSupport();
	return *this;
}

// --- Implementation of SherpaConfig::Builder::build() ---
SherpaConfig SherpaConfig::Builder::build() {
	// Validation for MANDATORY fields (adjusting for new names)
	if (b_first_encoder_path_.empty()) {
		throw std::runtime_error(
		    "SherpaConfig Build Error: First encoder path is mandatory and was not set.");
	}
	if (b_second_decoder_path_.empty()) {
		throw std::runtime_error(
		    "SherpaConfig Build Error: Second decoder path is mandatory and was not set.");
	}
	if (b_third_joiner_path_.empty()) {
		throw std::runtime_error(
		    "SherpaConfig Build Error: Third joiner path is mandatory and was not set.");
	}
	if (b_fourth_tokens_path_.empty()) {
		throw std::runtime_error(
		    "SherpaConfig Build Error: Fourth tokens path is mandatory and was not set.");
	}

	// More specific validations
	if (b_fifth_provider_ != "cpu" && b_fifth_provider_ != "rknn" /* add others */) {
		std::cerr << "Warning: Provider '" << b_fifth_provider_
		          << "' might not be supported or is unknown. Proceeding with the value."
		          << std::endl;
	}
	if (b_seventh_rule1_min_trailing_silence_ < 0.0f ||
	    b_eighth_rule2_min_trailing_silence_ < 0.0f || b_ninth_rule3_min_utterance_length_ < 0.0f) {
		throw std::runtime_error(
		    "SherpaConfig Build Error: Rule silence/length values must be non-negative.");
	}

	return SherpaConfig(b_first_encoder_path_, b_second_decoder_path_, b_third_joiner_path_,
	                    b_fourth_tokens_path_, b_fifth_provider_, b_sixth_num_threads_,
	                    b_seventh_rule1_min_trailing_silence_, b_eighth_rule2_min_trailing_silence_,
	                    b_ninth_rule3_min_utterance_length_, b_tenth_decoding_method_,
	                    b_eleventh_debug_level_, b_twelfth_enable_endpoint_detection_);
}

}  // namespace ai_asr
}  // namespace embedded
}  // namespace arcforge