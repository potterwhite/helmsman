/*
 * Copyright (c) 2026 PotterWhite
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

#include <rknn_api.h>                                         // rknn_xxx APIs
#include <string>                                             // for std::string
#include "common-define.h"                                    // TensorData
#include "pipeline/inference-engine/base/inference-engine.h"  // InferenceEngine

class InferenceEngineRKNNZeroCP : public InferenceEngine {
   public:
	InferenceEngineRKNNZeroCP();
	~InferenceEngineRKNNZeroCP();

	void load(const std::string& model_path) override;
	TensorData infer(const TensorData& input) override;

	// Get model input dimensions (NHWC format: [batch, height, width, channels])
	int getInputHeight() const { return static_cast<int>(input_attr_.dims[1]); }
	int getInputWidth() const { return static_cast<int>(input_attr_.dims[2]); }

   private:
	// member functions
	void releaseBuffers();

   private:
	// member variables
	rknn_context ctx_{0};

	rknn_input_output_num io_num_;
	rknn_tensor_attr input_attr_;
	rknn_tensor_attr output_attr_;

	rknn_tensor_mem* input_mem_{nullptr};
	rknn_tensor_mem* output_mem_{nullptr};

	size_t input_size_{0};
	size_t output_size_{0};
};
