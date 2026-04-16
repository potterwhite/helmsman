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

#include "Runtime/onnx/onnx.h"
#include "Utils/file/file-utils.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "Utils/math/math-utils.h"
#include "common/data_structure.h"
#include "pipeline/inference-engine/base/inference-engine.h"

class InferenceEngineONNX : public InferenceEngine {

   public:
	InferenceEngineONNX();
	~InferenceEngineONNX();

	void load(const std::string& model_path) override;

	// N-input / M-output inference (general interface).
	// For MODNet: inputs.size()==1, outputs will have 1 element.
	// For RVM:    inputs.size()==5, outputs will have 6 elements.
	void infer(
	    const std::vector<TensorData>& inputs,
	          std::vector<TensorData>& outputs
	) override;

   private:
	Ort::Env env_;
	std::unique_ptr<Ort::Session> session_;

	// Cached names for all inputs and outputs (populated in load()).
	std::vector<std::string> input_names_;
	std::vector<std::string> output_names_;
};
