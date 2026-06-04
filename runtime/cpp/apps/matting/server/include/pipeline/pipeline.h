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

#include <memory>
#include <string>
#include "Utils/logger/logger.h"
#include "common/types.h"
#include "pipeline/modes/modnet/modnet.h"
#include "pipeline/modes/rvm/rvm.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend/frontend-core/frontend.h"
#include "pipeline/stages/inference-engine/engine-core/inference-engine.h"

class Pipeline {
   public:
	static Pipeline& GetInstance();

	void Init(const AppConfig& config);

	int Run();

   private:
	Pipeline();
	~Pipeline();

	void VerifyParametersNecessary();

   private:
	AppConfig config_;
	std::unique_ptr<FrontEnd> frontend_;
	std::unique_ptr<InferenceEngine> engine_;
	std::unique_ptr<BackEnd> backend_;

	MODNetMode modnet_mode_;
	RVMMode rvm_mode_;
};