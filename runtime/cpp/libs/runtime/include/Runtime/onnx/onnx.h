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

// #include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "Runtime/common/common-types.h"
#include "Runtime/pch.h"

namespace arcforge {
namespace runtime {

// forward declaration of PIMPL implementation class
class Impl;

static const std::string k_empty_socket_path_sentinel = "";

class RuntimeONNX {
   public:
	static RuntimeONNX& GetInstance();

	explicit RuntimeONNX();
	virtual ~RuntimeONNX();

	// copy constructor and operator
	RuntimeONNX(const RuntimeONNX&) = delete;
	RuntimeONNX& operator=(const RuntimeONNX&) = delete;

	// std::move constructor and operator
	RuntimeONNX(RuntimeONNX&&) noexcept;
	RuntimeONNX& operator=(RuntimeONNX&&) noexcept;

	void show_input(const Ort::Session& session);
	void show_output(const Ort::Session& session);

   private:
	explicit RuntimeONNX(std::unique_ptr<Impl>);
	std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
}  // namespace arcforge