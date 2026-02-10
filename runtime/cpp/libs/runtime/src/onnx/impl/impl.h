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

// include/RuntimeONNX/base/impl/base-impl.h
#pragma once

// #include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "Runtime/common/common-types.h"
#include "Runtime/pch.h"

namespace arcforge {
namespace runtime {

inline constexpr int killegal_fd_value = -1;

class Impl;

class Impl {
   public:
	Impl();
	~Impl();

	// forbitd copy and assignment
	Impl(const Impl&) = delete;
	Impl& operator=(const Impl&) = delete;

	// permit move semantics
	Impl(Impl&&) noexcept = default;
	Impl& operator=(Impl&&) noexcept = default;

	void show_input(const Ort::Session& session);
	void show_output(const Ort::Session& session);
	Ort::SessionOptions init_session_option(void);

   private:
	// member functions

   private:
	// member variables
};

}  // namespace runtime
}  // namespace arcforge