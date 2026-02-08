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

// runtime/src/onnx/onnx.cpp

#include "Runtime/onnx/onnx.h"
#include "impl/impl.h"

namespace arcforge {
namespace runtime {

RuntimeONNX::RuntimeONNX() : impl_(std::make_unique<Impl>()) {}

RuntimeONNX::RuntimeONNX(std::unique_ptr<Impl> param_impl) : impl_(std::move(param_impl)) {}

RuntimeONNX::~RuntimeONNX() {}

RuntimeONNX::RuntimeONNX(RuntimeONNX&& other) noexcept = default;

RuntimeONNX& RuntimeONNX::operator=(RuntimeONNX&& other) noexcept = default;

void RuntimeONNX::show_input(const Ort::Session& session) {
	if (impl_ != nullptr) {
		impl_->show_input(session);
	}
}

void RuntimeONNX::show_output(const Ort::Session& session) {
	if (impl_ != nullptr) {
		impl_->show_output(session);
	}
}

}  // namespace runtime
}  // namespace arcforge