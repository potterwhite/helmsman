// Copyright (c) 2026 PotterWhite
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

// =============================================================================
// frontend-create-nohw.cpp — FrontendBase::Create() for non-hardware platforms
//
// Compiled when CMAKE_PLATFORM does NOT include "rockchip".
//
// =============================================================================

#include "pipeline/stages/frontend/frontend-core/frontend-base.h"
#include "pipeline/stages/frontend/frontend-core/impl/no-hw-frontend.h"

#include <stdexcept>

std::unique_ptr<FrontendBase> FrontendBase::Create(const AppConfig& config) {
    if (config.use_hardware_decoder) {
        throw std::runtime_error("Hardware decoder not supported on this build");
    }
    return std::make_unique<NoHwFrontend>(config.input_path, config.use_multithread);
}
