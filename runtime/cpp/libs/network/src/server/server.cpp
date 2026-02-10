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

#include "Network/server/server.h"
#include "Utils/logger/logger.h"

namespace arcforge {
namespace embedded {
namespace network_socket {

ServerBase::ServerBase() {}

ServerBase::~ServerBase() {

	std::ostringstream ss;
	ss << "[ServerBase PID:" << getpid()
	   << "] ServerBase destructor called. Attempting to unlink socket path.";
	arcforge::embedded::utils::Logger::GetInstance().Info(ss.str(), kcurrent_lib_name);

	unlinkSocketPath();

	ss.clear();
	ss << "[ServerBase PID:" << getpid()
	   << "] ServerBase destructor finished. Base destructor will handle fd closure.";
	arcforge::embedded::utils::Logger::GetInstance().Info(ss.str(), kcurrent_lib_name);
}

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge