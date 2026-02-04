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

// libs/network/include/Network/base/impl/base-impl.h
#pragma once

#include "Network/common/common-types.h"
#include "Network/pch.h"

namespace arcforge {
namespace embedded {
namespace network_socket {

inline constexpr int killegal_fd_value = -1;

class BaseImpl;

struct SocketAcceptImplReturn {
	SocketReturnValue return_value_impl;
	std::unique_ptr<BaseImpl> client_impl;

	SocketAcceptImplReturn();

	SocketAcceptImplReturn(SocketReturnValue, std::unique_ptr<BaseImpl>);
};

class BaseImpl {
   public:
	BaseImpl();
	~BaseImpl();

	// forbitd copy and assignment
	BaseImpl(const BaseImpl&) = delete;
	BaseImpl& operator=(const BaseImpl&) = delete;

	// permit move semantics
	BaseImpl(BaseImpl&&) noexcept = default;
	BaseImpl& operator=(BaseImpl&&) noexcept = default;

	SocketStatus isSocketFDValid_safe();
	void closeSocket_safe();

	// getter and setter
	int getFD_safe();
	void setFD_safe(int);
	const std::string& getSocketPath_safe();
	void setSocketPath_safe(const std::string& path);

	// rx & tx methods
	SocketReturnValue receiveFloat_safe(std::vector<float>& data);
	SocketReturnValue sendString_safe(const std::string& message);
	SocketReturnValue sendFloat_safe(const std::vector<float>& data);
	SocketReturnValue receiveString_safe(std::string& message);

	// // log functions
	// void log_safe(const std::string& msg);
	// void log_warning_safe(const std::string& msg);
	// void log_error_safe(const std::string& msg);
	// void log_alert_safe(const std::string& msg);

	// client-utilized-only functions
	SocketReturnValue connectToServer();

	// server-utilized-only functions
	SocketReturnValue startServer(
	    const size_t& timeout = static_cast<size_t>(-1)); 
	SocketAcceptImplReturn acceptClient();
	SocketReturnValue unlinkSocketPath();

   private:
	SocketStatus isSocketFDValid() const;
	void closeSocket();
	int getFD() const;
	void setFD(int);
	const std::string& getSocketPath() const;
	void setSocketPath(const std::string& path);
	// // log functions
	// void log(const std::string& msg);
	// void log_warning(const std::string& msg);
	// void log_error(const std::string& msg);
	// void log_alert(const std::string& msg);

   private:
	int socketfd_ = -1;
	std::string socketpath_;
	std::unique_ptr<std::mutex> socket_mutex_;
	std::unique_ptr<std::mutex> log_mutex_;
};

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge