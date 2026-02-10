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

#include "Network/common/common-types.h"
#include "Network/pch.h"

namespace arcforge {
namespace embedded {
namespace network_socket {

// forward declaration of PIMPL implementation class
class BaseImpl;

static const std::string k_empty_socket_path_sentinel = "";

class Base {

   public:
	explicit Base();
	virtual ~Base();

	// copy constructor and operator
	Base(const Base&) = delete;
	Base& operator=(const Base&) = delete;

	// std::move constructor and operator
	Base(Base&&) noexcept;
	Base& operator=(Base&&) noexcept;

	virtual SocketStatus isSocketFDValid() const;
	virtual void closeSocket();

	//getter & setter
	virtual int getFD() const;
	virtual void setFD(int);
	virtual const std::string& getSocketPath();
	virtual void setSocketPath(const std::string& path);

	// rx & tx
	virtual SocketReturnValue sendFloat(const std::vector<float>& data);
	virtual SocketReturnValue receiveFloat(std::vector<float>& data);
	virtual SocketReturnValue sendString(const std::string& message);
	virtual SocketReturnValue receiveString(std::string& message);

	// // log
	// virtual void log(const std::string& msg);
	// virtual void log_warning(const std::string& msg);
	// virtual void log_error(const std::string& msg);
	// virtual void log_alert(const std::string& msg);

	// client-utilized-only functions
	virtual SocketReturnValue connectToServer();

	// server-utilized-only functions
	virtual SocketReturnValue startServer(const size_t& timeout = static_cast<size_t>(-1));

	virtual SocketAcceptReturn acceptClient();
	virtual SocketReturnValue unlinkSocketPath();

   private:
	explicit Base(std::unique_ptr<BaseImpl>);
	std::unique_ptr<BaseImpl> impl_;
};

}  // namespace network_socket
}  // namespace embedded
}  // namespace arcforge