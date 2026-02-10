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

#include "pch.h"

// #include "client/client.h"
#include "Network/common/common-types.h"
#include "Network/server/server.h"
#include "Utils/logger/logger.h"
#include "asr-task-sherpa.h"

class Acceptor {
   public:
	// singelton mode
	// static Acceptor& GetInstance();
	// ------------------
	// factory mode
	static std::unique_ptr<Acceptor> Create(
	    std::unique_ptr<arcforge::embedded::network_socket::ServerBase>);

	void init();
	void setSocketPath(const std::string&);
	void process();
	~Acceptor();
	void stop_me();

	// assign constructor & deconstructor
	Acceptor(const Acceptor&) = delete;
	Acceptor& operator=(const Acceptor&) = delete;

	// move constructor & deconstructor
	Acceptor(Acceptor&&) = default;
	Acceptor& operator=(Acceptor&&) = default;

   private:
	explicit Acceptor(std::unique_ptr<arcforge::embedded::network_socket::ServerBase>);
	// ASRTaskStatus TaskChecker();

   private:
	std::string ksocket_path_;
	std::unique_ptr<arcforge::embedded::network_socket::ServerBase> server_ = nullptr;
	// std::unique_ptr<arcforge::embedded::network_socket::Base> client_connection_ = nullptr;
	// std::unique_ptr<ASRTaskSherpa> asr_task_sherpa_ = nullptr;
	// std::thread worker_thread_;
	size_t timeout_value_{2000};
	std::vector<TaskHandle> active_task_handlers_;
	static constexpr size_t kMAX_CONCURRENT_TASKS_ = 2;
};