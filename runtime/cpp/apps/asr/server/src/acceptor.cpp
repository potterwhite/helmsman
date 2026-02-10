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

#include "acceptor.h"
#include "Utils/logger/logger.h"
#include "common-types.h"

std::unique_ptr<Acceptor> Acceptor::Create(
    std::unique_ptr<arcforge::embedded::network_socket::ServerBase> server) {

	// auto acceptor = std::make_unique<Acceptor>();
	auto acceptor = std::unique_ptr<Acceptor>(new Acceptor(std::move(server)));

	return acceptor;
}

Acceptor::Acceptor(std::unique_ptr<arcforge::embedded::network_socket::ServerBase> server)
    : server_(std::move(server)) {

	arcforge::embedded::utils::Logger::GetInstance().Info("constructor of Acceptor class",
	                                                    kcurrent_app_name);
	// server_ = std::unique_ptr<arcforge::embedded::network_socket::ServerBase>(
	//     new arcforge::embedded::network_socket::ServerBase);
}
// Acceptor::Acceptor(const std::string& path) : ksocket_path_(path) {}

Acceptor::~Acceptor() {

	arcforge::embedded::utils::Logger::GetInstance().Info("deconstructor of Acceptor class",
	                                                    kcurrent_app_name);
	for (auto& task_handler : active_task_handlers_) {
		task_handler.task->stop_me();
	}

	for (auto& task_handler : active_task_handlers_) {
		if (task_handler.worker.joinable() == true) {
			task_handler.worker.join();
		}
	}
}

// Acceptor::~Acceptor() {

// 	arcforge::embedded::utils::Logger::GetInstance().Info("deconstructor of Acceptor class",
// 	                                                    kcurrent_app_name);
// 	if (worker_thread_.joinable() == true) {
// 		if (asr_task_sherpa_ != nullptr) {
// 			asr_task_sherpa_->stop_me();
// 		}

// 		worker_thread_.join();
// 	}
// }

void Acceptor::setSocketPath(const std::string& path) {
	ksocket_path_ = path;
}

void Acceptor::init() {

	// -- 2. create server object
	server_->setSocketPath(ksocket_path_);

	// -- 2. unlink exist server
	if (server_->unlinkSocketPath() ==
	    arcforge::embedded::network_socket::SocketReturnValue::ksuccess) {
		std::ostringstream oss;
		oss << "[ServerPID:" << getpid() << "] Removed existing socket file: " << ksocket_path_;
		arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_app_name);
	}

	// -- 3. Start the server
	if (server_->startServer(timeout_value_) >
	    arcforge::embedded::network_socket::SocketReturnValue::ksuccess) {
		std::ostringstream oss;
		oss << "[ServerPID:" << getpid() << "] FATAL: Server failed to start on " << ksocket_path_;
		arcforge::embedded::utils::Logger::GetInstance().Error(oss.str(), kcurrent_app_name);
		// Note: unlink won't be called here on failure if ServerBase doesn't do it on startServer failure.
		// However, startServer should ideally not create the socket file if it can't bind/listen.
		return;
	}

	std::ostringstream oss;
	oss << "[ServerPID:" << getpid() << "] Server started and waiting for any client to come in "
	    << ksocket_path_;
	arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_app_name);

	oss.clear();
	oss << "[ServerPID:" << getpid() << "] Press \"Ctrl+C\" or \"kill\" to shut down.";
	arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_app_name);
}

// ASRTaskStatus Acceptor::TaskChecker() {
// 	if (asr_task_sherpa_ == nullptr) {
// 		return ASRTaskStatus::kIdle;
// 	}

// 	// if asr_task_sherpa_ is not nullptr, means room is using now, worker is still under working now
// 	if (asr_task_sherpa_->isCompleted() == true) {
// 		// if asr_task_sherpa_ has finished its life(e.g. client peer closed), recycle asr_task_sherpa_
// 		std::stringstream str;
// 		str << "asr_task_sherpa_`s finished flag has been set, recycling now...";
// 		arcforge::embedded::utils::Logger::GetInstance().Debug(str.str(), kcurrent_app_name);

// 		// asr_task_sherpa_ = nullptr;

// 		return ASRTaskStatus::kCompleted;
// 	} else {
// 		// worker is busy
// 		std::stringstream str;
// 		str << "asr_task_sherpa_ has not finished, please wait for previous client end its "
// 		       "connection.";
// 		arcforge::embedded::utils::Logger::GetInstance().Debug(str.str(), kcurrent_app_name);

// 		return ASRTaskStatus::kRunning;
// 	}
// }

void Acceptor::process() {
	/*-----------------------------------------
	 * stage 1st. check if we have room in queue
	 ------------------------------------------*/
	for (auto it = active_task_handlers_.begin(); it != active_task_handlers_.end();) {
		if (it->task->isCompleted() == true) {
			it->worker.join();

			it = active_task_handlers_.erase(it);
		} else {
			++it;
		}
	}

	if (active_task_handlers_.size() >= kMAX_CONCURRENT_TASKS_) {
		arcforge::embedded::utils::Logger::GetInstance().Info("Task Queue is full, try next time",
		                                                    kcurrent_app_name);
		return;
	}

	/*-----------------------------------------
	 * stage 2nd. accept client
	 ------------------------------------------*/
	// if runs to this place means we have room for more client
	arcforge::embedded::utils::Logger::GetInstance().Error("hey, acceptclient start ",
	                                                     kcurrent_app_name);
	arcforge::embedded::network_socket::SocketAcceptReturn accept_retval = server_->acceptClient();
	if (arcforge::embedded::network_socket::SocketReturnValueIsSuccess(accept_retval.return_value) ==
	    false) {
		if (accept_retval.return_value ==
		    // timeout is not an error
		    arcforge::embedded::network_socket::SocketReturnValue::kaccept_timeout) {
			arcforge::embedded::utils::Logger::GetInstance().Debug(
			    "acceptClient() timeout, try next time", kcurrent_app_name);
			return;
		}

		// error occurrs
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "acceptClient() return" + arcforge::embedded::network_socket::SocketReturnValueToString(
		                                  accept_retval.return_value),
		    kcurrent_app_name);
		return;
	}

	/*-----------------------------------------
	 * stage 3rd. Create new Task
	 ------------------------------------------*/
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "\nNew client connected. Creating worker thread.", kcurrent_app_name);

	auto new_task = ASRTaskSherpa::Create(std::move(accept_retval.client));

	/*-----------------------------------------
	 * stage 4th. Create work to do the previous created Task
	 *------------------------------------------*/
	// std::thread t(&ASRTaskSherpa::run, asr_task_sherpa_.get());
	// std::thread t(&ASRTaskSherpa::run);
	// t.detach();
	auto new_worker = std::thread(&ASRTaskSherpa::run, new_task.get());

	// worker_thread_.detach();
	// worker_thread_.join();

	active_task_handlers_.push_back({std::move(new_task), std::move(new_worker)});
	// active_task_handlers_.emplace_back(std::move(new_task),
	//                                    std::thread(&ASRTaskSherpa::run, new_task.get()));
}

// void Acceptor::process() {

// 	ASRTaskStatus status = TaskChecker();
// 	switch (status) {
// 		case ASRTaskStatus::kRunning:
// 			std::this_thread::sleep_for(std::chrono::milliseconds(500));
// 			return;
// 		case ASRTaskStatus::kIdle:
// 			break;
// 		case ASRTaskStatus::kCompleted:
// 			// worker is unhired(completed), so let us check if there is any job to do
// 			if (worker_thread_.joinable()) {
// 				worker_thread_.join();
// 			}

// 			asr_task_sherpa_ = nullptr;
// 			break;
// 		default:
// 			arcforge::embedded::utils::Logger::GetInstance().Error(
// 			    "TaskChecker error-(" + std::to_string(static_cast<int>(status)) +
// 			    "), return now!");
// 			break;
// 	}

// 	arcforge::embedded::utils::Logger::GetInstance().Error("hey, acceptclient start ");

// 	// if nullptr, means no client connected
// 	arcforge::embedded::network_socket::SocketAcceptReturn accept_retval = server_->acceptClient();
// 	client_connection_ = std::move(accept_retval.client);
// 	if (arcforge::embedded::network_socket::SocketReturnValueIsSuccess(accept_retval.return_value) ==
// 	    false) {
// 		// just something wrong, so quite until next loop
// 		return;
// 	}
// 	arcforge::embedded::utils::Logger::GetInstance().Error("hey, acceptclient end ");

// 	// allocate new task and ready for worker to work
// 	arcforge::embedded::utils::Logger::GetInstance().Info(
// 	    "\nNew client connected. Creating worker thread.");
// 	asr_task_sherpa_ = ASRTaskSherpa::Create(std::move(client_connection_));

// 	// std::thread t(&ASRTaskSherpa::run, asr_task_sherpa_.get());
// 	// std::thread t(&ASRTaskSherpa::run);
// 	// t.detach();
// 	worker_thread_ = std::thread(&ASRTaskSherpa::run, asr_task_sherpa_.get());

// 	// worker_thread_.detach();
// 	// worker_thread_.join();
// }

void Acceptor::stop_me() {
	for (auto& task_handler : active_task_handlers_) {
		if (task_handler.task->isCompleted() == false) {
			arcforge::embedded::utils::Logger::GetInstance().Info(
			    "Acceptor will send stop signal to ASRTaskSherpa.");
			task_handler.task->stop_me();
		}
	}
}
// void Acceptor::stop_me() {
// 	if (asr_task_sherpa_ != nullptr) {
// 		arcforge::embedded::utils::Logger::GetInstance().Info(
// 		    "Acceptor will send stop signal to ASRTaskSherpa.");
// 		asr_task_sherpa_->stop_me();
// 	}
// }