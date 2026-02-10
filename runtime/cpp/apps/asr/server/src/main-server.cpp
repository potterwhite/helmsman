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

#include "pch.h"

#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"
#include "acceptor.h"
#include "common-types.h"

static std::atomic<bool> g_stop_signal_received(false);

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;

	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	arcforge::embedded::utils::Logger::GetInstance().Error(oss.str());
}

const std::string ksocket_path = "/tmp/soCket.paTh";

bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {

	signal(SIGINT, SignalHandler);
	signal(SIGTERM, SignalHandler);

	//*****************************************************
	// logger level configuration
	// obtain logger unique instance
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

	// std::cout << "[main] sizeof(arcforge::embedded::utils::FileSink) = "
	//           << sizeof(arcforge::embedded::utils::FileSink) ;

	// 1. Configure logger level
	if (isRelease() == true) {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kinfo);
	} else {
		logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);
	}

	// 2. Configure output targets (Sinks)
	logger.ClearSinks();
	logger.AddSink(
	    std::make_shared<arcforge::embedded::utils::FileSink>("/root/my_app_client.log"));
	logger.AddSink(std::make_shared<arcforge::embedded::utils::ConsoleSink>());

	logger.Info("Application has started.");

	// socket path initialize
	auto server = std::make_unique<arcforge::embedded::network_socket::ServerBase>();
	auto acceptor = Acceptor::Create(std::move(server));
	acceptor->setSocketPath(ksocket_path);
	acceptor->init();

	while (1) {
		if (g_stop_signal_received == true) {
			logger.Warning("main() will send stop signal to Acceptor.");

			acceptor->stop_me();

			break;
		} else {
			acceptor->process();
		}
	}
	// recognizer will be cleaned up automatically when main ends

	std::stringstream ss;
	ss << std::this_thread::get_id() << ": "
	   << "ASR + Socket Server main-thread execution finished.";
	logger.Info(ss.str());

	return 0;
}