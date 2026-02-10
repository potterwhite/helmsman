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

#include "Utils/logger/logger.h"
#include <iostream>
#include <sstream>
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/log_entry.h"

namespace arcforge {
namespace embedded {
namespace utils {

Logger& Logger::GetInstance() {
	static Logger instance;
	return instance;
}

Logger::Logger() : loggerlevel_(LoggerLevel::kdebug) {
	AddSink(std::make_shared<ConsoleSink>());
}

Logger::~Logger() {}

void Logger::setLevel(LoggerLevel lv) {
	std::lock_guard<std::mutex> lock(log_mutex_);
	loggerlevel_ = lv;
}

void Logger::AddSink(std::shared_ptr<LogSink> sink) {
	std::lock_guard<std::mutex> lock(log_mutex_);
	if (sink) {
		sinks_.push_back(sink);
	}
}

void Logger::ClearSinks() {
	std::lock_guard<std::mutex> lock(log_mutex_);
	sinks_.clear();
}

void Logger::dispatch(const LogEntry& entry) {
	std::vector<std::shared_ptr<LogSink>> sinks_copy;
	{
		// shrink the lock scope, only protect sinks_ copying
		// this way we don't hold the lock during I/O operations (sink->log), improving performance
		std::lock_guard<std::mutex> lock(log_mutex_);
		sinks_copy = sinks_;
	}

	for (const auto& sink : sinks_copy) {
		if (sink) {
			sink->log(entry);
		}
	}
}

/****************************************************************
                        Key Logging Method
****************************************************************/
void Logger::Log(LoggerLevel level, const std::string& message, std::string_view tag) {
	// Level check early to avoid unnecessary object creation and function calls
	if (loggerlevel_ > level) {
		return;
	}

	LogEntry entry;
	entry.timestamp = std::chrono::system_clock::now();
	entry.level = level;
	entry.message = message;
	entry.tag = tag;

	dispatch(entry);
}

/****************************************************************
					Batch Log / MultiLineLog
****************************************************************/
void Logger::BatchLog(LoggerLevel level, const std::vector<std::string>& messages,
                      std::string_view tag) {
	if (loggerlevel_ > level) {
		return;
	}
	for (const auto& msg : messages) {
		Log(level, msg, tag);
	}
}

void Logger::MultiLineLog(LoggerLevel level, const std::string& multiline_message,
                          std::string_view tag) {
	if (loggerlevel_ > level) {
		return;
	}
	std::istringstream stream(multiline_message);
	std::string line;
	while (std::getline(stream, line)) {
		Log(level, line, tag);
	}
}

/****************************************************************
					Convienience Logging Methods
****************************************************************/
void Logger::Info(const std::string& message, std::string_view tag) {
	Log(LoggerLevel::kinfo, message, tag);
}

void Logger::Debug(const std::string& message, std::string_view tag) {
	Log(LoggerLevel::kdebug, message, tag);
}

void Logger::Warning(const std::string& message, std::string_view tag) {
	Log(LoggerLevel::kwarning, message, tag);
}

void Logger::Error(const std::string& message, std::string_view tag) {
	Log(LoggerLevel::kerror, message, tag);
}

void Logger::Critical(const std::string& message, std::string_view tag) {
	Log(LoggerLevel::kcritical, message, tag);
}

}  // namespace utils
}  // namespace embedded
}  // namespace arcforge