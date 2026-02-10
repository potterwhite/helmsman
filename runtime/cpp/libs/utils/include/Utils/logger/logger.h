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

#include "Utils/logger/worker/logsink.h"
#include "Utils/pch.h"

namespace arcforge {
namespace embedded {
namespace utils {

enum class LoggerLevel {
	kdebug = 0x10,
	kinfo = 0x11,
	kwarning = 0x12,
	kerror = 0x13,
	kcritical = 0x14
};

class Logger {
   public:
	static Logger& GetInstance();

	~Logger();

	// delete copy and move semantics
	Logger(const Logger&) = delete;
	Logger& operator=(const Logger&) = delete;
	Logger(Logger&&) = delete;
	Logger& operator=(Logger&&) = delete;

	// ----------------------------------------------
	// Config Interface
	// ----------------------------------------------
	void setLevel(LoggerLevel lv);
	void AddSink(std::shared_ptr<LogSink> sink);
	void ClearSinks();

	// ----------------------------------------------
	// Log control interface
	// ----------------------------------------------
	void Log(LoggerLevel level, const std::string& message, std::string_view tag = {});
	void BatchLog(LoggerLevel level, const std::vector<std::string>& messages,
	              std::string_view tag = {});
	void MultiLineLog(LoggerLevel level, const std::string& multiline_message,
	                  std::string_view tag = {});

	// ----------------------------------------------
	// Convenient logging levels
	// ----------------------------------------------
	void Info(const std::string& message, std::string_view tag = {});
	void Debug(const std::string& message, std::string_view tag = {});
	void Warning(const std::string& message, std::string_view tag = {});
	void Error(const std::string& message, std::string_view tag = {});
	void Critical(const std::string& message, std::string_view tag = {});

   private:
	Logger();
	void dispatch(const LogEntry& entry);

   private:
	std::mutex log_mutex_;
	LoggerLevel loggerlevel_;
	std::vector<std::shared_ptr<LogSink>> sinks_;
};

}  // namespace utils
}  // namespace embedded
}  // namespace arcforge