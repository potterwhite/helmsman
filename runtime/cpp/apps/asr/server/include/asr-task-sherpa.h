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

#include "ASREngine/recognizer/recognizer.h"
#include "ASREngine/wav-reader/wav-reader.h"
#include "Network/common/common-types.h"
#include "Network/server/server.h"
#include "Utils/logger/logger.h"

enum class ASRTaskStatus {
	kIdle = 0x01,       // idle: task is created but not yet started
	kRunning = 0x02,    // running: a task is being executed by a thread
	kCompleted = 0x03,  // task finished successfully and can be cleaned up
	                    // kUnhired = 0x01,         //means worker is not hired
	// kHiredWorking = 0x02,    //means worker has been hired and actually working
	// kHiredCompleted = 0x03,  // means worker has been hired and finished his job
};

class ASRTaskSherpa {
   public:
	static std::unique_ptr<ASRTaskSherpa> Create(
	    std::unique_ptr<arcforge::embedded::network_socket::Base>);
	void run();
	bool init();
	void stop_me();
	bool isCompleted() const;

	// duplicate constructor must be deleted
	ASRTaskSherpa(const ASRTaskSherpa&) = delete;
	ASRTaskSherpa& operator=(const ASRTaskSherpa&) = delete;

	// std::move constructor must be default and public, because std::unique_ptr`s object need to be moved in the future
	ASRTaskSherpa(ASRTaskSherpa&&) = default;
	ASRTaskSherpa& operator=(ASRTaskSherpa&&) = default;

	// deconstructor must be public, due to the automatic recycling mechanism of std::unique_ptr
	~ASRTaskSherpa();

   private:
	explicit ASRTaskSherpa();
	void setClient(std::unique_ptr<arcforge::embedded::network_socket::Base> client);

   private:
	arcforge::embedded::ai_asr::Recognizer asr_engine_;
	// bool stop_flag_ = false;
	std::atomic<bool> stop_flag_ = false;
	// std::mutex mutex_;
	std::mutex client_mutex_;
	std::unique_ptr<arcforge::embedded::network_socket::Base> client_ = nullptr;
	std::atomic<bool> finished_flag_{false};
	// arcforge::embedded::ai_asr::SherpaConfig sherpa_config_;
};

struct TaskHandle {
	std::unique_ptr<ASRTaskSherpa> task;
	std::thread worker;
};