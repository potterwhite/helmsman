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

#include "asr-task-sherpa.h"
#include "Utils/logger/logger.h"
#include "common-types.h"

// --- Config ---
// sherpa-onnx model paths
// **** pls modify these paths according to your actual model locations ****
// **** also ensure the provider is set to "rknn" for RK3588 if you are using RKNN ****
const std::string ENCODER_PATH =
    "/home/asr/models/sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "encoder.rknn";
const std::string DECODER_PATH =
    "/home/asr/models/sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "decoder.rknn";
const std::string JOINER_PATH =
    "/home/asr/models/sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "joiner.rknn";
const std::string TOKENS_PATH =
    "/home/asr/models/sherpa-onnx-rk3588-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "tokens.txt";

const std::string PROVIDER = "rknn";
// --num-threads=1 to select RKNN_NPU_CORE_AUTO
// --num-threads=0 to select RKNN_NPU_CORE_0
// --num-threads=-1 to select RKNN_NPU_CORE_1
// --num-threads=-2 to select RKNN_NPU_CORE_2
// --num-threads=-3 to select RKNN_NPU_CORE_0_1
// --num-threads=-4 to select RKNN_NPU_CORE_0_1_2
const int NUM_THREADS = -4;

std::unique_ptr<ASRTaskSherpa> ASRTaskSherpa::Create(
    std::unique_ptr<arcforge::embedded::network_socket::Base> client) {

	// return std::make_unique<ASRTaskSherpa>();
	auto task = std::unique_ptr<ASRTaskSherpa>(new ASRTaskSherpa());
	task->setClient(std::move(client));

	return task;
}

ASRTaskSherpa::ASRTaskSherpa() {
	arcforge::embedded::utils::Logger::GetInstance().Info("constructor of ASRTaskSherpa class",
	                                                      kcurrent_app_name);
	init();
}

ASRTaskSherpa::~ASRTaskSherpa() {
	arcforge::embedded::utils::Logger::GetInstance().Info("deconstructor of ASRTaskSherpa class",
	                                                      kcurrent_app_name);
}

void ASRTaskSherpa::setClient(std::unique_ptr<arcforge::embedded::network_socket::Base> client) {
	client_ = std::move(client);
}

bool ASRTaskSherpa::isCompleted() const {
	return finished_flag_;
}

bool ASRTaskSherpa::init() {
	// --- 1. Init ASR Engine  ---
	arcforge::embedded::ai_asr::SherpaConfig config =
	    arcforge::embedded::ai_asr::SherpaConfig::Builder()
	        .setFirstEncoderPath(ENCODER_PATH)
	        .setSecondDecoderPath(DECODER_PATH)
	        .setThirdJoinerPath(JOINER_PATH)
	        .setFourthTokensPath(TOKENS_PATH)
	        .setFifthProvider(PROVIDER)
	        .setSixthNumThreads(NUM_THREADS)
	        .setTwelfthEndpointDetectionSupport(
	            arcforge::embedded::ai_asr::SherpaEndPointSupport::kenable)
	        .build();

	bool Erfolg = asr_engine_.Initialize(config);
	if (!Erfolg) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Failed to initialize ASR engine. Exiting.", kcurrent_app_name);
		return false;
	}

	arcforge::embedded::utils::Logger::GetInstance().Info("ASRTaskSherpa has done with init!",
	                                                      kcurrent_app_name);

	return true;
}

void ASRTaskSherpa::run() {
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "Worker thread started for a new client.");

	while (stop_flag_ == false) {

		std::vector<float> audio_chunk;
		arcforge::embedded::network_socket::SocketReturnValue retval;

		// step 1: Safely receive data
		// Before accessing client_, we must lock.
		// Must protect the entire receiveFloat call to ensure that the main thread does not reset client_ during reception.
		{
			std::lock_guard<std::mutex> lock(client_mutex_);

			// check again for client_ because it might have been destroyed by stop_me() between loops
			if (!client_) {
				// if client_ is null, it means we are asked to stop, so just exit the loop
				break;
			}

			// This call may block for a long time, but we must hold the lock to prevent client_ from being reset.
			retval = client_->receiveFloat(audio_chunk);
		}

		// --- Step 2: Process received data ---
		// if (receive failed, including being interrupted by stop_me), exit the loop
		if (retval != arcforge::embedded::network_socket::SocketReturnValue::ksuccess) {
			std::string reason;
			switch (retval) {
				case arcforge::embedded::network_socket::SocketReturnValue::ksuccess:
					reason = "Successful receive.";
					break;
				case arcforge::embedded::network_socket::SocketReturnValue::keof:
					reason = "Client closed connection gracefully (EOF).";
					break;
				case arcforge::embedded::network_socket::SocketReturnValue::kpeer_abnormally_closed:
					reason = "Peer abnormally closed connection.";
					break;
				case arcforge::embedded::network_socket::SocketReturnValue::kreceived_illegal:
					reason =
					    "recv() failed, likely because server initiated shutdown by closing the "
					    "socket.";
					break;
				case arcforge::embedded::network_socket::SocketReturnValue::kreceived_null:
				case arcforge::embedded::network_socket::SocketReturnValue::kreceivelength_failed:
				case arcforge::embedded::network_socket::SocketReturnValue::ksendcount_failed:
				case arcforge::embedded::network_socket::SocketReturnValue::ksenddata_failed:
				case arcforge::embedded::network_socket::SocketReturnValue::ksendlength_failed:
				case arcforge::embedded::network_socket::SocketReturnValue::kcount_too_large:
				case arcforge::embedded::network_socket::SocketReturnValue::kempty_string:
				case arcforge::embedded::network_socket::SocketReturnValue::kfd_illegal:
				case arcforge::embedded::network_socket::SocketReturnValue::ksocketpath_empty:
				case arcforge::embedded::network_socket::SocketReturnValue::kconnect_server_failed:
				case arcforge::embedded::network_socket::SocketReturnValue::klisten_error:
				case arcforge::embedded::network_socket::SocketReturnValue::kbind_error:
				case arcforge::embedded::network_socket::SocketReturnValue::kaccept_timeout:
				case arcforge::embedded::network_socket::SocketReturnValue::ksetsocketopt_error:
				case arcforge::embedded::network_socket::SocketReturnValue::kimpl_nullptr_error:
				case arcforge::embedded::network_socket::SocketReturnValue::kinit_state:
				case arcforge::embedded::network_socket::SocketReturnValue::kunknownerror:
				default:
					reason = "An unexpected socket error occurred: " +
					         arcforge::embedded::network_socket::SocketReturnValueToString(retval);
					break;
			}
			arcforge::embedded::utils::Logger::GetInstance().Info(
			    "Exiting worker thread. Reason: " + reason);
			break;
		}

		// --- Step 3: ASR processing (this is pure computation, no locking needed) ---
		asr_engine_.ProcessAudioChunk(audio_chunk);
		std::string recognized_text = asr_engine_.GetCurrentText();

		// --- Step 4: Safely send result ---
		// Similarly, lock before accessing client_
		{
			std::lock_guard<std::mutex> lock(client_mutex_);

			if (!client_) {
				arcforge::embedded::utils::Logger::GetInstance().Warning(
				    "Client connection was closed before sending result.");
				break;
			}

			retval = client_->sendString(recognized_text);
		}

		// if send failed, exit the loop
		if (retval != arcforge::embedded::network_socket::SocketReturnValue::ksuccess) {
			arcforge::embedded::utils::Logger::GetInstance().Error(
			    "Failed to send string data, exiting worker thread.");
			break;
		}

		// --- Step 5: Reset ASR stream (no locking needed) ---
		asr_engine_.ResetStream();
	}

	// universal cleanup after loop exit
	finished_flag_ = true;
	arcforge::embedded::utils::Logger::GetInstance().Info(
	    "ASRTaskSherpa run loop finished, worker thread is now exiting.");
}

// stop_me() final thread-safe version
void ASRTaskSherpa::stop_me() {
	stop_flag_ = true;
	std::lock_guard<std::mutex> lock(client_mutex_);
	if (client_) {
		client_.reset();
	}
}
