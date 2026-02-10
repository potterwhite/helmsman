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

#include "ASREngine/wav-reader/wav-reader.h"
#include "Network/client/client.h"
#include "Utils/logger/logger.h"
#include "Utils/logger/worker/consolesink.h"
#include "Utils/logger/worker/filesink.h"

#include <chrono>
#include <csignal>  // For signal handling
#include <iostream>
#include <sstream>  // For std::ostringstream
#include <string>
#include <thread>
#include <vector>

using namespace arcforge::embedded;

const std::string_view kcurrent_app_name = "test-client";

const std::string ksocket_path = "/tmp/soCket.paTh";
const int ksample_rate = 16000;
const int CHUNK_DURATION_MS = 1000;  // milliseconds

// --- kill signal capture ---
// static bool g_stop_signal_received = false;
static std::atomic<bool> g_stop_signal_received(false);

void SignalHandler(int signal_num) {
	g_stop_signal_received = true;
	std::ostringstream oss;
	oss << "\nInterrupt signal (" << signal_num << ") received. Shutting down...";
	arcforge::embedded::utils::Logger::GetInstance().Warning(oss.str(), kcurrent_app_name);
}

bool isDebug() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Debug";
}

bool isRelease() {
	constexpr std::string_view build_type = BUILD_TYPE;
	return build_type == "Release";
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
	// auto& logger = arcforge::embedded::utils::Logger::GetInstance();
	// logger.setLevel(arcforge::embedded::utils::LoggerLevel::kdebug);

	//*****************************************************
	// logger level configuration
	// obtain logger unique instance
	auto& logger = arcforge::embedded::utils::Logger::GetInstance();

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

	//----------------------------------

	if (argc < 2) {
		std::ostringstream oss;
		oss << "Usage: " << argv[0] << " <path_to_input_wav_file>"
		    << "\n"
		    << "  Example: " << argv[0] << " full_audio_stream.wav";
		arcforge::embedded::utils::Logger::GetInstance().Error(oss.str(), kcurrent_app_name);
		return 1;
	}

	std::string wav_filepath = argv[1];

	// setup signal handler
	signal(SIGINT, SignalHandler);
	// signal(SIGTERM, SignalHandler);

	// init client object
	network_socket::ClientBase client;
	client.setSocketPath(ksocket_path);

	// connect to server
	network_socket::SocketReturnValue retval_flag = client.connectToServer();
	if (retval_flag > network_socket::SocketReturnValue::ksuccess) {
		arcforge::embedded::utils::Logger::GetInstance().Error(
		    "Client failed to connect to server.", kcurrent_app_name);
		exit(1);
	}

	// --- 2. Open wav file ---
	ai_asr::WavReader reader;
	if (!reader.Open(wav_filepath, ksample_rate, 1 /*expected channels*/)) {
		std::ostringstream oss;
		oss << "Failed to open WAV file: " << wav_filepath;
		arcforge::embedded::utils::Logger::GetInstance().Error(oss.str(), kcurrent_app_name);
		return 1;
	}

	{
		std::ostringstream oss;
		oss << "\nStarting client wav reading for '" << wav_filepath << "'..."
		    << "\n"
		    << "Processing audio in " << CHUNK_DURATION_MS << "ms chunks."
		    << "\n"
		    << "Press Ctrl+C to stop.\n";
		arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_app_name);
	}

	const size_t samples_per_chunk = static_cast<size_t>((ksample_rate * CHUNK_DURATION_MS) / 1000);
	std::vector<float> audio_chunk;

	// --- 3. Processing with conditional loop ---
	while ((g_stop_signal_received == false) && (reader.Eof() == false)) {
		// At the start of each loop, as the caller, we proactively prepare a sufficiently large memory block.
		//    This ensures that regardless of how ReadSamples is implemented, the buffer we provide is always safe.
		audio_chunk.resize(samples_per_chunk);

		size_t samples_read = reader.ReadSamples(audio_chunk, samples_per_chunk);

		if (samples_read > 0) {
			{
				std::ostringstream oss;
				oss << "main-client.cpp: size of audio_chunk=" << audio_chunk.size();
				arcforge::embedded::utils::Logger::GetInstance().Debug(oss.str(),
				                                                       kcurrent_app_name);
			}

			// [key step] Precisely truncate the vector according to the actual number of samples read.
			//    This ensures that the data we send to the server is neither more nor less than samples_read samples.
			//    Even if ReadSamples does not resize internally, this line of code ensures correctness.
			//    Even if ReadSamples has already resized internally, this line of code is just a harmless redundant operation.
			//    This is defensive programming: do not rely on the other party, ensure correctness yourself.
			if (samples_read < audio_chunk.size()) {
				audio_chunk.resize(samples_read);
			}

			//send float to server
			network_socket::SocketReturnValue retval = client.sendFloat(audio_chunk);
			if (retval > network_socket::SocketReturnValue::ksuccess) {
				arcforge::embedded::utils::Logger::GetInstance().Error(
				    "Client failed to send float data.", kcurrent_app_name);
				continue;
			}

			//receive string from server
			std::string result;
			retval = client.receiveString(result);
			// if (result.empty() == false) {
			{
				std::ostringstream oss;
				oss << "receive:" << result;
				arcforge::embedded::utils::Logger::GetInstance().Info(oss.str(), kcurrent_app_name);
			}
			// }
		}
	}  // end of while()

	if (g_stop_signal_received) {
		arcforge::embedded::utils::Logger::GetInstance().Info("\nProcessing stopped by user.",
		                                                      kcurrent_app_name);
	} else {
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "\nEnd of WAV file reached in main loop.", kcurrent_app_name);
		//-----------------------------------------------------
		// send EOF marker (an empty chunk)
		std::vector<float> empty_chunk;
		client.sendFloat(empty_chunk);
		arcforge::embedded::utils::Logger::GetInstance().Info(
		    "!!!!!!!!!!!!!!!!!!!!!Sent EOF marker (empty chunk)", kcurrent_app_name);
	}

	return 0;
}