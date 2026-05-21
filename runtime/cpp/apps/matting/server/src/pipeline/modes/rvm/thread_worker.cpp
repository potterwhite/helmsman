#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include "RGAKit/rga_operation.h"
#include "Utils/timing/timer.h"
#include "pipeline/infra/single-slot-channel.h"
#include "pipeline/modes/rvm/rvm.h"
#include "pipeline/stages/backend/post-processor/guided-filter-post-processor.h"

using helmsman::utils::timing::ManualTimer;
using helmsman::utils::timing::ScopedTimer;
using helmsman::utils::timing::StageAccumulator;

void RVMMode::_runPrefetchWorker(size_t model_w, size_t model_h, SingleSlotChannel<cv::Mat>& raw_ch,
                                 SingleSlotChannel<TensorData>& tensor_ch,
                                 StageAccumulator& acc_preprocess) {
	while (true) {
		// Block until a raw frame is available, or the channel is closed (EOF).
		auto frame_opt = raw_ch.pop();
		if (!frame_opt)
			break;  // raw_ch was closed by the main thread — no more frames

		ManualTimer t;
		t.start();
		// BGR frame → letterbox-resized float32 tensor ready for inference.
		// frontend_->preprocess() is thread-safe (Preprocessor is stateless).
		auto tensor = frontend_->preprocess(*frame_opt, model_w, model_h);
		acc_preprocess.record(t.stop());

		tensor_ch.push(std::move(tensor));
	}
	// Signal EOF downstream: tensor_ch.pop() in the main loop returns nullopt.
	tensor_ch.close();
}
