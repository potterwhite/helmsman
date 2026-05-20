// Copyright (c) 2026 PotterWhite
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

// =============================================================================
// matting-server.cpp — DMA buffer fd-based per-frame inference server
// =============================================================================

#include "pipeline/modes/rvm/matting-server.h"

#include <opencv2/imgproc.hpp>
#include <sys/mman.h>
#include <unistd.h>
#include "RGAKit/rga_operation.h"
#include "common/common-define.h"
#include "pipeline/stages/inference-engine/inference-engine-factory.h"

using helmsman::rgakit::ImageDescriptor;
using helmsman::rgakit::RgaPixelFormat;

static constexpr const char* kMattingServerModule = "MattingServer";

MattingServer::MattingServer() = default;
MattingServer::~MattingServer() { shutdown(); }

bool MattingServer::init(const std::string& model_path, const std::string& bg_path,
                         int output_w, int output_h) {
	auto& logger = helmsman::utils::Logger::GetInstance();
	output_w_ = output_w;
	output_h_ = output_h;

	if (!initModel(model_path)) return false;
	if (!initRga()) return false;
	if (!initBackground(bg_path, output_w, output_h)) return false;

	// Allocate output DMA buffer (BGR888 at output resolution).
	const size_t buf_bytes = static_cast<size_t>(output_w) * static_cast<size_t>(output_h) * 3;
	dma_output_buf_ = helmsman::dmakit::DmaBuffer::Allocate(buf_bytes);
	if (!dma_output_buf_ || !dma_output_buf_->map()) {
		logger.Warning("MattingServer: failed to allocate output DMA buffer", kMattingServerModule);
		return false;
	}

	logger.Info("MattingServer initialized: model=" + std::to_string(model_w_) + "x" +
	                std::to_string(model_h_) + ", output=" + std::to_string(output_w_) + "x" +
	                std::to_string(output_h_) + ", dsr=" + std::to_string(dsr_),
	            kMattingServerModule);
	initialized_ = true;
	return true;
}

int MattingServer::processFrame(int input_fd, int input_w, int input_h) {
	if (!initialized_ || input_fd < 0) return -1;
	auto& logger = helmsman::utils::Logger::GetInstance();

	// 1. Map input DMA buffer to get pixel data.
	void* input_ptr = ::mmap(nullptr, static_cast<size_t>(input_w) * static_cast<size_t>(input_h) * 3,
	                         PROT_READ, MAP_SHARED, input_fd, 0);
	if (input_ptr == MAP_FAILED) {
		logger.Warning("MattingServer: mmap input fd failed", kMattingServerModule);
		return -1;
	}

	// 2. Wrap as cv::Mat (no copy — just a view).
	cv::Mat input_frame(input_h, input_w, CV_8UC3, input_ptr);

	// 3. Preprocess: resize + normalize → TensorData.
	TensorData tensor = frontend_.preprocess(input_frame, static_cast<size_t>(model_w_),
	                                         static_cast<size_t>(model_h_));

	// 4. Inject recurrent states.
	std::vector<TensorData> inputs = {tensor};
	state_mgr_.inject(inputs);

	if (engine_->needsDownsampleRatio()) {
		TensorData dsr;
		dsr.name = "downsample_ratio";
		dsr.shape = {1};
		dsr.data = {dsr_};
		inputs.push_back(std::move(dsr));
	}

	// 5. Run inference.
	std::vector<TensorData> outputs;
	engine_->infer(inputs, outputs);

	// 6. Update recurrent states.
	state_mgr_.update(outputs);

	// 7. Post-process → alpha matte.
	cv::Mat alpha_8u = backend_.postprocess(outputs, input_frame);

	// 8. Composite into DMA output buffer.
	//    Steps: resize alpha (CPU) → resize frame (RGA) → merge → composite → upscale to DMA.
	const int model_h = bg_model_u8_.rows;
	const int model_w = bg_model_u8_.cols;

	// 8a. Resize alpha to model resolution (CPU — RGA doesn't support YUV400).
	cv::Mat alpha_model(model_h, model_w, CV_8UC1);
	cv::resize(alpha_8u, alpha_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);

	// 8b. Resize frame to model resolution (RGA).
	cv::Mat frame_model(model_h, model_w, CV_8UC3);
	ImageDescriptor src_desc(input_frame.data, input_w, input_h, RgaPixelFormat::kBgr888);
	ImageDescriptor dst_desc(frame_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
	if (!rga_resize_->Execute(src_desc, dst_desc)) {
		cv::resize(input_frame, frame_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
	}

	// 8c. CPU alpha blend: fg_bgr * alpha + bg_bgr * (1-alpha) → composed_bgr.
	cv::Mat composed_model(model_h, model_w, CV_8UC3);
	{
		const int pixels = model_h * model_w;
		const uint8_t* fg_ptr = frame_model.ptr<uint8_t>(0);
		const uint8_t* bg_ptr = bg_model_u8_.ptr<uint8_t>(0);
		const uint8_t* a_ptr = alpha_model.ptr<uint8_t>(0);
		uint8_t* out = composed_model.ptr<uint8_t>(0);
		for (int i = 0; i < pixels; ++i) {
			const uint16_t a = a_ptr[i];
			const uint16_t inv = 255 - a;
			out[0] = static_cast<uint8_t>((fg_ptr[0] * a + bg_ptr[0] * inv + 1 + ((fg_ptr[0] * a + bg_ptr[0] * inv) >> 8)) >> 8);
			out[1] = static_cast<uint8_t>((fg_ptr[1] * a + bg_ptr[1] * inv + 1 + ((fg_ptr[1] * a + bg_ptr[1] * inv) >> 8)) >> 8);
			out[2] = static_cast<uint8_t>((fg_ptr[2] * a + bg_ptr[2] * inv + 1 + ((fg_ptr[2] * a + bg_ptr[2] * inv) >> 8)) >> 8);
			fg_ptr += 3; bg_ptr += 3; out += 3;
		}
	}

	// 8e. Upscale directly into DMA output buffer (RGA, zero-copy).
	void* dma_ptr = dma_output_buf_->map();
	ImageDescriptor upscale_src(composed_model.data, model_w, model_h, RgaPixelFormat::kBgr888);
	ImageDescriptor upscale_dst(dma_ptr, output_w_, output_h_, RgaPixelFormat::kBgr888);
	if (!rga_resize_->Execute(upscale_src, upscale_dst)) {
		// CPU fallback
		cv::Mat composed_full;
		cv::resize(composed_model, composed_full, cv::Size(output_w_, output_h_), 0, 0, cv::INTER_LINEAR);
		memcpy(dma_ptr, composed_full.data, composed_full.total() * composed_full.elemSize());
	}

	// 9. Unmap input.
	::munmap(input_ptr, static_cast<size_t>(input_w) * static_cast<size_t>(input_h) * 3);

	return dma_output_buf_->fd();
}

void MattingServer::shutdown() {
	dma_output_buf_.reset();
	rga_resize_.reset();
	engine_.reset();
	initialized_ = false;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool MattingServer::initRga() {
	rga_resize_ = helmsman::rgakit::CreateOperation<helmsman::rgakit::RgaResize>();
	return rga_resize_ != nullptr;
}

bool MattingServer::initModel(const std::string& model_path) {
	auto& logger = helmsman::utils::Logger::GetInstance();

	engine_ = createInferenceEngine();
	engine_->load(model_path);
	model_h_ = engine_->getInputHeight() > 0 ? static_cast<int>(engine_->getInputHeight()) : 288;
	model_w_ = engine_->getInputWidth() > 0 ? static_cast<int>(engine_->getInputWidth()) : 512;
	dsr_ = 512.0f / static_cast<float>(std::max(output_w_, output_h_));

	// Initialize recurrent states (r1i-r4i) — use model-reported shapes if available.
	auto shapes = engine_->getRecurrentStateShapes();
	if (shapes.size() == 4) {
		state_mgr_.init(shapes, {"r1i", "r2i", "r3i", "r4i"});
	} else {
		state_mgr_.init({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
		                {"r1i", "r2i", "r3i", "r4i"});
	}

	// Initialize frontend (preprocessing) and backend (postprocessing).
	frontend_.setOutputBinPath("/tmp/matting-server-dump");
	backend_.setOutputPath("/tmp/matting-server-dump");

	logger.Info("MattingServer: model loaded, input=" + std::to_string(model_w_) + "x" +
	                std::to_string(model_h_),
	            kMattingServerModule);
	return true;
}

bool MattingServer::initBackground(const std::string& bg_path, int w, int h) {
	cv::Mat bg_bgr;
	if (!bg_path.empty()) {
		bg_bgr = cv::imread(bg_path, cv::IMREAD_COLOR);
	}
	if (bg_bgr.empty()) {
		bg_bgr = cv::Mat(h, w, CV_8UC3, cv::Scalar(155, 255, 120));  // default green
	} else {
		cv::resize(bg_bgr, bg_bgr, cv::Size(w, h));
	}

	// Pre-compute background at model resolution.
	cv::Mat bg_model;
	cv::resize(bg_bgr, bg_model, cv::Size(model_w_, model_h_), 0, 0, cv::INTER_LINEAR);
	bg_model_u8_ = bg_model.clone();
	return true;
}
