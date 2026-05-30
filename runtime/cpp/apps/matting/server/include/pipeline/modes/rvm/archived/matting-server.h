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
// MattingServer — DMA buffer fd-based API for per-frame inference
//
// Provides a zero-copy interface for real-time video pipelines:
//   1. Init(model_path, bg_path, output_w, output_h) — load model, allocate buffers
//   2. ProcessFrame(input_fd, input_w, input_h) → output_fd — per-frame inference
//   3. shutdown() — release resources
//
// The input fd comes from V4L2 camera or another process (via SCM_RIGHTS).
// The output fd is a DMA buffer containing the composited BGR frame.
// Both fds can be shared between processes without copying pixel data.
//
// Usage:
//   MattingServer server;
//   server.init("/path/to/model.rknn", "", 1920, 1080);
//   int out_fd = server.ProcessFrame(v4l2_fd, 1920, 1080);
//   // pass out_fd to DRM display or another process
//   server.shutdown();
//
// =============================================================================

#pragma once

#include <memory>
#include <string>
#include "DmaKit/dma_buffer.h"
#include "RGAKit/rga_resize.h"
#include "pipeline/infra/recurrent-state-manager.h"
#include "pipeline/stages/backend/backend.h"
#include "pipeline/stages/frontend/stages/04-preprocess/preprocessor.h"
#include "pipeline/stages/inference-engine/base/inference-engine.h"

class MattingServer {
   public:
	MattingServer();
	~MattingServer();

	// Non-copyable, non-movable (owns hardware resources).
	MattingServer(const MattingServer&) = delete;
	MattingServer& operator=(const MattingServer&) = delete;

	// Load model, allocate output DMA buffer, prepare background.
	// output_w/output_h = desired output resolution (typically same as input).
	bool Init(const std::string& model_path, const std::string& bg_path,
	          int output_w, int output_h);

	// Process a single frame via DMA buffer fds.
	// input_fd: DMA buffer containing BGR888 pixels at (input_w × input_h).
	// Returns: DMA buffer fd containing composited BGR888 at (output_w × output_h).
	//          Returns -1 on failure.
	// The returned fd is valid until the next ProcessFrame() call or shutdown().
	int ProcessFrame(int input_fd, int input_w, int input_h);

	// Release all resources.
	void shutdown();

   private:
	bool InitRga();
	bool InitModel(const std::string& model_path);
	bool InitBackground(const std::string& bg_path, int w, int h);

	std::unique_ptr<InferenceEngine> engine_;
	Preprocessor frontend_;
	MattingBackend backend_;
	RecurrentStateManager state_mgr_;

	// Model dimensions (from engine)
	int model_h_ = 288;
	int model_w_ = 512;
	float dsr_ = 0.25f;

	// Output dimensions
	int output_w_ = 0;
	int output_h_ = 0;

	// Pre-computed background at model resolution
	cv::Mat bg_model_u8_;

	// RGA hardware operations
	std::unique_ptr<helmsman::rgakit::RgaResize> rga_resize_;

	// DMA buffers
	std::unique_ptr<helmsman::dmakit::DmaBuffer> dma_output_buf_;

	bool initialized_ = false;
};
