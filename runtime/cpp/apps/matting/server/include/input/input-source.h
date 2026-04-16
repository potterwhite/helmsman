
#pragma once
#include <opencv2/core/mat.hpp>
#include <string>

// Abstract input source — Phase-5: Mp4, Phase-6: V4L2, IPC
class InputSource {
   public:
	virtual ~InputSource() = default;

	// Open the source. Returns true on success.
	virtual bool open(const std::string& uri) = 0;

	// Read next frame. Returns false when no more frames (EOF / disconnect).
	// frame is BGR, uint8, original resolution (caller handles resize).
	virtual bool read(cv::Mat& frame) = 0;

	// Optional: get source properties
	virtual int width() const { return 0; }
	virtual int height() const { return 0; }
	virtual double fps() const { return 0.0; }

	// Release resources
	virtual void close() = 0;
};