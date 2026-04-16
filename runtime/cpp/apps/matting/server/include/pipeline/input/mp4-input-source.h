#pragma once
#include <opencv2/videoio.hpp>
#include "input-source.h"

class Mp4InputSource : public InputSource {
   public:
	bool open(const std::string& uri) override;  // cap_.open(uri)
	bool read(cv::Mat& frame) override;          // cap_.read(frame)
	int width() const override;                  // cap_.get(CAP_PROP_FRAME_WIDTH)
	int height() const override;                 // cap_.get(CAP_PROP_FRAME_HEIGHT)
	double fps() const override;                 // cap_.get(CAP_PROP_FPS)
	void close() override;                       // cap_.release()

   private:
	cv::VideoCapture cap_;
};