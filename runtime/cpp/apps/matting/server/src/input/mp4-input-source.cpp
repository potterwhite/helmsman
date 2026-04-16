#include "input/mp4-input-source.h"

bool Mp4InputSource::open(const std::string& uri) {
	return cap_.open(uri);
}  // cap_.open(uri)

bool Mp4InputSource::read(cv::Mat& frame) {
	return cap_.read(frame);
}  // cap_.read(frame)

int Mp4InputSource::width() const {
	return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}  // cap_.get(CAP_PROP_FRAME_WIDTH)

int Mp4InputSource::height() const {
	return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}  // cap_.get(CAP_PROP_FRAME_HEIGHT)

double Mp4InputSource::fps() const {
	return cap_.get(cv::CAP_PROP_FPS);
}  // cap_.get(CAP_PROP_FPS)

void Mp4InputSource::close() {
	cap_.release();
}  // cap_.release()
