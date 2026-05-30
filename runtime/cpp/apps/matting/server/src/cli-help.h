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

#pragma once

#include <iostream>

inline void PrintHelp(const char* program_name) {
	std::cerr
	    << "Usage:\n"
	    << "  " << program_name << " <input> <model> <output_dir> [background] [options]\n"
	    << "  " << program_name << " --info\n"
	    << "  " << program_name << " --help\n"
	    << "\n"
	    << "Positional:\n"
	    << "  input          Input image or video file (.mp4/.avi/.mkv/.mov/.webm)\n"
	    << "  model          ONNX or RKNN model file\n"
	    << "  output_dir     Output directory for results and debug dumps\n"
	    << "  background     (optional) Background image for compositing\n"
	    << "\n"
	    << "Model:\n"
	    << "  --rvm          RVM mode with recurrent states\n"
	    << "  --modnet       MODNet single-frame matting (default)\n"
	    << "\n"
	    << "Output:\n"
	    << "  --output=mp4   Write composited video to mp4 (default)\n"
	    << "  --output=drm   Display on DRM/KMS panel (embedded only)\n"
	    << "\n"
	    << "Decode:\n"
	    << "  --hwdecoder    Hardware decode via FFmpeg + MPPKit\n"
	    << "  --no-prefetch  Disable prefetch thread (all work on main thread)\n"
	    << "\n"
	    << "NPU:\n"
	    << "  --core-mask=X  NPU core selection: auto|0|1|2|0_1|0_1_2|all (default: all)\n"
	    << "  --profile      Enable per-layer NPU profiling (COLLECT_PERF_MASK)\n"
	    << "\n"
	    << "Debug:\n"
	    << "  --timing=off   Disable pipeline timing statistics\n"
	    << "  --timing=on    Enable pipeline timing statistics (default)\n"
	    << "  --dump         Enable binary dump for debugging\n"
	    << "  --inspect      Enable diagnostic logging for internal state inspection\n"
	    << "\n"
	    << "Info:\n"
	    << "  --info         Print build/version info and exit\n"
	    << "  --help, -h     Show this help and exit\n"
	    << "\n"
	    << "Examples:\n"
	    << "  # RVM video matting with DRM output\n"
	    << "  Helmsman_Matting_Server --rvm --output=drm --core-mask=all \\\n"
	    << "      video.mp4 rvm.rknn ./output/\n"
	    << "\n"
	    << "  # MODNet single image with background compositing\n"
	    << "  Helmsman_Matting_Server photo.png modnet.onnx ./output/ bg.jpg\n";
}
