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

// =============================================================================
// drm_test_red.cpp — fill the panel red for ~5 s, then exit cleanly
//
// What it proves:
//   * libdrm is reachable in the sysroot
//   * drmkit::DrmDisplay::Init successfully picks a connector + CRTC + mode
//   * the dumb framebuffer is mapped and visible on screen
//   * Close() restores the previous mode (you should land back in the tty)
//
// Usage:
//   drm_test_red [card_path]            # default /dev/dri/card0
// =============================================================================

#include "DRMKit/drm_display.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const char* card = (argc > 1) ? argv[1] : "";

    helmsman::drmkit::DrmDisplay disp;
    if (!disp.Init(/*req_w*/ 0, /*req_h*/ 0, card)) {
        std::fprintf(stderr, "DrmDisplay.Init failed\n");
        return 1;
    }
    auto [w, h] = disp.PanelSize();
    std::fprintf(stderr, "panel = %dx%d\n", w, h);

    // Build a pure red ARGB frame (XRGB8888 little-endian → bytes B,G,R,X).
    const size_t W = static_cast<size_t>(w);
    const size_t H = static_cast<size_t>(h);
    std::vector<uint8_t> frame(W * H * 4u, 0);
    for (size_t i = 0; i < W * H; ++i) {
        frame[i * 4u + 0u] = 0x00;  // B
        frame[i * 4u + 1u] = 0x00;  // G
        frame[i * 4u + 2u] = 0xFF;  // R
        frame[i * 4u + 3u] = 0xFF;  // X / A (ignored)
    }

    disp.ShowARGB(frame.data());
    std::fprintf(stderr, "showing red for 5 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(5));

    disp.Close();
    std::fprintf(stderr, "closed cleanly\n");
    return 0;
}
