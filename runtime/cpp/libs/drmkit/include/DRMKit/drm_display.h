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
// DRMKit/drm_display.h — Minimal DRM/KMS legacy display sink
//
// Purpose
//   Push a CPU-side ARGB buffer to the panel directly via /dev/dri/card*
//   using legacy modesetting (drmModeSetCrtc). One connector, one CRTC,
//   one dumb buffer — no atomic, no overlay, no zero-copy.
//
//   The point of this library is to make DRM/KMS familiar so it can be
//   refactored later (multi-plane, dmabuf import, atomic commit) without
//   pulling in libdrm headers at every call site.
//
// Usage
//   drmkit::DrmDisplay disp;
//   if (!disp.Init(960, 544)) return -1;   // logical request size
//   const auto [w, h] = disp.PanelSize();  // actual mode locked-in
//   std::vector<uint8_t> argb(w * h * 4);  // build your frame
//   disp.ShowARGB(argb.data());
//   disp.Close();                           // restores previous mode
//
// Pixel format
//   ShowARGB takes packed ARGB8888 (DRM_FORMAT_XRGB8888 internally; the
//   alpha byte is ignored by single-plane output). Stride = width * 4.
//
// Thread-safety
//   Not thread-safe. Hold one DrmDisplay per thread.
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

namespace helmsman {
namespace drmkit {

class DrmDisplay {
public:
    DrmDisplay();
    ~DrmDisplay();

    // Movable, non-copyable.
    DrmDisplay(const DrmDisplay&)            = delete;
    DrmDisplay& operator=(const DrmDisplay&) = delete;
    DrmDisplay(DrmDisplay&&) noexcept;
    DrmDisplay& operator=(DrmDisplay&&) noexcept;

    // Open /dev/dri/card0, find the first connected connector with a CRTC,
    // allocate a dumb buffer matching the panel's preferred mode, and switch
    // to it. The (req_w, req_h) parameters are advisory — the picked panel
    // mode is returned via PanelSize() and is the size you must feed to
    // ShowARGB. Returns true on success.
    //
    // If the card path is empty, /dev/dri/card0 is used.
    bool Init(int req_w, int req_h, const char* card_path = "");

    // Copy width*height*4 bytes from argb_data into the dumb buffer. The
    // mapping is persistent so this is one memcpy + the kernel sees the
    // change at the next vblank (no explicit page flip in the legacy path).
    // Caller's buffer must be exactly PanelSize().first * PanelSize().second * 4.
    void ShowARGB(const uint8_t* argb_data);

    // Restore the original mode/CRTC, unmap, and close the card fd.
    // Idempotent. Called automatically by destructor.
    void Close();

    bool IsOpen() const;

    // Actual panel mode in use. Valid only after Init() succeeds.
    std::pair<int, int> PanelSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace drmkit
}  // namespace helmsman
