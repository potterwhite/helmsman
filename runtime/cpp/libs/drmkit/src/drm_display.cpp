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
// drm_display.cpp — minimal legacy KMS modeset + dumb-buffer present
//
// Flow on Init():
//   1. open(card_path)                              → fd
//   2. drmSetClientCap(DRM_CLIENT_CAP_UNIVERSAL_PLANES)  (best effort)
//   3. drmModeGetResources                           → list connectors / crtcs / encoders
//   4. pick first connected connector with a non-empty mode list
//   5. choose preferred mode (or first mode)
//   6. find a usable encoder + crtc for that connector
//   7. drmModeCreateDumb (XRGB8888, mode.hdisplay × mode.vdisplay)
//   8. drmModeAddFB (24 bpp depth, 32 bpp pixel)
//   9. drmModeMapDumb + mmap                         → cpu-visible pointer
//  10. save current crtc (drmModeGetCrtc) for restore
//  11. drmModeSetCrtc to install our FB on the picked CRTC + connector
//
// Flow on Close():
//   - drmModeSetCrtc(saved_crtc) to restore tty
//   - munmap, drmModeRmFB, drmModeDestroyDumb, drmModeFreeCrtc, close(fd)
//
// Notes
//   - Legacy path (no atomic). On success the dumb buffer is the active scanout
//     surface; ShowARGB just memcpys into the same mapping and the panel scans
//     out the new contents at the next vblank. No explicit page flip needed.
//   - On failure during Init, every resource acquired so far is released and
//     IsOpen() returns false.
//   - All log lines go to stderr (this lib is independent of the project's
//     Logger to keep deps minimal — caller can wrap if it wants Logger output).
// =============================================================================

#include "DRMKit/drm_display.h"
#include "DRMKit/pch.h"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

namespace helmsman {
namespace drmkit {

namespace {

constexpr const char* kDefaultCardPath = "/dev/dri/card0";

void log_err(const char* what) {
    std::fprintf(stderr, "[DRMKit] %s failed: %s (errno=%d)\n", what, std::strerror(errno), errno);
}

void log_info(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[DRMKit] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

}  // namespace

// =============================================================================
// PIMPL: holds every libdrm handle so the public header stays libdrm-free.
// =============================================================================
struct DrmDisplay::Impl {
    int          fd          = -1;
    uint32_t     crtc_id     = 0;
    uint32_t     conn_id     = 0;
    uint32_t     fb_id       = 0;
    uint32_t     dumb_handle = 0;
    uint32_t     pitch       = 0;
    uint64_t     buffer_size = 0;
    uint8_t*     map_ptr     = nullptr;
    int          width       = 0;
    int          height      = 0;
    drmModeCrtc* saved_crtc  = nullptr;
    drmModeModeInfo mode{};

    bool init(int req_w, int req_h, const char* card_path);
    void close();
    bool is_open() const { return fd >= 0 && map_ptr != nullptr; }
};

// -----------------------------------------------------------------------------
bool DrmDisplay::Impl::init(int /*req_w*/, int /*req_h*/, const char* card_path) {
    if (card_path == nullptr || *card_path == '\0') {
        card_path = kDefaultCardPath;
    }

    fd = ::open(card_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        log_err("open(card)");
        return false;
    }
    log_info("opened %s (fd=%d)", card_path, fd);

    // Best-effort: enables universal-planes naming. Not strictly required for
    // single-plane legacy SetCrtc, but harmless.
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0) {
        // Not fatal — only some drivers care.
        log_info("drmSetClientCap(UNIVERSAL_PLANES) not supported (ok)");
    }

    drmModeRes* res = drmModeGetResources(fd);
    if (!res) {
        log_err("drmModeGetResources");
        close();
        return false;
    }

    // Walk connectors, take the first DRM_MODE_CONNECTED with at least one mode.
    drmModeConnector* conn = nullptr;
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            conn = c;
            break;
        }
        drmModeFreeConnector(c);
    }
    if (!conn) {
        std::fprintf(stderr, "[DRMKit] no connected connector with modes found\n");
        drmModeFreeResources(res);
        close();
        return false;
    }
    log_info("connector %u has %d mode(s); using #0", conn->connector_id, conn->count_modes);

    // Pick the connector's preferred mode if flagged, else mode #0.
    int mode_idx = 0;
    for (int i = 0; i < conn->count_modes; ++i) {
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED) {
            mode_idx = i;
            break;
        }
    }
    mode    = conn->modes[mode_idx];
    width   = mode.hdisplay;
    height  = mode.vdisplay;
    conn_id = conn->connector_id;
    log_info("selected mode: %dx%d @ %d Hz", width, height, mode.vrefresh);

    // Find a CRTC that this connector's encoder can drive.
    uint32_t picked_crtc = 0;
    if (conn->encoder_id) {
        drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoder_id);
        if (enc) {
            if (enc->crtc_id) picked_crtc = enc->crtc_id;
            drmModeFreeEncoder(enc);
        }
    }
    if (picked_crtc == 0) {
        // Fall back: scan encoders compatible with this connector and pick a
        // CRTC from the resources list whose bit appears in encoder.possible_crtcs.
        for (int i = 0; i < conn->count_encoders && picked_crtc == 0; ++i) {
            drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoders[i]);
            if (!enc) continue;
            for (int j = 0; j < res->count_crtcs; ++j) {
                if (enc->possible_crtcs & (1u << j)) {
                    picked_crtc = res->crtcs[j];
                    break;
                }
            }
            drmModeFreeEncoder(enc);
        }
    }
    if (picked_crtc == 0) {
        std::fprintf(stderr, "[DRMKit] no CRTC available for connector %u\n", conn_id);
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close();
        return false;
    }
    crtc_id = picked_crtc;
    log_info("using CRTC %u", crtc_id);

    // Save current CRTC config so Close() can restore the tty.
    saved_crtc = drmModeGetCrtc(fd, crtc_id);

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    // Allocate the dumb buffer (kernel-managed, CPU-mappable).
    drm_mode_create_dumb create{};
    create.width  = static_cast<uint32_t>(width);
    create.height = static_cast<uint32_t>(height);
    create.bpp    = 32;
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) {
        log_err("DRM_IOCTL_MODE_CREATE_DUMB");
        close();
        return false;
    }
    dumb_handle = create.handle;
    pitch       = create.pitch;
    buffer_size = create.size;
    log_info("dumb buffer: handle=%u pitch=%u size=%llu",
             dumb_handle, pitch, (unsigned long long)buffer_size);

    // Wrap the dumb buffer in a framebuffer object that the display engine can
    // scan out. Use 24-bit depth / 32 bpp to match XRGB8888 layout.
    if (drmModeAddFB(fd,
                     static_cast<uint32_t>(width),
                     static_cast<uint32_t>(height),
                     24, 32, pitch, dumb_handle, &fb_id) != 0) {
        log_err("drmModeAddFB");
        close();
        return false;
    }
    log_info("framebuffer id=%u", fb_id);

    // Map it so we can write pixels from CPU.
    drm_mode_map_dumb mreq{};
    mreq.handle = dumb_handle;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
        log_err("DRM_IOCTL_MODE_MAP_DUMB");
        close();
        return false;
    }
    void* p = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   static_cast<off_t>(mreq.offset));
    if (p == MAP_FAILED) {
        log_err("mmap");
        close();
        return false;
    }
    map_ptr = static_cast<uint8_t*>(p);
    std::memset(map_ptr, 0, buffer_size);  // start clean (black)

    // Install our FB on the chosen CRTC + connector. This is the legacy
    // modeset call that takes effect immediately (with a brief blank).
    if (drmModeSetCrtc(fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode) != 0) {
        log_err("drmModeSetCrtc");
        close();
        return false;
    }
    log_info("modeset OK — display is ours");

    return true;
}

// -----------------------------------------------------------------------------
void DrmDisplay::Impl::close() {
    if (fd >= 0 && saved_crtc != nullptr && saved_crtc->mode_valid) {
        // Best-effort restore of the previous mode (typically the framebuffer
        // console). Failure is non-fatal — we are tearing down anyway.
        drmModeSetCrtc(fd, saved_crtc->crtc_id, saved_crtc->buffer_id,
                       saved_crtc->x, saved_crtc->y, &conn_id, 1, &saved_crtc->mode);
    }
    if (saved_crtc) {
        drmModeFreeCrtc(saved_crtc);
        saved_crtc = nullptr;
    }
    if (map_ptr) {
        munmap(map_ptr, buffer_size);
        map_ptr = nullptr;
    }
    if (fb_id && fd >= 0) {
        drmModeRmFB(fd, fb_id);
        fb_id = 0;
    }
    if (dumb_handle && fd >= 0) {
        drm_mode_destroy_dumb dreq{};
        dreq.handle = dumb_handle;
        drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
        dumb_handle = 0;
    }
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
    crtc_id = conn_id = 0;
    width = height = 0;
    buffer_size = 0;
    pitch = 0;
}

// =============================================================================
// Public-facing methods (delegate to Impl)
// =============================================================================
DrmDisplay::DrmDisplay() : impl_(std::make_unique<Impl>()) {}
DrmDisplay::~DrmDisplay() {
    if (impl_) impl_->close();
}
DrmDisplay::DrmDisplay(DrmDisplay&&) noexcept            = default;
DrmDisplay& DrmDisplay::operator=(DrmDisplay&&) noexcept = default;

bool DrmDisplay::Init(int req_w, int req_h, const char* card_path) {
    return impl_->init(req_w, req_h, card_path);
}

void DrmDisplay::ShowARGB(const uint8_t* argb_data) {
    if (!impl_->is_open() || argb_data == nullptr) return;
    // Fast path: the dumb-buffer pitch usually equals width*4. If not, copy
    // row by row to honour the kernel's chosen pitch.
    const size_t row_bytes = static_cast<size_t>(impl_->width) * 4u;
    if (impl_->pitch == row_bytes) {
        std::memcpy(impl_->map_ptr, argb_data, row_bytes * static_cast<size_t>(impl_->height));
    } else {
        const uint8_t* src = argb_data;
        uint8_t*       dst = impl_->map_ptr;
        for (int y = 0; y < impl_->height; ++y) {
            std::memcpy(dst, src, row_bytes);
            src += row_bytes;
            dst += impl_->pitch;
        }
    }
}

void DrmDisplay::Close() { impl_->close(); }

bool DrmDisplay::IsOpen() const { return impl_->is_open(); }

std::pair<int, int> DrmDisplay::PanelSize() const {
    return {impl_->width, impl_->height};
}

}  // namespace drmkit
}  // namespace helmsman
