/*
 * Copyright (c) 2026 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

// ---------------------------------------------------------------------------
// Utils/timing/timer.h — header-only pipeline timing utilities
//
// Three tools, zero external dependencies (std::chrono + std::mutex only):
//
//   ScopedTimer       RAII guard: prints elapsed time via Logger on destruction.
//                     Cheapest call-site: just construct one local variable.
//
//   ManualTimer       Explicit start() / elapsed_ms() / stop().
//                     Use when you need to measure across multiple scopes or
//                     want to read the elapsed time without ending the timer.
//
//   StageAccumulator  Thread-safe collector for per-frame (or per-iteration)
//                     durations. Call record(ms) from any thread, then call
//                     report() once at the end to log min / avg / max / count.
//                     Designed for the prefetch-worker vs. main-thread pattern.
//
// Usage example (pipeline.cpp):
//
//   // --- whole-function guard ---
//   ScopedTimer total_timer("runRVM total", timing_enabled_, logger, kModule);
//
//   // --- per-frame infer ---
//   StageAccumulator infer_acc("infer");
//   ManualTimer      t;
//   t.start();
//   engine_->infer(inputs, outputs);
//   infer_acc.record(t.elapsed_ms());
//
//   // --- at the end ---
//   infer_acc.report(timing_enabled_, logger, kModule);
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "Utils/logger/logger.h"

namespace arcforge::utils::timing {

// ============================================================================
// ManualTimer
//
// Lightweight start/stop timer. Does NOT print anything by itself.
// Call elapsed_ms() at any point after start() — it does NOT stop the timer.
// Call stop() to finalize (also returns elapsed ms).
// ============================================================================
class ManualTimer {
   public:
    void start() {
        t0_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    // Returns elapsed milliseconds since start(). Does not stop the timer.
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

    // Stops the timer and returns elapsed ms.
    double stop() {
        double ms = elapsed_ms();
        running_ = false;
        return ms;
    }

    bool is_running() const { return running_; }

   private:
    std::chrono::high_resolution_clock::time_point t0_;
    bool running_ = false;
};

// ============================================================================
// ScopedTimer
//
// RAII guard. On construction: records start time.
// On destruction: calls logger.Info() with elapsed ms — but ONLY when
// timing_enabled is true. When disabled the destructor is a no-op, so the
// compiler can eliminate most of the overhead.
// ============================================================================
class ScopedTimer {
   public:
    // label          Human-readable stage name, e.g. "runRVM total"
    // timing_enabled Reference to Pipeline::timing_enabled_ — if false, silent.
    // logger         The Logger singleton reference.
    // module         kcurrent_module_name passed through.
    ScopedTimer(std::string label,
                bool timing_enabled,
                arcforge::embedded::utils::Logger& logger,
                std::string_view module)
        : label_(std::move(label)),
          enabled_(timing_enabled),
          logger_(logger),
          module_(module) {
        if (enabled_) {
            t0_ = std::chrono::high_resolution_clock::now();
        }
    }

    ~ScopedTimer() {
        if (!enabled_) return;
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        logger_.Info("[Timing] " + label_ + ": " + std::to_string(ms) + " ms", module_);
    }

    // Non-copyable, non-movable — lifetime must be scoped.
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&)                 = delete;
    ScopedTimer& operator=(ScopedTimer&&)      = delete;

   private:
    std::string  label_;
    bool         enabled_;
    arcforge::embedded::utils::Logger& logger_;
    std::string_view module_;
    std::chrono::high_resolution_clock::time_point t0_;
};

// ============================================================================
// StageAccumulator
//
// Thread-safe collector for repeated duration samples (one per frame/iter).
// Intended for multi-threaded pipelines where producer and consumer run on
// different threads and both want to report statistics at the end.
//
// Typical pattern:
//
//   StageAccumulator preprocess_acc("worker::preprocess");
//
//   // Inside prefetch worker thread:
//   ManualTimer t; t.start();
//   result = frontend_.preprocess(...);
//   preprocess_acc.record(t.stop());
//
//   // After join, on main thread:
//   preprocess_acc.report(timing_enabled_, logger, kModule);
// ============================================================================
class StageAccumulator {
   public:
    explicit StageAccumulator(std::string stage_name)
        : name_(std::move(stage_name)) {}

    // Thread-safe: may be called from any thread.
    void record(double ms) {
        std::lock_guard<std::mutex> lk(mtx_);
        samples_.push_back(ms);
    }

    // Print min / avg / max / count via logger. NOT thread-safe — call only
    // after all producer threads have been joined.
    void report(bool timing_enabled,
                arcforge::embedded::utils::Logger& logger,
                std::string_view module) const {
        if (!timing_enabled) return;
        if (samples_.empty()) {
            logger.Info("[Timing] " + name_ + ": no samples recorded.", module);
            return;
        }

        double sum  = 0.0;
        double vmin = std::numeric_limits<double>::max();
        double vmax = std::numeric_limits<double>::lowest();
        for (double s : samples_) {
            sum  += s;
            vmin  = std::min(vmin, s);
            vmax  = std::max(vmax, s);
        }
        double avg = sum / static_cast<double>(samples_.size());

        logger.Info("[Timing] " + name_ +
                        " — count: " + std::to_string(samples_.size()) +
                        "  min: "    + std::to_string(vmin) + " ms" +
                        "  avg: "    + std::to_string(avg)  + " ms" +
                        "  max: "    + std::to_string(vmax) + " ms",
                    module);
    }

    std::size_t count() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return samples_.size();
    }

   private:
    std::string          name_;
    mutable std::mutex   mtx_;
    std::vector<double>  samples_;
};

}  // namespace arcforge::utils::timing
