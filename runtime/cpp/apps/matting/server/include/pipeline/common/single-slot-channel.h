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

#include <condition_variable>
#include <mutex>
#include <optional>

/**
 * @brief Thread-safe single-slot rendezvous channel.
 *
 * Connects exactly one producer to one consumer with a capacity of one item.
 * The producer blocks on push() until the slot is empty; the consumer blocks
 * on pop() until an item is available.  Either side may call close() to signal
 * EOF: push() returns false and pop() returns std::nullopt after that point.
 *
 * Used in the RVM dual-buffer pipeline to pass raw frames from the main thread
 * to the prefetch worker (raw_ch) and preprocessed tensors back (tensor_ch).
 */
template <typename T>
class SingleSlotChannel {
   public:
    // -------------------------------------------------------------------------
    // push() — producer side
    //   Blocks until the slot is empty or the channel is closed.
    //   Returns true if the item was accepted, false if the channel was closed
    //   before the slot became free.
    // -------------------------------------------------------------------------
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_empty_.wait(lk, [this] { return !has_item_ || closed_; });
        if (closed_)
            return false;
        item_     = std::move(item);
        has_item_ = true;
        cv_full_.notify_one();
        return true;
    }

    // -------------------------------------------------------------------------
    // pop() — consumer side
    //   Blocks until an item is available or the channel is closed.
    //   Returns std::nullopt when the channel is closed and no item remains.
    // -------------------------------------------------------------------------
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_full_.wait(lk, [this] { return has_item_ || closed_; });
        if (!has_item_)
            return std::nullopt;
        T out     = std::move(*item_);
        item_     = std::nullopt;
        has_item_ = false;
        cv_empty_.notify_one();
        return out;
    }

    // -------------------------------------------------------------------------
    // close() — signal EOF
    //   Safe to call from either producer or consumer.
    //   Wakes all waiters so they can check the closed_ flag and exit cleanly.
    // -------------------------------------------------------------------------
    void close() {
        std::unique_lock<std::mutex> lk(mtx_);
        closed_ = true;
        cv_full_.notify_all();
        cv_empty_.notify_all();
    }

   private:
    std::mutex              mtx_;
    std::condition_variable cv_empty_;   // signalled when slot becomes free
    std::condition_variable cv_full_;    // signalled when slot becomes occupied
    std::optional<T>        item_;
    bool has_item_ = false;
    bool closed_   = false;
};
