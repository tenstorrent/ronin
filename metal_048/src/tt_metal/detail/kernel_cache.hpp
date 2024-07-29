// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::detail
{
    struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
#if 0 // [RONIN]
        unique_lock<mutex> lock(mutex_);
#endif
        return hashes_.find(khash) != hashes_.end();
    }
    bool add(size_t khash) {
#if 0 // [RONIN]
        unique_lock<mutex> lock(mutex_);
#endif
        bool ret = false;
        if (hashes_.find(khash) == hashes_.end() ){
            hashes_.insert(khash);
            ret = true;
        }
        return ret;
    }

    bool is_bin_generated(size_t khash) {
#if 0 // [RONIN]
        unique_lock<mutex> lock(mutex_);
#endif
        return generated_bins_.find(khash) != generated_bins_.end();
    }

    void add_generated_bin(size_t khash) {
#if 0 // [RONIN]
        unique_lock<mutex> lock(mutex_);
#endif
        generated_bins_.insert(khash);
    }

    void clear() {
#if 0 // [RONIN]
        unique_lock<mutex> lock(mutex_);
#endif
        hashes_.clear();
        generated_bins_.clear();
    }

    private:
#if 0 // [RONIN]
        std::mutex mutex_;
#endif
        std::unordered_set<size_t > hashes_;
        std::unordered_set<size_t > generated_bins_;
    };


    /**
     * Clear the current kernel compilation cache.
     *
     * Return value: void
     */
    inline void ClearKernelCache()
    {
        detail::HashLookup::inst().clear();
    }
}
