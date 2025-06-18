// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/assert.hpp"
#include "tt_metal/impl/dispatch/worker_config_buffer.hpp"

namespace tt {

namespace tt_metal {

constexpr uint32_t kernel_config_entry_count = 8;

WorkerConfigBufferMgr::WorkerConfigBufferMgr() {

    entries_.resize(kernel_config_entry_count);
}

void WorkerConfigBufferMgr::init_add_core(uint32_t base_addr, uint32_t size) {

    this->base_addrs_.push_back(base_addr);
    this->end_addrs_.push_back(base_addr + size);

    for (auto& entry : this->entries_) {
        entry.push_back({0, 0});
    }

    // when free == alloc, buffer is empty
    // entries[alloc_index].addr is always the next address
    this->alloc_index_.push_back(0);
    this->free_index_.push_back(0);
    this->entries_[0].back().addr = base_addr;

    this->reservation_.push_back({});
}

// First part of returned pair is true if reserving size bytes requires a sync on some core type
// The vector contains whether or not the core type needs a sync and if so the sync value
// To avoid allocs in a perf path, returns a reference to internal data
const std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&> WorkerConfigBufferMgr::reserve(
    const std::vector<uint32_t>& sizes) {

    ConfigBufferSync sync_info;
    sync_info.need_sync = false;

    size_t num_core_types = this->reservation_.size();
    TT_ASSERT(sizes.size() == num_core_types);
    for (uint32_t idx = 0; idx < num_core_types; idx++) {
        uint32_t free_index = this->free_index_[idx];
        uint32_t alloc_index = this->alloc_index_[idx];

        bool done = false;
        while (!done) {
            done = true;

            uint32_t size = sizes[idx];
            uint32_t addr = this->entries_[alloc_index][idx].addr;

            this->reservation_[idx].size = size;
            if (size == 0) {
                this->reservation_[idx].addr = addr;
                break;
            }
            TT_ASSERT(size <= this->end_addrs_[idx] - this->base_addrs_[idx]);

            // alloc_index may be ahead or behind free_index
            // so compare to either end of buffer or next to be freed addr
            uint32_t end = (addr >= this->entries_[free_index][idx].addr) ?
                this->end_addrs_[idx] :
                this->entries_[free_index][idx].addr;

            if (addr + size > end && end == this->end_addrs_[idx]) {
                // Wrap the ring buffer
                addr = this->base_addrs_[idx];
                end = this->entries_[free_index][idx].addr;
            }

            if (addr + size > end) {
                // Need a sync...but will this entry free enough space?  Look at the next
                uint32_t next_free_index = free_index + 1;
                if (next_free_index == kernel_config_entry_count) {
                    next_free_index = 0;
                }

                if (next_free_index == alloc_index) {
                    // The sync will free the whole buffer, reset to the top
                    addr = this->base_addrs_[idx];
                } else {
                    uint32_t next_end = (addr >= this->entries_[next_free_index][idx].addr) ?
                        this->end_addrs_[idx] :
                        this->entries_[next_free_index][idx].addr;
                    if (addr + size > next_end) {
                        // Need to free multiple entries
                        // Move the free index forward to the next entry and retry
                        free_index = next_free_index;
                        done = false;
                        continue;
                    }
                }

                sync_info.need_sync = true;
            } else if (alloc_index + 1 == free_index ||
                       (alloc_index + 1 == kernel_config_entry_count && free_index == 0)) {
                // We need a sync because the table of entries is too small
                sync_info.need_sync = true;
            }

            sync_info.sync_count = this->entries_[free_index][idx].sync_count;
            this->reservation_[idx].addr = addr;
        }
    }

    return std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&>(sync_info, this->reservation_);
}

// Repeatedly move free_index up until it catches up w/ the reserved sync_counts or alloc_index
void WorkerConfigBufferMgr::free(uint32_t free_up_to_sync_count) {

    size_t num_core_types = this->reservation_.size();
    for (uint32_t idx = 0; idx < num_core_types; idx++) {
        uint32_t free_index = this->free_index_[idx];
        if (free_up_to_sync_count >= this->entries_[free_index][idx].sync_count) {
            if (free_index != this->alloc_index_[idx]) {
                free_index++;
                if (free_index == kernel_config_entry_count) {
                    free_index = 0;
                }
                this->free_index_[idx] = free_index;
            }
        }
    }
}

void WorkerConfigBufferMgr::alloc(uint32_t when_freeable_sync_count) {

    size_t num_core_types = this->reservation_.size();
    for (uint32_t idx = 0; idx < num_core_types; idx++) {
        uint32_t alloc_index = this->alloc_index_[idx];

        this->entries_[alloc_index][idx].addr = this->reservation_[idx].addr;
        this->entries_[alloc_index][idx].size = this->reservation_[idx].size;
        this->entries_[alloc_index][idx].sync_count = when_freeable_sync_count;

        uint32_t old_alloc_index = alloc_index;
        alloc_index++;
        if (alloc_index == kernel_config_entry_count) {
            alloc_index = 0;
        }

        this->entries_[alloc_index][idx].addr = this->entries_[old_alloc_index][idx].addr + this->entries_[old_alloc_index][idx].size;
        this->entries_[alloc_index][idx].size = 0;
        this->entries_[alloc_index][idx].sync_count = 0xbabababa; // debug

        this->alloc_index_[idx] = alloc_index;
    }
}

uint32_t WorkerConfigBufferMgr::get_last_slot_addr(HalProgrammableCoreType programmable_core_type) const {
    // TODO: support all programmable core types?
    TT_ASSERT(programmable_core_type != HalProgrammableCoreType::IDLE_ETH);
    uint32_t index = static_cast<uint32_t>(programmable_core_type);
    return this->reservation_[index].addr;
}

}  // namespace tt_metal

}  // namespace tt
