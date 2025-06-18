// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include "tt_metal/common/base.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/impl/dispatch/worker_config_buffer.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/llrt/hal.hpp"

using namespace tt::tt_metal;

// todo consider moving these to dispatch_addr_map
static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30; // 1GB;
static constexpr uint32_t MAX_DEV_CHANNEL_SIZE = 1 << 28; // 256 MB;
static constexpr uint32_t DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE / MAX_DEV_CHANNEL_SIZE; // 256 MB;

static constexpr uint32_t MEMCPY_ALIGNMENT = sizeof(__m128i);

struct dispatch_constants {
   public:
    dispatch_constants &operator=(const dispatch_constants &) = delete;
    dispatch_constants &operator=(dispatch_constants &&other) noexcept = delete;
    dispatch_constants(const dispatch_constants &) = delete;
    dispatch_constants(dispatch_constants &&other) noexcept = delete;

    static const dispatch_constants &get(const CoreType &core_type) {
        static dispatch_constants inst = dispatch_constants(core_type);
        return inst;
    }

    static constexpr uint8_t MAX_NUM_HW_CQS = 2;
    typedef uint16_t prefetch_q_entry_type;
    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
    static constexpr uint32_t PREFETCH_Q_BASE = DISPATCH_L1_UNRESERVED_BASE;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;
    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;
    static constexpr uint32_t ISSUE_Q_ALIGNMENT = PCIE_ALIGNMENT;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
    static constexpr uint32_t DISPATCH_BUFFER_BASE =
        ((DISPATCH_L1_UNRESERVED_BASE - 1) | ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) + 1;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;
    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken
    // down into equal sized partial pages BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is
    // incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    uint32_t prefetch_q_entries() const { return prefetch_q_entries_; }

    uint32_t prefetch_q_size() const { return prefetch_q_size_; }

    uint32_t max_prefetch_command_size() const { return max_prefetch_command_size_; }

    uint32_t cmddat_q_base() const { return cmddat_q_base_; }

    uint32_t cmddat_q_size() const { return cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_pages() const { return dispatch_buffer_pages_; }

    uint32_t prefetch_d_buffer_size() const { return prefetch_d_buffer_size_; }

    uint32_t prefetch_d_buffer_pages() const { return prefetch_d_buffer_pages_; }

    uint32_t mux_buffer_size(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_size_ / num_hw_cqs; }

    uint32_t mux_buffer_pages(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_pages_ / num_hw_cqs; }

   private:
    dispatch_constants(const CoreType &core_type) {
        TT_ASSERT(core_type == CoreType::WORKER or core_type == CoreType::ETH);
        // make this 2^N as required by the packetized stages
        uint32_t dispatch_buffer_block_size;
        if (core_type == CoreType::WORKER) {
            prefetch_q_entries_ = 1534;
            max_prefetch_command_size_ = 128 * 1024;
            cmddat_q_size_ = 256 * 1024;
            scratch_db_size_ = 128 * 1024;
            dispatch_buffer_block_size = 512 * 1024;
            prefetch_d_buffer_size_ = 256 * 1024;
        } else {
            prefetch_q_entries_ = 128;
            max_prefetch_command_size_ = 32 * 1024;
            cmddat_q_size_ = 64 * 1024;
            scratch_db_size_ = 19 * 1024;
            dispatch_buffer_block_size = 128 * 1024;
            prefetch_d_buffer_size_ = 128 * 1024;
        }
        TT_ASSERT(cmddat_q_size_ >= 2 * max_prefetch_command_size_);
        TT_ASSERT(scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);

        prefetch_q_size_ = prefetch_q_entries_ * sizeof(prefetch_q_entry_type);
        cmddat_q_base_ = PREFETCH_Q_BASE + ((prefetch_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        scratch_db_base_ = cmddat_q_base_ + ((cmddat_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        const uint32_t l1_size = core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;
        TT_ASSERT(scratch_db_base_ + scratch_db_size_ < l1_size);
        dispatch_buffer_block_size_pages_ =
            dispatch_buffer_block_size / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;
        dispatch_buffer_pages_ = dispatch_buffer_block_size_pages_ * DISPATCH_BUFFER_SIZE_BLOCKS;
        uint32_t dispatch_cb_end = DISPATCH_BUFFER_BASE + (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages_;
        TT_ASSERT(dispatch_cb_end < l1_size);
        prefetch_d_buffer_pages_ = prefetch_d_buffer_size_ >> PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
    }

    uint32_t prefetch_q_entries_;
    uint32_t prefetch_q_size_;
    uint32_t max_prefetch_command_size_;
    uint32_t cmddat_q_base_;
    uint32_t cmddat_q_size_;
    uint32_t scratch_db_base_;
    uint32_t scratch_db_size_;
    uint32_t dispatch_buffer_block_size_pages_;
    uint32_t dispatch_buffer_pages_;
    uint32_t prefetch_d_buffer_size_;
    uint32_t prefetch_d_buffer_pages_;
};

