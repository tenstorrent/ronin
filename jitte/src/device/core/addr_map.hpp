// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {
namespace metal {
namespace device {

struct AddrMap {
    static constexpr uint32_t 
        BRISC_L1_ARG_BASE = 98 * 1024,
        BRISC_L1_RESULT_BASE = 99 * 1024,
        NCRISC_L1_ARG_BASE = 100 * 1024,
        NCRISC_L1_RESULT_BASE = 101 * 1024,
        TRISC_L1_ARG_BASE = 102 * 1024;

    // config for 32 L1 buffers is at addr BUFFER_CONFIG_BASE
    // 12 bytes for each buffer: (addr, size, size_in_tiles)
    // addr and size are in 16B words (byte address >> 4)
    // this is a total of 32 * 3 * 4 = 384B
    static constexpr uint32_t 
        CIRCULAR_BUFFER_CONFIG_BASE = 103 * 1024,
        NUM_CIRCULAR_BUFFERS = 32,
        UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4,
        CIRCULAR_BUFFER_CONFIG_SIZE = 
            NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

// TODO: Revise from this point
    // 4 semaphores per core aligned to 16B
    static constexpr uint32_t 
        SEMAPHORE_BASE = CIRCULAR_BUFFER_CONFIG_BASE + CIRCULAR_BUFFER_CONFIG_SIZE,
        NUM_SEMAPHORES = 4,
        UINT32_WORDS_PER_SEMAPHORE = 1,
        SEMAPHORE_ALIGNMENT = 16,
        ALIGNED_SIZE_PER_SEMAPHORE = 
            (((UINT32_WORDS_PER_SEMAPHORE * sizeof(uint32_t)) + SEMAPHORE_ALIGNMENT - 1) / 
                SEMAPHORE_ALIGNMENT) * SEMAPHORE_ALIGNMENT,
        SEMAPHORE_SIZE = NUM_SEMAPHORES * ALIGNED_SIZE_PER_SEMAPHORE;

    // Start of unreserved space
    static constexpr uint32_t L1_UNRESERVED_BASE = 120 * 1024;

    // Space allocated for op info, used in graph interpreter
    // So far, holds up to 10 ops
    static constexpr uint32_t 
        OP_INFO_BASE_ADDR = 109628,
        OP_INFO_SIZE = 280;

    // Command queue pointers
    static constexpr uint32_t 
        CQ_READ_PTR = 110944,        // 0x1b160
        CQ_WRITE_PTR = 110976,       // 0x1b180
        CQ_READ_TOGGLE = 111008,     // 0x1b1a0
        CQ_WRITE_TOGGLE = 111040;    // 0x1b1c0

    // Host addresses for dispatch
    static constexpr uint32_t 
        HOST_CQ_READ_PTR = 0,
        HOST_CQ_READ_TOGGLE_PTR = 32,
        HOST_CQ_FINISH_PTR = 64,
        CQ_START = 96;
};

} // namespace device
} // namespace metal
} // namespace tt

