// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "arch/mem_map.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

//
//    MemMapGrayskull
//

class MemMapGrayskull: public MemMap {
public:
    MemMapGrayskull();
    ~MemMapGrayskull();
};

#if 0 // TODO: Revise this
MemMapGrayskull::MemMapGrayskull() {
    m_l1_base = 0x0;
    m_l1_size = 1024 * 1024;

    m_eth_base = 0x0;
    m_eth_size = 0;

    m_local_base = 0xFFB00000;
    m_brisc_local_size = 4 * 1024;
    m_ncrisc_local_size = 4 * 1024;
    m_trisc_local_size = 2 * 1024;

    m_ncrisc_iram_base = 0xFFC00000;
    m_ncrisc_iram_size = 16 * 1024;

    m_boot_code_size = 4;
    m_brisc_firmware_size = 10 * 1024;
    m_ncrisc_firmware_size = 16 * 1024;
    m_trisc0_size = 16 * 1024;
    m_trisc1_size = 16 * 1024;
    m_trisc2_size = 16 * 1024;
    m_zeros_size = 512;

    m_boot_code_base = 0;
    m_mailbox_base = 20;
    m_mailbox_end = m_mailbox_base + 128;
    m_zeros_base = 2048;
    m_brisc_firmware_base = m_zeros_base + m_zeros_size;
    m_ncrisc_firmware_base = m_ncrisc_iram_base;
    m_trisc0_base = m_brisc_firmware_base + m_brisc_firmware_size;
    m_trisc1_base = m_trisc0_base + m_trisc0_size;
    m_trisc2_base = m_trisc1_base + m_trisc1_size;

    m_ncrisc_halt_stack_mailbox_address = m_mailbox_base + 4;
    m_slave_run_mailbox_address = m_mailbox_base + 28;

    m_brisc_init_local_l1_base = m_trisc2_base + m_trisc2_size;
    m_ncrisc_init_local_l1_base = m_brisc_init_local_l1_base + m_brisc_local_size;
    m_trisc0_init_local_l1_base = m_ncrisc_init_local_l1_base + m_ncrisc_local_size;
    m_trisc1_init_local_l1_base = m_trisc0_init_local_l1_base + m_trisc_local_size;
    m_trisc2_init_local_l1_base = m_trisc1_init_local_l1_base + m_trisc_local_size;

    m_ncrisc_init_iram_l1_base = m_trisc2_init_local_l1_base + m_trisc_local_size;

    m_brisc_stack_size = 768;
    m_ncrisc_stack_size = 768;
    m_trisc0_stack_size = 256;
    m_trisc1_stack_size = 256;
    m_trisc2_stack_size = 768;

    // emulator uses L1 instead of local memory to keep local data and stack
    m_brisc_stack_base = m_brisc_init_local_l1_base + m_brisc_local_size - m_brisc_stack_size;
    m_ncrisc_stack_base = m_ncrisc_init_local_l1_base + m_ncrisc_local_size - m_ncrisc_stack_size;
    m_trisc0_stack_base = m_trisc0_init_local_l1_base + m_trisc_local_size - m_trisc0_stack_size;
    m_trisc1_stack_base = m_trisc1_init_local_l1_base + m_trisc_local_size - m_trisc1_stack_size;
    m_trisc2_stack_base = m_trisc2_init_local_l1_base + m_trisc_local_size - m_trisc2_stack_size;

    // emulator uses L1 instead of IRAM to keep NCRISC firmware code
    m_ncrisc_firmware_base = m_ncrisc_init_iram_l1_base;
}
#endif

MemMapGrayskull::MemMapGrayskull() {
    m_l1_base = 0x0;
    m_l1_size = 1024 * 1024;

    m_eth_base = 0x0;
    m_eth_size = 0;

    m_local_base = 0xFFB00000;
    m_brisc_local_size = 4 * 1024;
    m_ncrisc_local_size = 4 * 1024;
    m_trisc_local_size = 2 * 1024;

    m_ncrisc_iram_base = 0xFFC00000;
    m_ncrisc_iram_size = 16 * 1024;

    m_boot_code_size = 4;
    m_brisc_firmware_size = 10 * 1024;
    m_ncrisc_firmware_size = 16 * 1024;
    m_trisc0_size = 16 * 1024;
    m_trisc1_size = 16 * 1024;
    m_trisc2_size = 16 * 1024;
    m_zeros_size = 512;

    m_boot_code_base = 0;
    m_mailbox_base = 16;
    uint32_t mailbox_size = 5 * 2 * 1024 + 128 + 1600;
    m_mailbox_end = m_mailbox_base + mailbox_size;
    m_zeros_base = (m_mailbox_end + 31) & ~31;
    m_brisc_firmware_base = m_zeros_base + m_zeros_size;
    m_ncrisc_firmware_base = m_ncrisc_iram_base;
    m_trisc0_base = m_brisc_firmware_base + m_brisc_firmware_size;
    m_trisc1_base = m_trisc0_base + m_trisc0_size;
    m_trisc2_base = m_trisc1_base + m_trisc1_size;

    m_ncrisc_halt_stack_mailbox_address = m_mailbox_base + 4;
    m_slave_run_mailbox_address = m_mailbox_base + 8;

    m_brisc_init_local_l1_base = m_trisc2_base + m_trisc2_size;
    m_ncrisc_init_local_l1_base = m_brisc_init_local_l1_base + m_brisc_local_size;
    m_trisc0_init_local_l1_base = m_ncrisc_init_local_l1_base + m_ncrisc_local_size;
    m_trisc1_init_local_l1_base = m_trisc0_init_local_l1_base + m_trisc_local_size;
    m_trisc2_init_local_l1_base = m_trisc1_init_local_l1_base + m_trisc_local_size;

    m_ncrisc_init_iram_l1_base = m_trisc2_init_local_l1_base + m_trisc_local_size;

    m_brisc_stack_size = 752;
    m_ncrisc_stack_size = 884;
    m_trisc0_stack_size = 320;
    m_trisc1_stack_size = 256;
    m_trisc2_stack_size = 768;

    // emulator uses L1 instead of local memory to keep local data and stack
    m_brisc_stack_base = m_brisc_init_local_l1_base + m_brisc_local_size - m_brisc_stack_size;
    m_ncrisc_stack_base = m_ncrisc_init_local_l1_base + m_ncrisc_local_size - m_ncrisc_stack_size;
    m_trisc0_stack_base = m_trisc0_init_local_l1_base + m_trisc_local_size - m_trisc0_stack_size;
    m_trisc1_stack_base = m_trisc1_init_local_l1_base + m_trisc_local_size - m_trisc1_stack_size;
    m_trisc2_stack_base = m_trisc2_init_local_l1_base + m_trisc_local_size - m_trisc2_stack_size;

    // emulator uses L1 instead of IRAM to keep NCRISC firmware code
    m_ncrisc_firmware_base = m_ncrisc_init_iram_l1_base;
}

MemMapGrayskull::~MemMapGrayskull() { }

MemMapGrayskull g_mem_map_grayskull;

} // namespace

//
//    Public functions
//

MemMap *get_mem_map_grayskull() {
    return &g_mem_map_grayskull;
}

} // namespace device
} // namespace metal
} // namespace tt

