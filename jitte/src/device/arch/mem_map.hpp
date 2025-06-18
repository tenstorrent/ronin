// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {
namespace metal {
namespace device {

//
//    MemMap
//

class MemMap {
public:
    MemMap();
    ~MemMap();
public:
    uint32_t l1_base() {
        return m_l1_base;
    }
    uint32_t l1_size() {
        return m_l1_size;
    }
    uint32_t eth_base() {
        return m_eth_base;
    }
    uint32_t eth_size() {
        return m_eth_size;
    }
    uint32_t local_base() {
        return m_local_base;
    }
    uint32_t brisc_local_size() {
        return m_brisc_local_size;
    }
    uint32_t ncrisc_local_size() {
        return m_ncrisc_local_size;
    }
    uint32_t trisc_local_size() {
        return m_trisc_local_size;
    }
    uint32_t ncrisc_iram_base() {
        return m_ncrisc_iram_base;
    }
    uint32_t ncrisc_iram_size() {
        return m_ncrisc_iram_size;
    }
    uint32_t boot_code_size() {
        return m_boot_code_size;
    }
    uint32_t brisc_firmware_size() {
        return m_brisc_firmware_size;
    }
    uint32_t ncrisc_firmware_size() {
        return m_ncrisc_firmware_size;
    }
    uint32_t trisc0_size() {
        return m_trisc0_size;
    }
    uint32_t trisc1_size() {
        return m_trisc1_size;
    }
    uint32_t trisc2_size() {
        return m_trisc2_size;
    }
    uint32_t zeros_size() {
        return m_zeros_size;
    }
    uint32_t boot_code_base() {
        return m_boot_code_base;
    }
    uint32_t mailbox_base() {
        return m_mailbox_base;
    }
    uint32_t mailbox_end() {
        return m_mailbox_end;
    }
    uint32_t zeros_base() {
        return m_zeros_base;
    }
    uint32_t brisc_firmware_base() {
        return m_brisc_firmware_base;
    }
    uint32_t ncrisc_firmware_base() {
        return m_ncrisc_firmware_base;
    }
    uint32_t trisc0_base() {
        return m_trisc0_base;
    }
    uint32_t trisc1_base() {
        return m_trisc1_base;
    }
    uint32_t trisc2_base() {
        return m_trisc2_base;
    }
    uint32_t ncrisc_halt_stack_mailbox_address() {
        return m_ncrisc_halt_stack_mailbox_address;
    }
    uint32_t slave_run_mailbox_address() {
        return m_slave_run_mailbox_address;
    }
    uint32_t brisc_init_local_l1_base() {
        return m_brisc_init_local_l1_base;
    }
    uint32_t ncrisc_init_local_l1_base() {
        return m_ncrisc_init_local_l1_base;
    }
    uint32_t trisc0_init_local_l1_base() {
        return m_trisc0_init_local_l1_base;
    }
    uint32_t trisc1_init_local_l1_base() {
        return m_trisc1_init_local_l1_base;
    }
    uint32_t trisc2_init_local_l1_base() {
        return m_trisc2_init_local_l1_base;
    }
    uint32_t ncrisc_init_iram_l1_base() {
        return m_ncrisc_init_iram_l1_base;
    }
    uint32_t brisc_stack_size() {
        return m_brisc_stack_size;
    }
    uint32_t ncrisc_stack_size() {
        return m_ncrisc_stack_size;
    }
    uint32_t trisc0_stack_size() {
        return m_trisc0_stack_size;
    }
    uint32_t trisc1_stack_size() {
        return m_trisc1_stack_size;
    }
    uint32_t trisc2_stack_size() {
        return m_trisc2_stack_size;
    }
    uint32_t brisc_stack_base() {
        return m_brisc_stack_base;
    }
    uint32_t ncrisc_stack_base() {
        return m_ncrisc_stack_base;
    }
    uint32_t trisc0_stack_base() {
        return m_trisc0_stack_base;
    }
    uint32_t trisc1_stack_base() {
        return m_trisc1_stack_base;
    }
    uint32_t trisc2_stack_base() {
        return m_trisc2_stack_base;
    }
    void diag_print();
protected:
    uint32_t m_l1_base;
    uint32_t m_l1_size;
    uint32_t m_eth_base;
    uint32_t m_eth_size;
    uint32_t m_local_base;
    uint32_t m_brisc_local_size;
    uint32_t m_ncrisc_local_size;
    uint32_t m_trisc_local_size;
    uint32_t m_ncrisc_iram_base;
    uint32_t m_ncrisc_iram_size;
    uint32_t m_boot_code_size;
    uint32_t m_brisc_firmware_size;
    uint32_t m_ncrisc_firmware_size;
    uint32_t m_trisc0_size;
    uint32_t m_trisc1_size;
    uint32_t m_trisc2_size;
    uint32_t m_zeros_size;
    uint32_t m_boot_code_base;
    uint32_t m_mailbox_base;
    uint32_t m_mailbox_end;
    uint32_t m_zeros_base;
    uint32_t m_brisc_firmware_base;
    uint32_t m_ncrisc_firmware_base;
    uint32_t m_trisc0_base;
    uint32_t m_trisc1_base;
    uint32_t m_trisc2_base;
    uint32_t m_ncrisc_halt_stack_mailbox_address;
    uint32_t m_slave_run_mailbox_address;
    uint32_t m_brisc_init_local_l1_base;
    uint32_t m_ncrisc_init_local_l1_base;
    uint32_t m_trisc0_init_local_l1_base;
    uint32_t m_trisc1_init_local_l1_base;
    uint32_t m_trisc2_init_local_l1_base;
    uint32_t m_ncrisc_init_iram_l1_base;
    uint32_t m_brisc_stack_size;
    uint32_t m_ncrisc_stack_size;
    uint32_t m_trisc0_stack_size;
    uint32_t m_trisc1_stack_size;
    uint32_t m_trisc2_stack_size;
    uint32_t m_brisc_stack_base;
    uint32_t m_ncrisc_stack_base;
    uint32_t m_trisc0_stack_base;
    uint32_t m_trisc1_stack_base;
    uint32_t m_trisc2_stack_base;
};

//
//    Public functions
//

MemMap *get_mem_map_grayskull();
MemMap *get_mem_map_wormhole_b0();

} // namespace device
} // namespace metal
} // namespace tt

