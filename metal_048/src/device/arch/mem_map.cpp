
#include <cstdint>
#include <cstdio>

#include "arch/mem_map.hpp"

namespace tt {
namespace metal {
namespace device {

//
//    MemMap
//

MemMap::MemMap():
        m_l1_base(0),
        m_l1_size(0),
        m_eth_base(0),
        m_eth_size(0),
        m_local_base(0),
        m_brisc_local_size(0),
        m_ncrisc_local_size(0),
        m_trisc_local_size(0),
        m_ncrisc_iram_base(0),
        m_ncrisc_iram_size(0),
        m_boot_code_size(0),
        m_brisc_firmware_size(0),
        m_ncrisc_firmware_size(0),
        m_trisc0_size(0),
        m_trisc1_size(0),
        m_trisc2_size(0),
        m_zeros_size(0),
        m_boot_code_base(0),
        m_mailbox_base(0),
        m_mailbox_end(0),
        m_zeros_base(0),
        m_brisc_firmware_base(0),
        m_ncrisc_firmware_base(0),
        m_trisc0_base(0),
        m_trisc1_base(0),
        m_trisc2_base(0),
        m_ncrisc_halt_stack_mailbox_address(0),
        m_slave_run_mailbox_address(0),
        m_brisc_init_local_l1_base(0),
        m_ncrisc_init_local_l1_base(0),
        m_trisc0_init_local_l1_base(0),
        m_trisc1_init_local_l1_base(0),
        m_trisc2_init_local_l1_base(0),
        m_ncrisc_init_iram_l1_base(0),
        m_brisc_stack_size(0),
        m_ncrisc_stack_size(0),
        m_trisc0_stack_size(0),
        m_trisc1_stack_size(0),
        m_trisc2_stack_size(0),
        m_brisc_stack_base(0),
        m_ncrisc_stack_base(0),
        m_trisc0_stack_base(0),
        m_trisc1_stack_base(0),
        m_trisc2_stack_base(0) { }

MemMap::~MemMap() { }

#define PRINT_FIELD(name) printf("    %s = 0x%x (%d)\n", #name, m_##name, m_##name);

void MemMap::diag_print() {
    printf("MEM_MAP\n");
    PRINT_FIELD(l1_base)
    PRINT_FIELD(l1_size)
    PRINT_FIELD(eth_base)
    PRINT_FIELD(eth_size)
    PRINT_FIELD(local_base)
    PRINT_FIELD(brisc_local_size)
    PRINT_FIELD(ncrisc_local_size)
    PRINT_FIELD(trisc_local_size)
    PRINT_FIELD(ncrisc_iram_base)
    PRINT_FIELD(ncrisc_iram_size)
    PRINT_FIELD(boot_code_size)
    PRINT_FIELD(brisc_firmware_size)
    PRINT_FIELD(ncrisc_firmware_size)
    PRINT_FIELD(trisc0_size)
    PRINT_FIELD(trisc1_size)
    PRINT_FIELD(trisc2_size)
    PRINT_FIELD(zeros_size)
    PRINT_FIELD(boot_code_base)
    PRINT_FIELD(mailbox_base)
    PRINT_FIELD(mailbox_end)
    PRINT_FIELD(zeros_base)
    PRINT_FIELD(brisc_firmware_base)
    PRINT_FIELD(ncrisc_firmware_base)
    PRINT_FIELD(trisc0_base)
    PRINT_FIELD(trisc1_base)
    PRINT_FIELD(trisc2_base)
    PRINT_FIELD(ncrisc_halt_stack_mailbox_address)
    PRINT_FIELD(slave_run_mailbox_address)
    PRINT_FIELD(brisc_init_local_l1_base)
    PRINT_FIELD(ncrisc_init_local_l1_base)
    PRINT_FIELD(trisc0_init_local_l1_base)
    PRINT_FIELD(trisc1_init_local_l1_base)
    PRINT_FIELD(trisc2_init_local_l1_base)
    PRINT_FIELD(ncrisc_init_iram_l1_base)
    PRINT_FIELD(brisc_stack_size)
    PRINT_FIELD(ncrisc_stack_size)
    PRINT_FIELD(trisc0_stack_size)
    PRINT_FIELD(trisc1_stack_size)
    PRINT_FIELD(trisc2_stack_size)
    PRINT_FIELD(brisc_stack_base)
    PRINT_FIELD(ncrisc_stack_base)
    PRINT_FIELD(trisc0_stack_base)
    PRINT_FIELD(trisc1_stack_base)
    PRINT_FIELD(trisc2_stack_base)
}

} // namespace device
} // namespace metal
} // namespace tt

