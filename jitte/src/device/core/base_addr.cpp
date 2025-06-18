// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "core/memory.hpp"
#include "core/base_addr.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

constexpr uint32_t MAILBOX_BASE_ADDR = 16;

uint32_t get_mailbox_uint32(Memory *mem, uint32_t offset) {
    uint32_t addr = MAILBOX_BASE_ADDR + offset;
    uint32_t *ptr = reinterpret_cast<uint32_t *>(mem->map_addr(addr));
    return *ptr;
}

uint32_t get_mailbox_uint16(Memory *mem, uint32_t offset) {
    uint32_t addr = MAILBOX_BASE_ADDR + offset;
    uint16_t *ptr = reinterpret_cast<uint16_t *>(mem->map_addr(addr));
    return uint32_t(*ptr);
}

uint32_t get_kernel_config_base(Memory *mem) {
    return get_mailbox_uint32(mem, 26);
}

uint32_t *get_arg_base(Memory *mem, uint32_t offset) {
    uint32_t kernel_config_base = get_kernel_config_base(mem);
    uint32_t rta_offset = get_mailbox_uint16(mem, offset);
    return reinterpret_cast<uint32_t *>(mem->map_addr(kernel_config_base + rta_offset));
}

} // namespace

//
//    BaseAddr
//

uint32_t *BaseAddr::get_brisc_arg_base(Memory *mem) {
    return get_arg_base(mem, 46);
}

uint32_t *BaseAddr::get_ncrisc_arg_base(Memory *mem) {
    return get_arg_base(mem, 50);
}

uint32_t *BaseAddr::get_trisc_arg_base(Memory *mem) {
    return get_arg_base(mem, 54);
}

uint32_t *BaseAddr::get_cb_base(Memory *mem) {
    uint32_t kernel_config_base = get_kernel_config_base(mem);
    uint32_t cb_offset = get_mailbox_uint16(mem, 44);
    return reinterpret_cast<uint32_t *>(mem->map_addr(kernel_config_base + cb_offset));
}

uint32_t BaseAddr::get_semaphore_base(Memory *mem) {
    // caller needs address in L1 rather than pointer
    uint32_t kernel_config_base = get_kernel_config_base(mem);
    uint32_t sem_offset = get_mailbox_uint16(mem, 38);
    return kernel_config_base + sem_offset;
}

} // namespace device
} // namespace metal
} // namespace tt

