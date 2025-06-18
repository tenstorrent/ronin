// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core/memory.hpp"

namespace tt {
namespace metal {
namespace device {

class BaseAddr {
public:
    static uint32_t *get_brisc_arg_base(Memory *mem);
    static uint32_t *get_ncrisc_arg_base(Memory *mem);
    static uint32_t *get_trisc_arg_base(Memory *mem);
    static uint32_t *get_cb_base(Memory *mem);
    static uint32_t get_semaphore_base(Memory *mem);
};

} // namespace device
} // namespace metal
} // namespace tt

