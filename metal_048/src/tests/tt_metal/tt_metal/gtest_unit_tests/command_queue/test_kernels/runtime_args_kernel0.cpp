// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t arg0 = get_arg_val<uint32_t>(0);
    uint32_t arg1 = get_arg_val<uint32_t>(1);
    uint32_t arg2 = get_arg_val<uint32_t>(2);
    uint32_t arg3 = get_arg_val<uint32_t>(3);
    uint32_t arg4 = get_arg_val<uint32_t>(4);
    uint32_t arg5 = get_arg_val<uint32_t>(5);
    uint32_t arg6 = get_arg_val<uint32_t>(6);
    uint32_t arg7 = get_arg_val<uint32_t>(7);
    uint32_t arg8 = get_arg_val<uint32_t>(8);
}
