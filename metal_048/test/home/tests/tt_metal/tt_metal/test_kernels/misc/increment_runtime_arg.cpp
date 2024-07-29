// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

namespace {
void kernel_main() {

    // Get configurable number of unique and common runtime args, and increment them all in place by a fixed value.
    constexpr uint32_t rt_args_base = get_arg_addr(0);
    constexpr uint32_t num_unique_rt_args = get_compile_time_arg_val(0);
    constexpr uint32_t num_common_rt_args = get_compile_time_arg_val(1);
    constexpr uint32_t unique_arg_incr_val = 10;
    constexpr uint32_t common_arg_incr_val = 100;

    // DPRINT << "increment_runtime_arg: " << num_unique_rt_args << " " << num_common_rt_args << " " << unique_arg_incr_val << " " << common_arg_incr_val << " ";
    // DPRINT << rt_args_base << " " << COMMON_RT_ARGS_OFFSET << ENDL();

    // This verifies get_arg_val and get_common_arg_val APIs
    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        uint32_t rt_arg = get_arg_val<uint32_t>(i);
        volatile tt_l1_ptr std::uint32_t* arg_ptr = (volatile tt_l1_ptr uint32_t*)(rt_args_base + (i * 4));
        arg_ptr[0] = rt_arg + unique_arg_incr_val;
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        uint32_t rt_arg = get_common_arg_val<uint32_t>(i);
        volatile tt_l1_ptr std::uint32_t* arg_ptr = (volatile tt_l1_ptr uint32_t*)(rt_args_base + COMMON_RT_ARGS_OFFSET + (i * 4));
        arg_ptr[0] = rt_arg + common_arg_incr_val;
    }

}
}
