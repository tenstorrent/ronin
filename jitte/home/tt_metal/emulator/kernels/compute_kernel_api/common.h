// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/unpack.h"
#include "compute_kernel_api/cb_api.h"

// borrowed from "hw/ckernels/<ARCH_NAME>/common/inc/ckernel.h"
inline uint32_t mulsi3 (uint32_t a, uint32_t b) {
    uint32_t r = 0;
    while (a) {
        if (a & 1) {
            r += b;
        }
        a >>= 1;
        b <<= 1;
    }
    return r;
}

#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_ ## arg_idx

// JIT Build flow will set this as needed.
#ifndef COMMON_RT_ARGS_OFFSET
    #define COMMON_RT_ARGS_OFFSET 0
#endif

API uint32_t get_arg_uint32(int arg_idx);

template <typename T>
T get_arg_val(int arg_idx) {
    // only uint32_t is supported
    return static_cast<T>(get_arg_uint32(arg_idx));
}

/* TODO
API uint32_t get_common_arg_uint32(int arg_idx);

template <typename T>
T get_common_arg_val(int arg_idx) {
    // only uint32_t is supported
    return static_cast<T>(get_common_arg_uint32(arg_idx));
}
*/

#define NAMESPACE ckernel

#define MAIN main()

// conventional main function for linker

namespace ckernel {

void main();

}

extern "C" int main() {
    ckernel::main();
    return 0;
}

