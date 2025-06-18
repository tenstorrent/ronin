// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void cb_wait_front(uint32_t cbid, uint32_t ntiles);
API void cb_pop_front(uint32_t cbid, uint32_t ntiles);
API void cb_reserve_back(uint32_t cbid, uint32_t ntiles);
API void cb_push_back(uint32_t cbid, uint32_t ntiles);

// Experimental extensions

API uint32_t get_write_ptr(uint32_t cbid);
API uint32_t get_read_ptr(uint32_t cbid);
API void set_write_ptr(uint32_t cbid, uint32_t ptr);
API void set_read_ptr(uint32_t cbid, uint32_t ptr);

} // namespace ckernel

