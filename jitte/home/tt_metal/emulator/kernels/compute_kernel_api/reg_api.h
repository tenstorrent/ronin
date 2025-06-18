// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void acquire_dst(tt::DstMode mode);
API void tile_regs_acquire();
API void tile_regs_wait();
API void release_dst(tt::DstMode mode);
API void tile_regs_commit();
API void tile_regs_release();

}// namespace ckernel

