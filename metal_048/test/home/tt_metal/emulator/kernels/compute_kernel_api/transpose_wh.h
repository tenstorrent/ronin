#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void transpose_wh_init(uint32_t icb, uint32_t ocb = 16);
// API void transpose_wh_init_short(uint32_t icb); // TODO
API void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst);

} // namespace ckernel

