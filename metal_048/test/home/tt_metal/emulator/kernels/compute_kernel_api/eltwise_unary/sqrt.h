#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void sqrt_tile_init();
API void sqrt_tile(uint32_t idst);

} // namespace ckernel

