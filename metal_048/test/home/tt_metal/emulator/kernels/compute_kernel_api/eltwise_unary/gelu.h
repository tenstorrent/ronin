#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void gelu_tile_init(bool fast_and_approx = true);
API void gelu_tile(uint32_t idst, bool fast_and_approx = true);

} // namespace ckernel

