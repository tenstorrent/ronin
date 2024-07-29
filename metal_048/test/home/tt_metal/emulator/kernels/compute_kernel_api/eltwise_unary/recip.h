#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void recip_tile_init();
API void recip_tile(uint32_t idst);

} // namespace ckernel

