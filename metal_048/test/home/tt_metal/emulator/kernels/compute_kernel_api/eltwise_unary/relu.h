#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void relu_max_tile(uint32_t idst, uint32_t param0);
API void relu_max_tile_init();
API void relu_min_tile(uint32_t idst, uint32_t param0);
API void relu_min_tile_init();
API void relu_tile(uint32_t idst);
API void relu_tile_init();
API void leaky_relu_tile(uint32_t idst, uint32_t param0);
API void leaky_relu_tile_init();

} // namespace ckernel

