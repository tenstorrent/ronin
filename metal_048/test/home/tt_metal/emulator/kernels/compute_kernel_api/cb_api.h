#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void cb_wait_front(uint32_t cbid, uint32_t ntiles);
API void cb_pop_front(uint32_t cbid, uint32_t ntiles);
API void cb_reserve_back(uint32_t cbid, uint32_t ntiles);
API void cb_push_back(uint32_t cbid, uint32_t ntiles);

} // namespace ckernel

