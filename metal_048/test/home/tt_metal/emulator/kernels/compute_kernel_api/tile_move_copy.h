#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void copy_tile_to_dst_init_short(uint32_t cbid = 0, uint32_t transpose = 0);
API void copy_tile_init();
API void copy_tile_to_dst_init_short_with_dt(
    uint32_t old_cbid, 
    uint32_t new_cbid, 
    uint32_t transpose = 0);
API void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index);
/* TODO: Remove this
API void copy_tile_matmul_partials_init_short_with_dt(uint32_t cbid);
*/
API void copy_block_matmul_partials(
    uint32_t in_cb_id, 
    uint32_t start_in_tile_index, 
    uint32_t start_dst_tile_index, 
    uint32_t ntiles);

} // namespace ckernel

