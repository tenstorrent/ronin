// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void mm_init(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t out_cb_id = 16, 
    uint32_t transpose = 0);
#if 0 // TODO: Remove this
API void mm_init_once();
#endif
API void matmul_tiles(
    uint32_t c_in0, 
    uint32_t c_in1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst, 
    uint32_t transpose);

/* TODO
API void matmul_tiles_math(uint32_t idst, uint32_t num_faces);

template <uint32_t num_faces = 4>
void matmul_tiles_math(uint32_t idst) {
    matmul_tiles_math(idst, num_faces);
}
*/

API void mm_init_short(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t transpose = 0);
API void mm_init_short_with_dt(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t c_in_old_srca = 2, 
    uint32_t transpose = 0);
API void mm_block_init(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t out_cb_id = 16, 
    uint32_t transpose = 0, 
    uint32_t ct_dim = 1, 
    uint32_t rt_dim = 1, 
    uint32_t kt_dim = 1);
API void matmul_block(
    uint32_t in0_cb_id, 
    uint32_t in1_cb_id, 
    uint32_t in0_tile_index, 
    uint32_t in1_tile_index, 
    uint32_t idst, 
    uint32_t transpose,
    uint32_t ct_dim, 
    uint32_t rt_dim, 
    uint32_t kt_dim);
API void mm_block_init_short(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t transpose = 0, 
    uint32_t ct_dim = 1, 
    uint32_t rt_dim = 1, 
    uint32_t kt_dim = 1);
API void mm_block_init_short_with_dt(
    uint32_t in0_cb_id = 0, 
    uint32_t in1_cb_id = 1, 
    uint32_t old_in1_cb_id = 2, 
    uint32_t transpose = 0, 
    uint32_t ct_dim = 1, 
    uint32_t rt_dim = 1, 
    uint32_t kt_dim = 1);

} // namespace ckernel

