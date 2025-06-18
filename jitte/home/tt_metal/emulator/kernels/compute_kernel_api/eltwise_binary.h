// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb = 16);
API void mul_tiles_init_f();
API void mul_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1);
API void add_tiles_init_nof(); 
API void add_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1, bool acc_to_dest = false);
API void sub_tiles_init_nof(); 
API void sub_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1, bool acc_to_dest = false);
API void mul_tiles(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void add_tiles(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void sub_tiles(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

API void binary_op_specific_init(
    bool full_init, 
    EltwiseBinaryType eltwise_binary_op_type);

template<bool full_init = false, EltwiseBinaryType eltwise_binary_op_type = ELWADD>
void binary_op_specific_init() {
    binary_op_specific_init(full_init, eltwise_binary_op_type);
}

/* TODO
API void binary_dest_reuse_tiles_init(
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType binary_reuse_dest,
    uint32_t icb0)

template<
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
void binary_dest_reuse_tiles_init(uint32_t icb0) {
    binary_dest_reuse_tiles_init(eltwise_binary_type, binary_reuse_dest, icb0);
}

API void binary_dest_reuse_tiles(
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType binary_reuse_dest,
    uint32_t in_cb_id, 
    uint32_t in_tile_index, 
    uint32_t dst_tile_index);

template<
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    binary_dest_reuse_tiles(
        eltwise_binary_type,
        binary_reuse_dest,
        in_cb_id, 
        in_tile_index, 
        dst_tile_index);
}
*/

} // namespace ckernel

