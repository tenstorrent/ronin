// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

#include "compute_kernel_api/common.h"

namespace ckernel {

API void sub_tiles_bcast_cols(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1,
    uint32_t idst);
/* TODO
API void sub_tiles_bcast_scalar(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
*/
API void mul_tiles_bcast_cols(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void mul_tiles_bcast_rows(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void add_tiles_bcast_rows(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void add_tiles_bcast_cols(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
/* TODO
API void add_tiles_bcast_scalar(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
*/

API void init_bcast(
    EltwiseBinaryType tBcastOp, 
    BroadcastType tBcastDim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t ocb);

template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void init_bcast(uint32_t icb0, uint32_t icb1, uint32_t ocb = 16) {
    init_bcast(tBcastOp, tBcastDim, icb0, icb1, ocb);
}

API void any_tiles_bcast(
    EltwiseBinaryType tBcastOp, 
    BroadcastType tBcastDim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void any_tiles_bcast(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    any_tiles_bcast(tBcastOp, tBcastDim, icb0, icb1, itile0, itile1, idst);
}

API void add_tiles_bcast(
    BroadcastType tBcastDim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

template<BroadcastType tBcastDim>
void add_tiles_bcast(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) { 
    add_tiles_bcast(tBcastDim, icb0, icb1, itile0, itile1, idst);
}

API void sub_tiles_bcast(
    BroadcastType tBcastDim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

template<BroadcastType tBcastDim>
void sub_tiles_bcast(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) { 
    sub_tiles_bcast(tBcastDim, icb0, icb1, itile0, itile1, idst);
}

API void mul_tiles_bcast(
    BroadcastType tBcastDim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

template<BroadcastType tBcastDim>
void mul_tiles_bcast(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) { 
    mul_tiles_bcast(tBcastDim, icb0, icb1, itile0, itile1, idst);
}

API void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
#if 0 // TODO: Remove this
API void add_bcast_rows_init_short_post_matmul(); // TODO: Implement in Compute
#endif
API void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
// API void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1); // TODO
API void mul_tiles_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
API void mul_tiles_bcast_scalar(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);
API void mul_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
API void mul_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
API void sub_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1);
// API void sub_tiles_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1); // TODO

} // namespace ckernel


