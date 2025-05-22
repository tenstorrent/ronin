// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/untilize.h"

#include "compute_kernel_api.h"

#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/elu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/erfinv.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/i0.h"
#include "compute_kernel_api/eltwise_unary/isinf_isnan.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"

// temporary solution
#define FAST_AND_APPROX false

typedef uint32_t uint32;

struct Global {
    uint32 addr;
    uint32 log2_page_size;
};

struct Local {
    uint32 addr;
};

struct Pipe {
    uint32 cb_id;
    uint32 frame_size;
};

struct Semaphore {
    uint32 addr;
};

inline void tanto_max_tile(uint32_t idst) {
    max_tile(idst, idst + 1);
}

API void print_uint32(uint32 arg);

API void tanto_compute_init();
API void tanto_copy_init();
API void tanto_add_init();
API void tanto_sub_init();
API void tanto_mul_init();
API void tanto_add_bcast_rows_init();
API void tanto_sub_bcast_rows_init();
API void tanto_mul_bcast_rows_init();
API void tanto_add_bcast_cols_init();
API void tanto_sub_bcast_cols_init();
API void tanto_mul_bcast_cols_init();
API void tanto_add_bcast_scalar_init();
API void tanto_sub_bcast_scalar_init();
API void tanto_mul_bcast_scalar_init();
API void tanto_matmul_init(bool transpose);
API void tanto_reduce_max_rows_init();
API void tanto_reduce_max_cols_init();
API void tanto_reduce_max_scalar_init();
API void tanto_reduce_sum_rows_init();
API void tanto_reduce_sum_cols_init();
API void tanto_reduce_sum_scalar_init();
API void tanto_transpose_init();
API void tanto_tilize_block_init();
API void tanto_untilize_block_init();
API void tanto_abs_init();
API void tanto_acos_init();
API void tanto_asin_init();
API void tanto_atan_init();
API void tanto_binary_scalar_init();
API void tanto_cos_init();
API void tanto_elu_init();
API void tanto_eqz_init();
API void tanto_erf_init();
API void tanto_erfc_init();
API void tanto_erfinv_init();
API void tanto_exp_init();
API void tanto_exp2_init();
API void tanto_expm1_init();
API void tanto_gelu_init();
API void tanto_gez_init();
API void tanto_gtz_init();
API void tanto_heaviside_init();
API void tanto_i0_init();
API void tanto_isfinite_init();
API void tanto_isinf_init();
API void tanto_isnan_init();
API void tanto_isneginf_init();
API void tanto_isposinf_init();
API void tanto_leaky_relu_init();
API void tanto_lez_init();
API void tanto_log_init();
API void tanto_log_with_base_init();
API void tanto_logical_not_init();
API void tanto_ltz_init();
API void tanto_max_init();
API void tanto_nez_init();
API void tanto_power_init();
API void tanto_recip_init();
API void tanto_relu_init();
API void tanto_relu_max_init();
API void tanto_relu_min_init();
API void tanto_rsqrt_init();
API void tanto_sigmoid_init();
API void tanto_sign_init();
API void tanto_signbit_init();
API void tanto_sin_init();
API void tanto_sqrt_init();
API void tanto_square_init();
API void tanto_tan_init();
API void tanto_tanh_init();

API void tanto_unpack_binary_init(uint32 icb0, uint32 icb1);
API void tanto_unpack_bcast_rows_init(uint32 icb0, uint32 icb1);
API void tanto_unpack_bcast_cols_init(uint32 icb0, uint32 icb1);
API void tanto_unpack_bcast_scalar_init(uint32 icb0, uint32 icb1);
API void tanto_unpack_matmul_init(uint32 icb0, uint32 icb1, bool transpose);
API void tanto_unpack_unary_init(uint32 icb);
API void tanto_unpack_reduce_rows_init(uint32 icb, uint32 icb_scaler);
API void tanto_unpack_reduce_cols_init(uint32 icb, uint32 icb_scaler);
API void tanto_unpack_reduce_scalar_init(uint32 icb, uint32 icb_scaler);
API void tanto_unpack_tilize_block_init(uint32 icb, uint32 block);
API void tanto_unpack_transpose_init(uint32 icb);
API void tanto_unpack_untilize_block_init(uint32 icb);

API void tanto_pack_init(uint32 ocb);
API void tanto_pack_row_init(uint32 ocb);
API void tanto_pack_col_init(uint32 ocb);
API void tanto_pack_scalar_init(uint32 ocb);

