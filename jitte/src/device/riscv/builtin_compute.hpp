// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

//
//    Generic list of compute builtins
//

#define COMPUTE_BUILTINS \
    DECL_BUILTIN(get_arg_uint32, 1) \
    DECL_BUILTIN(acquire_dst, 1) \
    DECL_BUILTIN(tile_regs_acquire, 0) \
    DECL_BUILTIN(tile_regs_wait, 0) \
    DECL_BUILTIN(release_dst, 1) \
    DECL_BUILTIN(tile_regs_commit, 0) \
    DECL_BUILTIN(tile_regs_release, 0) \
    DECL_BUILTIN(pack_relu_config, 1) \
    DECL_BUILTIN(pack_tile, 4) \
    DECL_BUILTIN(matmul_pack_tile, 3) \
    DECL_BUILTIN(pack_reconfig_data_format, 1) \
    DECL_BUILTIN(unpack_reconfig_data_format, 2) \
    DECL_BUILTIN(cb_wait_front, 2) \
    DECL_BUILTIN(cb_pop_front, 2) \
    DECL_BUILTIN(cb_reserve_back, 2) \
    DECL_BUILTIN(cb_push_back, 2) \
    DECL_BUILTIN(get_write_ptr, 1) \
    DECL_BUILTIN(get_read_ptr, 1) \
    DECL_BUILTIN(set_write_ptr, 2) \
    DECL_BUILTIN(set_read_ptr, 2) \
    DECL_BUILTIN(sub_tiles_bcast_cols, 5) \
    DECL_BUILTIN(mul_tiles_bcast_cols, 5) \
    DECL_BUILTIN(mul_tiles_bcast_rows, 5) \
    DECL_BUILTIN(add_tiles_bcast_rows, 5) \
    DECL_BUILTIN(add_tiles_bcast_cols, 5) \
    DECL_BUILTIN(init_bcast, 5) \
    DECL_BUILTIN(any_tiles_bcast, 7) \
    DECL_BUILTIN(add_tiles_bcast, 6) \
    DECL_BUILTIN(sub_tiles_bcast, 6) \
    DECL_BUILTIN(mul_tiles_bcast, 6) \
    DECL_BUILTIN(add_bcast_rows_init_short, 2) \
    DECL_BUILTIN(add_bcast_cols_init_short, 2) \
    DECL_BUILTIN(mul_tiles_bcast_scalar_init_short, 2) \
    DECL_BUILTIN(mul_tiles_bcast_scalar, 5) \
    DECL_BUILTIN(mul_bcast_cols_init_short, 2) \
    DECL_BUILTIN(mul_bcast_rows_init_short, 2) \
    DECL_BUILTIN(sub_bcast_cols_init_short, 2) \
    DECL_BUILTIN(binary_op_init_common, 3) \
    DECL_BUILTIN(mul_tiles_init_f, 0) \
    DECL_BUILTIN(mul_tiles_init, 2) \
    DECL_BUILTIN(add_tiles_init_nof, 0) \
    DECL_BUILTIN(add_tiles_init, 3) \
    DECL_BUILTIN(sub_tiles_init_nof, 0) \
    DECL_BUILTIN(sub_tiles_init, 3) \
    DECL_BUILTIN(mul_tiles, 5) \
    DECL_BUILTIN(add_tiles, 5) \
    DECL_BUILTIN(sub_tiles, 5) \
    DECL_BUILTIN(binary_op_specific_init, 2) \
    DECL_BUILTIN(unary_op_init_common, 1) \
    DECL_BUILTIN(init_sfpu, 1) \
    DECL_BUILTIN(mm_init, 4) \
    DECL_BUILTIN(matmul_tiles, 6) \
    DECL_BUILTIN(mm_init_short, 3) \
    DECL_BUILTIN(mm_init_short_with_dt, 4) \
    DECL_BUILTIN(mm_block_init, 7) \
    DECL_BUILTIN(matmul_block, 9) \
    DECL_BUILTIN(mm_block_init_short, 6) \
    DECL_BUILTIN(mm_block_init_short_with_dt, 7) \
    DECL_BUILTIN(reduce_init, 6) \
    DECL_BUILTIN(reduce_init_short, 5) \
    DECL_BUILTIN(reduce_init_delta, 6) \
    DECL_BUILTIN(reduce_revert_delta, 2) \
    DECL_BUILTIN(reduce_tile, 7) \
    DECL_BUILTIN(copy_tile_to_dst_init_short, 2) \
    DECL_BUILTIN(copy_tile_init, 0) \
    DECL_BUILTIN(copy_tile_to_dst_init_short_with_dt, 3) \
    DECL_BUILTIN(copy_tile, 3) \
    DECL_BUILTIN(copy_block_matmul_partials, 4) \
    DECL_BUILTIN(tilize_init, 3) \
    DECL_BUILTIN(tilize_init_short, 2) \
    DECL_BUILTIN(tilize_block, 3) \
    DECL_BUILTIN(tilize_uninit, 1) \
    DECL_BUILTIN(transpose_wh_init, 2) \
    DECL_BUILTIN(transpose_wh_tile, 3) \
    DECL_BUILTIN(untilize_init, 2) \
    DECL_BUILTIN(untilize_init_short, 1) \
    DECL_BUILTIN(untilize_block, 4) \
    DECL_BUILTIN(untilize_uninit, 1) \
    DECL_BUILTIN(pack_untilize_init, 3) \
    DECL_BUILTIN(pack_untilize_block, 4) \
    DECL_BUILTIN(pack_untilize_uninit, 1) \
    DECL_BUILTIN(pack_untilize_dst_init_short, 6) \
    DECL_BUILTIN(pack_untilize_dst, 8) \
    DECL_BUILTIN(pack_untilize_init_short, 3) \
    DECL_BUILTIN(rsqrt_tile_init, 0) \
    DECL_BUILTIN(rsqrt_tile, 2) \
    DECL_BUILTIN(sigmoid_tile_init, 0) \
    DECL_BUILTIN(sigmoid_tile, 1) \
    DECL_BUILTIN(log_tile_init, 0) \
    DECL_BUILTIN(log_tile, 1) \
    DECL_BUILTIN(log_with_base_tile_init, 0) \
    DECL_BUILTIN(log_with_base_tile, 2) \
    DECL_BUILTIN(tanh_tile_init, 0) \
    DECL_BUILTIN(tanh_tile, 1) \
    DECL_BUILTIN(signbit_tile_init, 0) \
    DECL_BUILTIN(signbit_tile, 1) \
    DECL_BUILTIN(abs_tile_init, 0) \
    DECL_BUILTIN(abs_tile, 1) \
    DECL_BUILTIN(sign_tile_init, 0) \
    DECL_BUILTIN(sign_tile, 1) \
    DECL_BUILTIN(square_tile_init, 0) \
    DECL_BUILTIN(square_tile, 1) \
    DECL_BUILTIN(ltz_tile_init, 0) \
    DECL_BUILTIN(ltz_tile, 1) \
    DECL_BUILTIN(eqz_tile_init, 0) \
    DECL_BUILTIN(eqz_tile, 1) \
    DECL_BUILTIN(lez_tile_init, 0) \
    DECL_BUILTIN(lez_tile, 1) \
    DECL_BUILTIN(gtz_tile_init, 0) \
    DECL_BUILTIN(gtz_tile, 1) \
    DECL_BUILTIN(nez_tile_init, 0) \
    DECL_BUILTIN(nez_tile, 1) \
    DECL_BUILTIN(gez_tile_init, 0) \
    DECL_BUILTIN(gez_tile, 1) \
    DECL_BUILTIN(power_tile_init, 0) \
    DECL_BUILTIN(power_tile, 2) \
    DECL_BUILTIN(max_tile_init, 0) \
    DECL_BUILTIN(max_tile, 2) \
    DECL_BUILTIN(exp2_tile_init, 0) \
    DECL_BUILTIN(exp2_tile, 1) \
    DECL_BUILTIN(heaviside_tile_init, 0) \
    DECL_BUILTIN(heaviside_tile, 2) \
    DECL_BUILTIN(expm1_tile_init, 0) \
    DECL_BUILTIN(expm1_tile, 1) \
    DECL_BUILTIN(asin_tile_init, 0) \
    DECL_BUILTIN(asin_tile, 1) \
    DECL_BUILTIN(atan_tile_init, 0) \
    DECL_BUILTIN(atan_tile, 1) \
    DECL_BUILTIN(acos_tile_init, 0) \
    DECL_BUILTIN(acos_tile, 1) \
    DECL_BUILTIN(binop_with_scalar_tile_init, 0) \
    DECL_BUILTIN(add_unary_tile, 2) \
    DECL_BUILTIN(sub_unary_tile, 2) \
    DECL_BUILTIN(mul_unary_tile, 2) \
    DECL_BUILTIN(div_unary_tile, 2) \
    DECL_BUILTIN(rsub_unary_tile, 2) \
    DECL_BUILTIN(elu_tile_init, 0) \
    DECL_BUILTIN(elu_tile, 2) \
    DECL_BUILTIN(erf_tile_init, 0) \
    DECL_BUILTIN(erf_tile, 2) \
    DECL_BUILTIN(erfc_tile_init, 0) \
    DECL_BUILTIN(erfc_tile, 2) \
    DECL_BUILTIN(erfinv_tile_init, 0) \
    DECL_BUILTIN(erfinv_tile, 1) \
    DECL_BUILTIN(exp_tile_init, 0) \
    DECL_BUILTIN(exp_tile, 1) \
    DECL_BUILTIN(gelu_tile_init, 0) \
    DECL_BUILTIN(gelu_tile, 2) \
    DECL_BUILTIN(i0_tile_init, 0) \
    DECL_BUILTIN(i0_tile, 1) \
    DECL_BUILTIN(isinf_tile_init, 0) \
    DECL_BUILTIN(isinf_tile, 1) \
    DECL_BUILTIN(isposinf_tile_init, 0) \
    DECL_BUILTIN(isposinf_tile, 1) \
    DECL_BUILTIN(isneginf_tile_init, 0) \
    DECL_BUILTIN(isneginf_tile, 1) \
    DECL_BUILTIN(isnan_tile_init, 0) \
    DECL_BUILTIN(isnan_tile, 1) \
    DECL_BUILTIN(isfinite_tile_init, 0) \
    DECL_BUILTIN(isfinite_tile, 1) \
    DECL_BUILTIN(logical_not_unary_tile_init, 0) \
    DECL_BUILTIN(logical_not_unary_tile, 1) \
    DECL_BUILTIN(recip_tile_init, 0) \
    DECL_BUILTIN(recip_tile, 1) \
    DECL_BUILTIN(relu_max_tile_init, 0) \
    DECL_BUILTIN(relu_max_tile, 2) \
    DECL_BUILTIN(relu_min_tile_init, 0) \
    DECL_BUILTIN(relu_min_tile, 2) \
    DECL_BUILTIN(relu_tile_init, 0) \
    DECL_BUILTIN(relu_tile, 1) \
    DECL_BUILTIN(leaky_relu_tile_init, 0) \
    DECL_BUILTIN(leaky_relu_tile, 2) \
    DECL_BUILTIN(sqrt_tile_init, 0) \
    DECL_BUILTIN(sqrt_tile, 1) \
    DECL_BUILTIN(sin_tile_init, 0) \
    DECL_BUILTIN(sin_tile, 1) \
    DECL_BUILTIN(cos_tile_init, 0) \
    DECL_BUILTIN(cos_tile, 1) \
    DECL_BUILTIN(tan_tile_init, 0) \
    DECL_BUILTIN(tan_tile, 1)

//
//    Compute builtin enumeration
//

#define DECL_BUILTIN(name, count) name,

enum class ComputeBuiltinId {
    START = 0,
COMPUTE_BUILTINS
};

#undef DECL_BUILTIN

// public functions

std::unordered_map<ComputeBuiltinId, std::pair<std::string, int>> &get_compute_builtin_map();

