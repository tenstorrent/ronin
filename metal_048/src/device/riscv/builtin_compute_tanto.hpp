#pragma once

#include <string>
#include <unordered_map>
#include <utility>

//
//    Generic list of compute Tanto builtins
//

#define COMPUTE_TANTO_BUILTINS \
    DECL_BUILTIN(tanto_copy_init, 0) \
    DECL_BUILTIN(tanto_add_init, 0) \
    DECL_BUILTIN(tanto_sub_init, 0) \
    DECL_BUILTIN(tanto_mul_init, 0) \
    DECL_BUILTIN(tanto_add_bcast_rows_init, 0) \
    DECL_BUILTIN(tanto_sub_bcast_rows_init, 0) \
    DECL_BUILTIN(tanto_mul_bcast_rows_init, 0) \
    DECL_BUILTIN(tanto_add_bcast_cols_init, 0) \
    DECL_BUILTIN(tanto_sub_bcast_cols_init, 0) \
    DECL_BUILTIN(tanto_mul_bcast_cols_init, 0) \
    DECL_BUILTIN(tanto_add_bcast_scalar_init, 0) \
    DECL_BUILTIN(tanto_sub_bcast_scalar_init, 0) \
    DECL_BUILTIN(tanto_mul_bcast_scalar_init, 0) \
    DECL_BUILTIN(tanto_matmul_init, 1) \
    DECL_BUILTIN(tanto_reduce_max_rows_init, 0) \
    DECL_BUILTIN(tanto_reduce_max_cols_init, 0) \
    DECL_BUILTIN(tanto_reduce_max_scalar_init, 0) \
    DECL_BUILTIN(tanto_reduce_sum_rows_init, 0) \
    DECL_BUILTIN(tanto_reduce_sum_cols_init, 0) \
    DECL_BUILTIN(tanto_reduce_sum_scalar_init, 0) \
    DECL_BUILTIN(tanto_transpose_init, 0) \
    DECL_BUILTIN(tanto_tilize_block_init, 0) \
    DECL_BUILTIN(tanto_untilize_block_init, 0) \
    DECL_BUILTIN(tanto_abs_init, 0) \
    DECL_BUILTIN(tanto_acos_init, 0) \
    DECL_BUILTIN(tanto_asin_init, 0) \
    DECL_BUILTIN(tanto_atan_init, 0) \
    DECL_BUILTIN(tanto_cos_init, 0) \
    DECL_BUILTIN(tanto_elu_init, 0) \
    DECL_BUILTIN(tanto_eqz_init, 0) \
    DECL_BUILTIN(tanto_erf_init, 0) \
    DECL_BUILTIN(tanto_erfc_init, 0) \
    DECL_BUILTIN(tanto_erfinv_init, 0) \
    DECL_BUILTIN(tanto_exp_init, 0) \
    DECL_BUILTIN(tanto_exp2_init, 0) \
    DECL_BUILTIN(tanto_expm1_init, 0) \
    DECL_BUILTIN(tanto_gelu_init, 0) \
    DECL_BUILTIN(tanto_gez_init, 0) \
    DECL_BUILTIN(tanto_gtz_init, 0) \
    DECL_BUILTIN(tanto_heaviside_init, 0) \
    DECL_BUILTIN(tanto_i0_init, 0) \
    DECL_BUILTIN(tanto_isfinite_init, 0) \
    DECL_BUILTIN(tanto_isinf_init, 0) \
    DECL_BUILTIN(tanto_isnan_init, 0) \
    DECL_BUILTIN(tanto_isneginf_init, 0) \
    DECL_BUILTIN(tanto_isposinf_init, 0) \
    DECL_BUILTIN(tanto_leaky_relu_init, 0) \
    DECL_BUILTIN(tanto_lez_init, 0) \
    DECL_BUILTIN(tanto_log_init, 0) \
    DECL_BUILTIN(tanto_log_with_base_init, 0) \
    DECL_BUILTIN(tanto_logical_not_init, 0) \
    DECL_BUILTIN(tanto_ltz_init, 0) \
    DECL_BUILTIN(tanto_nez_init, 0) \
    DECL_BUILTIN(tanto_power_init, 0) \
    DECL_BUILTIN(tanto_recip_init, 0) \
    DECL_BUILTIN(tanto_relu_init, 0) \
    DECL_BUILTIN(tanto_relu_max_init, 0) \
    DECL_BUILTIN(tanto_relu_min_init, 0) \
    DECL_BUILTIN(tanto_rsqrt_init, 0) \
    DECL_BUILTIN(tanto_sigmoid_init, 0) \
    DECL_BUILTIN(tanto_sign_init, 0) \
    DECL_BUILTIN(tanto_signbit_init, 0) \
    DECL_BUILTIN(tanto_sin_init, 0) \
    DECL_BUILTIN(tanto_sqrt_init, 0) \
    DECL_BUILTIN(tanto_square_init, 0) \
    DECL_BUILTIN(tanto_tan_init, 0) \
    DECL_BUILTIN(tanto_tanh_init, 0) \
    DECL_BUILTIN(tanto_unpack_binary_init, 2) \
    DECL_BUILTIN(tanto_unpack_bcast_rows_init, 2) \
    DECL_BUILTIN(tanto_unpack_bcast_cols_init, 2) \
    DECL_BUILTIN(tanto_unpack_bcast_scalar_init, 2) \
    DECL_BUILTIN(tanto_unpack_matmul_init, 3) \
    DECL_BUILTIN(tanto_unpack_unary_init, 1) \
    DECL_BUILTIN(tanto_unpack_tilize_block_init, 2) \
    DECL_BUILTIN(tanto_unpack_transpose_init, 1) \
    DECL_BUILTIN(tanto_unpack_untilize_block_init, 1) \
    DECL_BUILTIN(tanto_pack_init, 1) \
    DECL_BUILTIN(tanto_pack_reduce_rows_init, 1) \
    DECL_BUILTIN(tanto_pack_reduce_cols_init, 1) \
    DECL_BUILTIN(tanto_pack_reduce_scalar_init, 1)

//
//    Compute builtin enumeration
//

#define DECL_BUILTIN(name, count) name,

enum class ComputeTantoBuiltinId {
    START = 3072,
COMPUTE_TANTO_BUILTINS
};

#undef DECL_BUILTIN

// public functions

std::unordered_map<ComputeTantoBuiltinId, std::pair<std::string, int>> &
    get_compute_tanto_builtin_map();

