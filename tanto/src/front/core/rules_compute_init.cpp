// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>

#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

#include "core/matchers.hpp"
#include "core/rules.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

namespace {

RewriteRule _make_math_init_rule(const std::string &func, const std::string &api) {
    // <func>();
    //     =>
    // <api>();
    return makeRule(
        make_func_call_0_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "();")));
}

RewriteRule _make_math_init_param_rule(const std::string &func, const std::string &api) {
    // <func>(param);
    //     =>
    // <api>(param);
    return makeRule(
        make_func_call_1_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "(", node("arg0"), ");")));
}

RewriteRule _make_unpack_unary_init_rule(const std::string &func, const std::string &api) {
    // <func>(icb);
    //     =>
    // <api>(icb.cb_id);
    return makeRule(
        make_func_call_1_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "(", access("arg0", "cb_id"), ");")));
}

RewriteRule _make_unpack_unary_init_param_rule(const std::string &func, const std::string &api) {
    // <func>(icb, param);
    //     =>
    // <api>(icb.cb_id, param);
    return makeRule(
        make_func_call_2_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "(", access("arg0", "cb_id"), ",", node("arg1"), ");")));
}

RewriteRule _make_unpack_binary_init_rule(const std::string &func, const std::string &api) {
    // <func>(icb0, icb1);
    //     =>
    // <api>(icb0.cb_id, icb1.cb_id);
    return makeRule(
        make_func_call_2_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "(", access("arg0", "cb_id"), ", ", access("arg1", "cb_id"), ");")));
}

RewriteRule _make_unpack_binary_init_param_rule(const std::string &func, const std::string &api) {
    // <func>(icb0, icb1, param);
    //     =>
    // <api>(icb0.cb_id, icb1.cb_id, param);
    return makeRule(
        make_func_call_3_matcher(func),
        changeTo(
            statement("stmt"),
            cat(
                api, "(", 
                    access("arg0", "cb_id"), ", ", 
                    access("arg1", "cb_id"), ", ",
                    node("arg2"), ");")));
}

RewriteRule _make_pack_init_rule(const std::string &func, const std::string &api) {
    // <func>(ocb);
    //     =>
    // <api>(ocb.cb_id);
    return makeRule(
        make_func_call_1_matcher(func),
        changeTo(
            statement("stmt"),
            cat(api, "(", access("arg0", "cb_id"), ");")));
}

} // namespace

//
//    RuleFactory
//

RewriteRule RuleFactory::make_copy_init_rule() {
    // __copy_init();
    //     =>
    // tanto_copy_init();
    return _make_math_init_rule("__copy_init", "tanto_copy_init");
}

RewriteRule RuleFactory::make_add_init_rule() {
    // __add_init();
    //     =>
    // tanto_add_init();
    return _make_math_init_rule("__add_init", "tanto_add_init");
}

RewriteRule RuleFactory::make_sub_init_rule() {
    // __sub_init();
    //     =>
    // tanto_sub_init();
    return _make_math_init_rule("__sub_init", "tanto_sub_init");
}

RewriteRule RuleFactory::make_mul_init_rule() {
    // __mul_init();
    //     =>
    // tanto_mul_init();
    return _make_math_init_rule("__mul_init", "tanto_mul_init");
}

RewriteRule RuleFactory::make_add_bcast_rows_init_rule() {
    // __add_bcast_rows_init();
    //     =>
    // tanto_add_bcast_rows_init();
    return _make_math_init_rule("__add_bcast_rows_init", "tanto_add_bcast_rows_init");
}

RewriteRule RuleFactory::make_sub_bcast_rows_init_rule() {
    // __sub_bcast_rows_init();
    //     =>
    // tanto_sub_bcast_rows_init();
    return _make_math_init_rule("__sub_bcast_rows_init", "tanto_sub_bcast_rows_init");
}

RewriteRule RuleFactory::make_mul_bcast_rows_init_rule() {
    // __mul_bcast_rows_init();
    //     =>
    // tanto_mul_bcast_rows_init();
    return _make_math_init_rule("__mul_bcast_rows_init", "tanto_mul_bcast_rows_init");
}

RewriteRule RuleFactory::make_add_bcast_cols_init_rule() {
    // __add_bcast_cols_init();
    //     =>
    // tanto_add_bcast_cols_init();
    return _make_math_init_rule("__add_bcast_cols_init", "tanto_add_bcast_cols_init");
}

RewriteRule RuleFactory::make_sub_bcast_cols_init_rule() {
    // __sub_bcast_cols_init_init();
    //     =>
    // tanto_sub_bcast_cols_init_init();
    return _make_math_init_rule("__sub_bcast_cols_init", "tanto_sub_bcast_cols_init");
}

RewriteRule RuleFactory::make_mul_bcast_cols_init_rule() {
    // __mul_bcast_cols_init();
    //     =>
    // tanto_mul_bcast_cols_init();
    return _make_math_init_rule("__mul_bcast_cols_init", "tanto_mul_bcast_cols_init");
}

RewriteRule RuleFactory::make_add_bcast_scalar_init_rule() {
    // __add_bcast_scalar_init();
    //     =>
    // tanto_add_bcast_scalar_init();
    return _make_math_init_rule("__add_bcast_scalar_init", "tanto_add_bcast_scalar_init");
}

RewriteRule RuleFactory::make_sub_bcast_scalar_init_rule() {
    // __sub_bcast_scalar_init();
    //     =>
    // tanto_sub_bcast_scalar_init();
    return _make_math_init_rule("__sub_bcast_scalar_init", "tanto_sub_bcast_scalar_init");
}

RewriteRule RuleFactory::make_mul_bcast_scalar_init_rule() {
    // __mul_bcast_scalar_init();
    //     =>
    // tanto_mul_bcast_scalar_init();
    return _make_math_init_rule("__mul_bcast_scalar_init", "tanto_mul_bcast_scalar_init");
}

RewriteRule RuleFactory::make_matmul_init_rule() {
    // __matmul_init(transpose);
    //     =>
    // tanto_matmul_init(transpose);
    return _make_math_init_param_rule("__matmul_init", "tanto_matmul_init");
}

RewriteRule RuleFactory::make_reduce_max_rows_init_rule() {
    // __reduce_max_rows_init();
    //     =>
    // tanto_reduce_max_rows_init();
    return _make_math_init_rule("__reduce_max_rows_init", "tanto_reduce_max_rows_init");
}

RewriteRule RuleFactory::make_reduce_max_cols_init_rule() {
    // __reduce_max_cols_init();
    //     =>
    // tanto_reduce_max_cols_init();
    return _make_math_init_rule("__reduce_max_cols_init", "tanto_reduce_max_cols_init");
}

RewriteRule RuleFactory::make_reduce_max_scalar_init_rule() {
    // __reduce_max_scalar_init();
    //     =>
    // tanto_reduce_max_scalar_init();
    return _make_math_init_rule("__reduce_max_scalar_init", "tanto_reduce_max_scalar_init");
}

RewriteRule RuleFactory::make_reduce_sum_rows_init_rule() {
    // __reduce_sum_rows_init();
    //     =>
    // tanto_reduce_sum_rows_init();
    return _make_math_init_rule("__reduce_sum_rows_init", "tanto_reduce_sum_rows_init");
}

RewriteRule RuleFactory::make_reduce_sum_cols_init_rule() {
    // __reduce_sum_cols_init();
    //     =>
    // tanto_reduce_sum_cols_init();
    return _make_math_init_rule("__reduce_sum_cols_init", "tanto_reduce_sum_cols_init");
}

RewriteRule RuleFactory::make_reduce_sum_scalar_init_rule() {
    // __reduce_sum_scalar_init();
    //     =>
    // tanto_reduce_sum_scalar_init();
    return _make_math_init_rule("__reduce_sum_scalar_init", "tanto_reduce_sum_scalar_init");
}

RewriteRule RuleFactory::make_transpose_init_rule() {
    // __transpose_init();
    //     =>
    // tanto_transpose_init();
    return _make_math_init_rule("__transpose_init", "tanto_transpose_init");
}

RewriteRule RuleFactory::make_tilize_block_init_rule() {
    // __tilize_block_init();
    //     =>
    // tanto_tilize_block_init();
    return _make_math_init_rule("__tilize_block_init", "tanto_tilize_block_init");
}

RewriteRule RuleFactory::make_untilize_block_init_rule() {
    // __untilize_block_init();
    //     =>
    // tanto_untilize_block_init();
    return _make_math_init_rule("__untilize_block_init", "tanto_untilize_block_init");
}

RewriteRule RuleFactory::make_copy_dst_init_rule() {
    // __copy_dst_init();
    //     =>
    // tanto_copy_dst_init();
    return _make_math_init_rule("__copy_dst_init", "tanto_copy_dst_init");
}

RewriteRule RuleFactory::make_add_dst_init_rule() {
    // __add_dst_init();
    //     =>
    // tanto_add_dst_init();
    return _make_math_init_rule("__add_dst_init", "tanto_add_dst_init");
}

RewriteRule RuleFactory::make_sub_dst_init_rule() {
    // __sub_dst_init();
    //     =>
    // tanto_sub_dst_init();
    return _make_math_init_rule("__sub_dst_init", "tanto_sub_dst_init");
}

RewriteRule RuleFactory::make_rsub_dst_init_rule() {
    // __rsub_dst_init();
    //     =>
    // tanto_rsub_dst_init();
    return _make_math_init_rule("__rsub_dst_init", "tanto_rsub_dst_init");
}

RewriteRule RuleFactory::make_mul_dst_init_rule() {
    // __mul_dst_init();
    //     =>
    // tanto_mul_dst_init();
    return _make_math_init_rule("__mul_dst_init", "tanto_mul_dst_init");
}

RewriteRule RuleFactory::make_div_dst_init_rule() {
    // __div_dst_init();
    //     =>
    // tanto_div_dst_init();
    return _make_math_init_rule("__div_dst_init", "tanto_div_dst_init");
}

RewriteRule RuleFactory::make_power_dst_init_rule() {
    // __power_dst_init();
    //     =>
    // tanto_power_dst_init();
    return _make_math_init_rule("__power_dst_init", "tanto_power_dst_init");
}

RewriteRule RuleFactory::make_abs_init_rule() {
    // __abs_init();
    //     =>
    // tanto_abs_init();
    return _make_math_init_rule("__abs_init", "tanto_abs_init");
}

RewriteRule RuleFactory::make_acos_init_rule() {
    // __acos_init();
    //     =>
    // tanto_acos_init();
    return _make_math_init_rule("__acos_init", "tanto_acos_init");
}

RewriteRule RuleFactory::make_asin_init_rule() {
    // __asin_init();
    //     =>
    // tanto_asin_init();
    return _make_math_init_rule("__asin_init", "tanto_asin_init");
}

RewriteRule RuleFactory::make_atan_init_rule() {
    // __atan_init();
    //     =>
    // tanto_atan_init();
    return _make_math_init_rule("__atan_init", "tanto_atan_init");
}

RewriteRule RuleFactory::make_binary_scalar_init_rule() {
    // __binary_scalar_init();
    //     =>
    // binary_scalar_atan_init();
    return _make_math_init_rule("__binary_scalar_init", "tanto_binary_scalar_init");
}

RewriteRule RuleFactory::make_cast_init_rule() {
    // __cast_init();
    //     =>
    // tanto_cast_init();
    return _make_math_init_rule("__cast_init", "tanto_cast_init");
}

RewriteRule RuleFactory::make_ceil_init_rule() {
    // __ceil_init();
    //     =>
    // tanto_ceil_init();
    return _make_math_init_rule("__ceil_init", "tanto_ceil_init");
}

RewriteRule RuleFactory::make_cos_init_rule() {
    // __cos_init();
    //     =>
    // tanto_cos_init();
    return _make_math_init_rule("__cos_init", "tanto_cos_init");
}

RewriteRule RuleFactory::make_elu_init_rule() {
    // __elu_init();
    //     =>
    // tanto_elu_init();
    return _make_math_init_rule("__elu_init", "tanto_elu_init");
}

RewriteRule RuleFactory::make_eqz_init_rule() {
    // __eqz_init();
    //     =>
    // tanto_eqz_init();
    return _make_math_init_rule("__eqz_init", "tanto_eqz_init");
}

RewriteRule RuleFactory::make_erf_init_rule() {
    // __erf_init();
    //     =>
    // tanto_erf_init();
    return _make_math_init_rule("__erf_init", "tanto_erf_init");
}

RewriteRule RuleFactory::make_erfc_init_rule() {
    // __erfc_init();
    //     =>
    // tanto_erfc_init();
    return _make_math_init_rule("__erfc_init", "tanto_erfc_init");
}

RewriteRule RuleFactory::make_erfinv_init_rule() {
    // __erfinv_init();
    //     =>
    // tanto_erfinv_init();
    return _make_math_init_rule("__erfinv_init", "tanto_erfinv_init");
}

RewriteRule RuleFactory::make_exp_init_rule() {
    // __exp_init();
    //     =>
    // tanto_exp_init();
    return _make_math_init_rule("__exp_init", "tanto_exp_init");
}

RewriteRule RuleFactory::make_exp2_init_rule() {
    // __exp2_init();
    //     =>
    // tanto_exp2_init();
    return _make_math_init_rule("__exp2_init", "tanto_exp2_init");
}

RewriteRule RuleFactory::make_expm1_init_rule() {
    // __expm1_init();
    //     =>
    // tanto_expm1_init();
    return _make_math_init_rule("__expm1_init", "tanto_expm1_init");
}

RewriteRule RuleFactory::make_fill_init_rule() {
    // __fill_init();
    //     =>
    // tanto_fill_init();
    return _make_math_init_rule("__fill_init", "tanto_fill_init");
}

RewriteRule RuleFactory::make_floor_init_rule() {
    // __floor_init();
    //     =>
    // tanto_floor_init();
    return _make_math_init_rule("__floor_init", "tanto_floor_init");
}

RewriteRule RuleFactory::make_gelu_init_rule() {
    // __gelu_init();
    //     =>
    // tanto_gelu_init();
    return _make_math_init_rule("__gelu_init", "tanto_gelu_init");
}

RewriteRule RuleFactory::make_gez_init_rule() {
    // __gez_init();
    //     =>
    // tanto_gez_init();
    return _make_math_init_rule("__gez_init", "tanto_gez_init");
}

RewriteRule RuleFactory::make_gtz_init_rule() {
    // __gtz_init();
    //     =>
    // tanto_gtz_init();
    return _make_math_init_rule("__gtz_init", "tanto_gtz_init");
}

RewriteRule RuleFactory::make_heaviside_init_rule() {
    // __heaviside_init();
    //     =>
    // tanto_heaviside_init();
    return _make_math_init_rule("__heaviside_init", "tanto_heaviside_init");
}

RewriteRule RuleFactory::make_i0_init_rule() {
    // __i0_init();
    //     =>
    // tanto_i0_init();
    return _make_math_init_rule("__i0_init", "tanto_i0_init");
}

RewriteRule RuleFactory::make_isfinite_init_rule() {
    // __isfinite_init();
    //     =>
    // tanto_isfinite_init();
    return _make_math_init_rule("__isfinite_init", "tanto_isfinite_init");
}

RewriteRule RuleFactory::make_isinf_init_rule() {
    // __isinf_init();
    //     =>
    // tanto_isinf_init();
    return _make_math_init_rule("__isinf_init", "tanto_isinf_init");
}

RewriteRule RuleFactory::make_isnan_init_rule() {
    // __isnan_init();
    //     =>
    // tanto_isnan_init();
    return _make_math_init_rule("__isnan_init", "tanto_isnan_init");
}

RewriteRule RuleFactory::make_isneginf_init_rule() {
    // __isneginf_init();
    //     =>
    // tanto_isneginf_init();
    return _make_math_init_rule("__isneginf_init", "tanto_isneginf_init");
}

RewriteRule RuleFactory::make_isposinf_init_rule() {
    // __isposinf_init();
    //     =>
    // tanto_isposinf_init();
    return _make_math_init_rule("__isposinf_init", "tanto_isposinf_init");
}

RewriteRule RuleFactory::make_leaky_relu_init_rule() {
    // __leaky_relu_init();
    //     =>
    // tanto_leaky_relu_init();
    return _make_math_init_rule("__leaky_relu_init", "tanto_leaky_relu_init");
}

RewriteRule RuleFactory::make_lez_init_rule() {
    // __lez_init();
    //     =>
    // tanto_lez_init();
    return _make_math_init_rule("__lez_init", "tanto_lez_init");
}

RewriteRule RuleFactory::make_log_init_rule() {
    // __log_init();
    //     =>
    // tanto_log_init();
    return _make_math_init_rule("__log_init", "tanto_log_init");
}

RewriteRule RuleFactory::make_log_with_base_init_rule() {
    // __log_with_base_init();
    //     =>
    // tanto_log_with_base_init();
    return _make_math_init_rule("__log_with_base_init", "tanto_log_with_base_init");
}

RewriteRule RuleFactory::make_logical_not_init_rule() {
    // __logical_not_init();
    //     =>
    // tanto_logical_not_init();
    return _make_math_init_rule("__logical_not_init", "tanto_logical_not_init");
}

RewriteRule RuleFactory::make_ltz_init_rule() {
    // __ltz_init();
    //     =>
    // tanto_ltz_init();
    return _make_math_init_rule("__ltz_init", "tanto_ltz_init");
}

RewriteRule RuleFactory::make_max_init_rule() {
    // __max_init();
    //     =>
    // tanto_max_init();
    return _make_math_init_rule("__max_init", "tanto_max_init");
}

RewriteRule RuleFactory::make_nez_init_rule() {
    // __nez_init();
    //     =>
    // tanto_nez_init();
    return _make_math_init_rule("__nez_init", "tanto_nez_init");
}

RewriteRule RuleFactory::make_power_init_rule() {
    // __power_init();
    //     =>
    // tanto_power_init();
    return _make_math_init_rule("__power_init", "tanto_power_init");
}

RewriteRule RuleFactory::make_recip_init_rule() {
    // __recip_init();
    //     =>
    // tanto_recip_init();
    return _make_math_init_rule("__recip_init", "tanto_recip_init");
}

RewriteRule RuleFactory::make_relu_init_rule() {
    // __relu_init();
    //     =>
    // tanto_relu_init();
    return _make_math_init_rule("__relu_init", "tanto_relu_init");
}

RewriteRule RuleFactory::make_relu_max_init_rule() {
    // __relu_max_init();
    //     =>
    // tanto_relu_max_init();
    return _make_math_init_rule("__relu_max_init", "tanto_relu_max_init");
}

RewriteRule RuleFactory::make_relu_min_init_rule() {
    // __relu_min_init();
    //     =>
    // tanto_relu_min_init();
    return _make_math_init_rule("__relu_min_init", "tanto_relu_min_init");
}

RewriteRule RuleFactory::make_rsqrt_init_rule() {
    // __rsqrt_init();
    //     =>
    // tanto_rsqrt_init();
    return _make_math_init_rule("__rsqrt_init", "tanto_rsqrt_init");
}

RewriteRule RuleFactory::make_sigmoid_init_rule() {
    // __sigmoid_init();
    //     =>
    // tanto_sigmoid_init();
    return _make_math_init_rule("__sigmoid_init", "tanto_sigmoid_init");
}

RewriteRule RuleFactory::make_sign_init_rule() {
    // __sign_init();
    //     =>
    // tanto_sign_init();
    return _make_math_init_rule("__sign_init", "tanto_sign_init");
}

RewriteRule RuleFactory::make_signbit_init_rule() {
    // __signbit_init();
    //     =>
    // tanto_signbit_init();
    return _make_math_init_rule("__signbit_init", "tanto_signbit_init");
}

RewriteRule RuleFactory::make_sin_init_rule() {
    // __sin_init();
    //     =>
    // tanto_sin_init();
    return _make_math_init_rule("__sin_init", "tanto_sin_init");
}

RewriteRule RuleFactory::make_sqrt_init_rule() {
    // __sqrt_init();
    //     =>
    // tanto_sqrt_init();
    return _make_math_init_rule("__sqrt_init", "tanto_sqrt_init");
}

RewriteRule RuleFactory::make_square_init_rule() {
    // __square_init();
    //     =>
    // tanto_square_init();
    return _make_math_init_rule("__square_init", "tanto_square_init");
}

RewriteRule RuleFactory::make_tan_init_rule() {
    // __tan_init();
    //     =>
    // tanto_tan_init();
    return _make_math_init_rule("__tan_init", "tanto_tan_init");
}

RewriteRule RuleFactory::make_tanh_init_rule() {
    // __tanh_init();
    //     =>
    // tanto_tanh_init();
    return _make_math_init_rule("__tanh_init", "tanto_tanh_init");
}

RewriteRule RuleFactory::make_unpack_binary_init_rule() {
    // __unpack_binary_init(icb0, icb1);
    //     =>
    // tanto_unpack_binary_init(icb0.cb_id, icb1.cb_id);
    return _make_unpack_binary_init_rule("__unpack_binary_init", "tanto_unpack_binary_init");
}

RewriteRule RuleFactory::make_unpack_bcast_rows_init_rule() {
    // __unpack_bcast_rows_init(icb0, icb1);
    //     =>
    // tanto_unpack_bcast_rows_init(icb0.cb_id, icb1.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_bcast_rows_init", "tanto_unpack_bcast_rows_init");
}

RewriteRule RuleFactory::make_unpack_bcast_cols_init_rule() {
    // __unpack_bcast_cols_init(icb0, icb1);
    //     =>
    // tanto_unpack_bcast_cols_init(icb0.cb_id, icb1.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_bcast_cols_init", "tanto_unpack_bcast_cols_init");
}

RewriteRule RuleFactory::make_unpack_bcast_scalar_init_rule() {
    // __unpack_bcast_scalar_init(icb0, icb1);
    //     =>
    // tanto_unpack_bcast_scalar_init(icb0.cb_id, icb1.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_bcast_scalar_init", "tanto_unpack_bcast_scalar_init");
}

RewriteRule RuleFactory::make_unpack_matmul_init_rule() {
    // __unpack_matmul_init(icb0, icb1);
    //     =>
    // tanto_unpack_matmul_init(icb0.cb_id, icb1.cb_id);
    return _make_unpack_binary_init_param_rule(
        "__unpack_matmul_init", "tanto_unpack_matmul_init");
}

RewriteRule RuleFactory::make_unpack_unary_init_rule() {
    // __unpack_unary_init(icb);
    //     =>
    // tanto_unpack_unary_init(icb.cb_id);
    return _make_unpack_unary_init_rule("__unpack_unary_init", "tanto_unpack_unary_init");
}

RewriteRule RuleFactory::make_unpack_reduce_rows_init_rule() {
    // __unpack_reduce_rows_init(icb, icb_scaler);
    //     =>
    // tanto_unpack_reduce_rows_init(icb.cb_id, icb_scaler.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_reduce_rows_init", "tanto_unpack_reduce_rows_init");
}

RewriteRule RuleFactory::make_unpack_reduce_cols_init_rule() {
    // __unpack_reduce_cols_init(icb, icb_scaler);
    //     =>
    // tanto_unpack_reduce_cols_init(icb.cb_id, icb_scaler.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_reduce_cols_init", "tanto_unpack_reduce_cols_init");
}

RewriteRule RuleFactory::make_unpack_reduce_scalar_init_rule() {
    // __unpack_reduce_scalar_init(icb, icb_scaler);
    //     =>
    // tanto_unpack_reduce_scalar_init(icb.cb_id, icb_scaler.cb_id);
    return _make_unpack_binary_init_rule(
        "__unpack_reduce_scalar_init", "tanto_unpack_reduce_scalar_init");
}

RewriteRule RuleFactory::make_unpack_tilize_block_init_rule() {
    // __unpack_tilize_block_init(icb, block);
    //     =>
    // tanto_unpack_tilize_block_init(icb.cb_id, block);
    return _make_unpack_unary_init_param_rule(
        "__unpack_tilize_block_init", "tanto_unpack_tilize_block_init");
}

RewriteRule RuleFactory::make_unpack_transpose_init_rule() {
    // __unpack_transpose_init(icb);
    //     =>
    // tanto_unpack_transpose_init(icb.cb_id);
    return _make_unpack_unary_init_rule(
        "__unpack_transpose_init", "tanto_unpack_transpose_init");
}

RewriteRule RuleFactory::make_unpack_untilize_block_init_rule() {
    // __unpack_untilize_block_init(icb);
    //     =>
    // tanto_unpack_untilize_block_init(icb.cb_id);
    return _make_unpack_unary_init_rule(
        "__unpack_untilize_block_init", "tanto_unpack_untilize_block_init");
}

RewriteRule RuleFactory::make_pack_init_rule() {
    // __pack_init(ocb);
    //     =>
    // tanto_pack_init(ocb.cb_id);
    return _make_pack_init_rule("__pack_init", "tanto_pack_init");
}

RewriteRule RuleFactory::make_pack_row_init_rule() {
    // __pack_row_init(ocb);
    //     =>
    // tanto_pack_row_init(ocb.cb_id);
    return _make_pack_init_rule("__pack_row_init", "tanto_pack_row_init");
}

RewriteRule RuleFactory::make_pack_col_init_rule() {
    // __pack_col_init(ocb);
    //     =>
    // tanto_pack_col_init(ocb.cb_id);
    return _make_pack_init_rule("__pack_col_init", "tanto_pack_col_init");
}

RewriteRule RuleFactory::make_pack_scalar_init_rule() {
    // __pack_scalar_init(ocb);
    //     =>
    // tanto_pack_scalar_init(ocb.cb_id);
    return _make_pack_init_rule("__pack_scalar_init", "tanto_pack_scalar_init");
}

} // namespace front
} // namespace tanto
} // namespace ronin

