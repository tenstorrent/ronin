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

RewriteRule _make_math_eltwise_binary_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(src0, src1, isrc0, isrc1, idst);
    //    =>
    // <api>(src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return makeRule(
        make_member_call_5_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(
                api, "(", 
                    access("arg0", "cb_id"), ", ", 
                    access("arg1", "cb_id"), ", ",
                    node("arg2"), ", ",
                    node("arg3"), ", ",
                    node("arg4"), ");")));
}

RewriteRule _make_math_bcast_rule(
        const std::string &method,
        const std::string &bcast_op,
        const std::string &bcast_dim) {
    // self.<method>(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::<bcast_op>, BroadcastType::<bcast_dim>>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    std::string templ_args = 
        "EltwiseBinaryType::" + bcast_op + ", " + "BroadcastType::" + bcast_dim;
    return makeRule(
        make_member_call_5_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(
                "any_tiles_bcast<", templ_args, ">(", 
                    access("arg0", "cb_id"), ", ", 
                    access("arg1", "cb_id"), ", ",
                    node("arg2"), ", ",
                    node("arg3"), ", ",
                    node("arg4"), ");")));
}

RewriteRule _make_math_reduce_rule(
        const std::string &method,
        const std::string &pool_type,
        const std::string &reduce_dim) {
    // self.<method>(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::<pool_type>, ReduceDim::<reduce_dim>>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    //         NOTE: Requires non-standard templated versions of "reduce_tile"
    //             (must be added to both emulator and metal packages)
    std::string templ_args = "PoolType::" + pool_type + ", " + "ReduceDim::" + reduce_dim;
    return makeRule(
        make_member_call_5_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(
                "reduce_tile<", templ_args, ">(", 
                    access("arg0", "cb_id"), ", ", 
                    access("arg1", "cb_id"), ", ",
                    node("arg2"), ", ",
                    node("arg3"), ", ",
                    node("arg4"), ");")));
}

RewriteRule _make_math_eltwise_binary_dst_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst0, idst1);
    //    =>
    // <api>(idst0, idst1);
    return makeRule(
        make_member_call_2_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ", ", node("arg1"), ");")));
}

RewriteRule _make_math_eltwise_unary_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst);
    //     =>
    // <api>(idst);
    return makeRule(
        make_member_call_1_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ");")));
}

#if 0 // TODO: Revise this (see comment below)
RewriteRule _make_math_eltwise_unary_approx_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst);
    //     =>
    // <api>(idst, FAST_AND_APPROX);
    //     NOTE: Macro FAST_AND_APPROX must be defined
    return makeRule(
        make_member_call_1_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ", FAST_AND_APPROX);")));
}
#endif

RewriteRule _make_math_eltwise_unary_approx_rule(
        const std::string &method,
        const std::string &api) {
    // current solution uses default "fast_and_approx" parameters
    // therefore just fall back to regular unary rule
    return _make_math_eltwise_unary_rule(method, api);
}

RewriteRule _make_math_eltwise_unary_param_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst, param);
    //     =>
    // <api>(idst, param);
    return makeRule(
        make_member_call_2_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ", ", node("arg1"), ");")));
}

#if 0 // TODO: Revise this
RewriteRule _make_math_eltwise_unary_param_u16b_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst, param);
    //     =>
    // <api>(idst, f32_as_u16b(param));
    return makeRule(
        make_member_call_2_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ", f32_as_u16b(", node("arg1"), "));")));
}

RewriteRule _make_math_eltwise_unary_param_u32_rule(
        const std::string &method,
        const std::string &api) {
    // self.<method>(idst, param);
    //     =>
    // <api>(idst, f32_as_u32(param));
    return makeRule(
        make_member_call_2_matcher("math", method),
        changeTo(
            statement("stmt"), 
            cat(api, "(", node("arg0"), ", f32_as_u32(", node("arg1"), "));")));
}
#endif

} // namespace

//
//    RuleFactory
//

// math

RewriteRule RuleFactory::make_math_decl_rule() {
    // { ... math<T> name; ... }
    //     =>
    // {
    //     tile_regs_acquire();
    //     tile_regs_wait();
    //     ...
    //     tile_regs_commit();
    //     tile_regs_release();
    // }
    return makeRule(
        declStmt(
            hasSingleDecl(varDecl(hasType(cxxRecordDecl(hasName("math"))))),
            hasParent(compoundStmt().bind("parent"))
        ).bind("stmt"),
        {
            changeTo(
                statement("stmt"),
                cat(
                    "tile_regs_acquire();",
                    "tile_regs_wait();")),
            insertAfter(
                statements("parent"),
                cat(
                    "tile_regs_commit();",
                    "tile_regs_release();"))
        });
}

RewriteRule RuleFactory::make_math_pack_rule() {
    // self.pack(isrc, dst);
    //     =>
    // pack_tile(isrc, dst.cb_id);
    return makeRule(
        make_member_call_2_matcher("math", "pack"),
        changeTo(
            statement("stmt"), 
            cat("pack_tile(", node("arg0"), ", ", access("arg1", "cb_id"), ");")));
}

RewriteRule RuleFactory::make_math_pack_row_rule() {
    // self.pack_row(isrc, dst);
    //     =>
    // pack_tile(isrc, dst.cb_id);
    return makeRule(
        make_member_call_2_matcher("math", "pack_row"),
        changeTo(
            statement("stmt"), 
            cat("pack_tile(", node("arg0"), ", ", access("arg1", "cb_id"), ");")));
}

RewriteRule RuleFactory::make_math_pack_col_rule() {
    // self.pack_col(isrc, dst);
    //     =>
    // pack_tile(isrc, dst.cb_id);
    return makeRule(
        make_member_call_2_matcher("math", "pack_col"),
        changeTo(
            statement("stmt"), 
            cat("pack_tile(", node("arg0"), ", ", access("arg1", "cb_id"), ");")));
}

RewriteRule RuleFactory::make_math_pack_scalar_rule() {
    // self.pack_scalar(isrc, dst);
    //     =>
    // pack_tile(isrc, dst.cb_id);
    return makeRule(
        make_member_call_2_matcher("math", "pack_scalar"),
        changeTo(
            statement("stmt"), 
            cat("pack_tile(", node("arg0"), ", ", access("arg1", "cb_id"), ");")));
}

RewriteRule RuleFactory::make_math_copy_rule() {
    // self.copy(src, isrc, idst);
    //     =>
    // copy_tile(src.cb_id, isrc, idst);
    return makeRule(
        make_member_call_3_matcher("math", "copy"),
        changeTo(
            statement("stmt"), 
            cat(
                "copy_tile(", 
                    access("arg0", "cb_id"), ", ",
                    node("arg1"), ", ", 
                    node("arg2"), ");")));
}

RewriteRule RuleFactory::make_math_add_rule() {
    // self.add(src0, src1, isrc0, isrc1, idst);
    //    =>
    // add_tiles(src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_eltwise_binary_rule("add", "add_tiles");
}

RewriteRule RuleFactory::make_math_sub_rule() {
    // self.sub(src0, src1, isrc0, isrc1, idst);
    //    =>
    // sub_tiles(src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_eltwise_binary_rule("sub", "sub_tiles");
}

RewriteRule RuleFactory::make_math_mul_rule() {
    // self.mul(src0, src1, isrc0, isrc1, idst);
    //    =>
    // mul_tiles(src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_eltwise_binary_rule("mul", "mul_tiles");
}

RewriteRule RuleFactory::make_math_add_bcast_rows_rule() {
    // self.add_bcast_rows(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("add_bcast_rows", "ELWADD", "ROW");
}

RewriteRule RuleFactory::make_math_sub_bcast_rows_rule() {
    // self.sub_bcast_rows(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::ROW>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("sub_bcast_rows", "ELWSUB", "ROW");
}

RewriteRule RuleFactory::make_math_mul_bcast_rows_rule() {
    // self.mul_bcast_rows(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("mul_bcast_rows", "ELWMUL", "ROW");
}

RewriteRule RuleFactory::make_math_add_bcast_cols_rule() {
    // self.add_bcast_cols(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("add_bcast_cols", "ELWADD", "COL");
}

RewriteRule RuleFactory::make_math_sub_bcast_cols_rule() {
    // self.sub_bcast_cols(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("sub_bcast_cols", "ELWSUB", "COL");
}

RewriteRule RuleFactory::make_math_mul_bcast_cols_rule() {
    // self.mul_bcast_cols(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("mul_bcast_cols", "ELWMUL", "COL");
}

RewriteRule RuleFactory::make_math_add_bcast_scalar_rule() {
    // self.add_bcast_scalar(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("add_bcast_scalar", "ELWADD", "SCALAR");
}

RewriteRule RuleFactory::make_math_sub_bcast_scalar_rule() {
    // self.sub_bcast_scalar(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("sub_bcast_scalar", "ELWSUB", "SCALAR");
}

RewriteRule RuleFactory::make_math_mul_bcast_scalar_rule() {
    // self.mul_bcast_scalar(src0, src1, isrc0, isrc1, idst);
    //    =>
    // any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_bcast_rule("mul_bcast_scalar", "ELWMUL", "SCALAR");
}

RewriteRule RuleFactory::make_math_matmul_rule() {
    // self.matmul(src0, src1, isrc0, isrc1, idst, transpose);
    //     =>
    // matmul_tiles(src0.cb_id, src1.cb_id, isrc0, isrc1, idst, transpose);
    return makeRule(
        make_member_call_6_matcher("math", "matmul"),
        changeTo(
            statement("stmt"), 
            cat(
                "matmul_tiles(", 
                    access("arg0", "cb_id"), ", ", 
                    access("arg1", "cb_id"), ", ",
                    node("arg2"), ", ", 
                    node("arg3"), ", ", 
                    node("arg4"), ", ", 
                    node("arg5"), ");")));
}

RewriteRule RuleFactory::make_math_reduce_max_rows_rule() {
    // self.reduce_max_rows(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_max_rows", "MAX", "REDUCE_ROW");
}

RewriteRule RuleFactory::make_math_reduce_max_cols_rule() {
    // self.reduce_max_cols(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_max_cols", "MAX", "REDUCE_COL");
}

RewriteRule RuleFactory::make_math_reduce_max_scalar_rule() {
    // self.reduce_max_scalar(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_max_scalar", "MAX", "REDUCE_SCALAR");
}

RewriteRule RuleFactory::make_math_reduce_sum_rows_rule() {
    // self.reduce_sum_rows(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_sum_rows", "SUM", "REDUCE_ROW");
}

RewriteRule RuleFactory::make_math_reduce_sum_cols_rule() {
    // self.reduce_sum_cols(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_sum_cols", "SUM", "REDUCE_COL");
}

RewriteRule RuleFactory::make_math_reduce_sum_scalar_rule() {
    // self.reduce_sum_scalar(src0, src1, isrc0, isrc1, idst);
    //    =>
    // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(
    //     src0.cb_id, src1.cb_id, isrc0, isrc1, idst);
    return _make_math_reduce_rule("reduce_sum_scalar", "SUM", "REDUCE_SCALAR");
}

RewriteRule RuleFactory::make_math_transpose_rule() {
    // transpose(src, isrc, idst);
    //     =>
    // transpose_wh_tile(src.cb_id, isrc, idst);
    return makeRule(
        make_member_call_3_matcher("math", "transpose"),
        changeTo(
            statement("stmt"), 
            cat(
                "transpose_wh_tile(", 
                    access("arg0", "cb_id"), ", ", 
                    node("arg1"), ", ",
                    node("arg2"), ");")));
}

RewriteRule RuleFactory::make_math_copy_dst_rule() {
    // self.copy_dst(idst0, idst1);
    //     =>
    // copy_dest_values(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("copy_dst", "copy_dest_values");
}

RewriteRule RuleFactory::make_math_add_dst_rule() {
    // self.add_dst(idst0, idst1);
    //     =>
    // add_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("add_dst", "add_binary_tile");
}

RewriteRule RuleFactory::make_math_sub_dst_rule() {
    // self.sub_dst(idst0, idst1);
    //     =>
    // sub_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("sub_dst", "sub_binary_tile");
}

RewriteRule RuleFactory::make_math_rsub_dst_rule() {
    // self.rsub_dst(idst0, idst1);
    //     =>
    // rsub_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("rsub_dst", "rsub_binary_tile");
}

RewriteRule RuleFactory::make_math_mul_dst_rule() {
    // self.mul_dst(idst0, idst1);
    //     =>
    // mul_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("mul_dst", "mul_binary_tile");
}

RewriteRule RuleFactory::make_math_div_dst_rule() {
    // self.div_dst(idst0, idst1);
    //     =>
    // div_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("div_dst", "div_binary_tile");
}

RewriteRule RuleFactory::make_math_power_dst_rule() {
    // self.power_dst(idst0, idst1);
    //     =>
    // power_binary_tile(idst0, idst1);
    return _make_math_eltwise_binary_dst_rule("power_dst", "power_binary_tile");
}

RewriteRule RuleFactory::make_math_abs_rule() {
    // self.abs(idst);
    //     =>
    // abs_tile(idst);
    return _make_math_eltwise_unary_rule("abs", "abs_tile");
}

RewriteRule RuleFactory::make_math_acos_rule() {
    // self.acos(idst);
    //     =>
    // acos_tile(idst);
    return _make_math_eltwise_unary_rule("acos", "acos_tile");
}

RewriteRule RuleFactory::make_math_add_scalar_rule() {
    // self.add_scalar(idst, param);
    //     =>
    // add_unary_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("add_scalar", "add_unary_tile");
}

RewriteRule RuleFactory::make_math_asin_rule() {
    // self.asin(idst);
    //     =>
    // asin_tile(idst);
    return _make_math_eltwise_unary_rule("asin", "asin_tile");
}

RewriteRule RuleFactory::make_math_atan_rule() {
    // self.atan(idst);
    //     =>
    // atan_tile(idst);
    return _make_math_eltwise_unary_rule("atan", "atan_tile");
}

RewriteRule RuleFactory::make_math_cos_rule() {
    // self.cos(idst);
    //     =>
    // cos_tile(idst);
    return _make_math_eltwise_unary_rule("cos", "cos_tile");
}

RewriteRule RuleFactory::make_math_cast_bf16_u16_rule() {
    // self.cast_bf16_u16(idst);
    //     =>
    // tanto_cast_bf16_u16(idst);
    return _make_math_eltwise_unary_rule("cast_bf16_u16", "tanto_cast_bf16_u16");
}

RewriteRule RuleFactory::make_math_cast_u16_bf16_rule() {
    // self.cast_u16_bf16(idst);
    //     =>
    // tanto_cast_u16_bf16(idst);
    return _make_math_eltwise_unary_rule("cast_u16_bf16", "tanto_cast_u16_bf16");
}

RewriteRule RuleFactory::make_math_ceil_rule() {
    // ACHTUNG: 16-bit output assumed
    //     (general case requires choice between 'ceil_tile' and 'ceil_tile_float32'
    // self.ceil(idst);
    //     =>
    // ceil_tile(idst);
    return _make_math_eltwise_unary_approx_rule("ceil", "ceil_tile");
}

RewriteRule RuleFactory::make_math_div_scalar_rule() {
    // self.div_scalar(idst, param);
    //     =>
    // div_unary_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("div_scalar", "div_unary_tile");
}

RewriteRule RuleFactory::make_math_elu_rule() {
    // self.elu(idst, param);
    //     =>
    // elu_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("elu", "elu_tile");
}

RewriteRule RuleFactory::make_math_eqz_rule() {
    // self.eqz(idst);
    //     =>
    // eqz_tile(idst);
    return _make_math_eltwise_unary_rule("eqz", "eqz_tile");
}

RewriteRule RuleFactory::make_math_erf_rule() {
    // self.erf(idst);
    //     =>
    // erf_tile(idst, FAST_AND_APPROX);
    return _make_math_eltwise_unary_approx_rule("erf", "erf_tile");
}

RewriteRule RuleFactory::make_math_erfc_rule() {
    // self.erfc(idst);
    //     =>
    // erfc_tile(idst, FAST_AND_APPROX);
    return _make_math_eltwise_unary_approx_rule("erfc", "erfc_tile");
}

RewriteRule RuleFactory::make_math_erfinv_rule() {
    // self.erfinv(idst);
    //     =>
    // erfinv_tile(idst);
    return _make_math_eltwise_unary_rule("erfinv", "erfinv_tile");
}

RewriteRule RuleFactory::make_math_exp_rule() {
    // self.exp(idst);
    //     =>
    // exp_tile(idst, FAST_AND_APPROX);
    return _make_math_eltwise_unary_approx_rule("exp", "exp_tile");
}

RewriteRule RuleFactory::make_math_exp2_rule() {
    // self.exp2(idst);
    //     =>
    // exp2_tile(idst);
    return _make_math_eltwise_unary_rule("exp2", "exp2_tile");
}

RewriteRule RuleFactory::make_math_expm1_rule() {
    // self.expm1(idst);
    //     =>
    // expm1_tile(idst);
    return _make_math_eltwise_unary_rule("expm1", "expm1_tile");
}

RewriteRule RuleFactory::make_math_fill_rule() {
    // self.fill(idst, param);
    //     =>
    // fill_tile_bitcast(idst, param);
    return _make_math_eltwise_unary_param_rule("fill", "fill_tile_bitcast");
}

RewriteRule RuleFactory::make_math_floor_rule() {
    // ACHTUNG: 16-bit output assumed
    //     (general case requires choice between 'floor_tile' and 'floor_tile_float32'
    // self.floor(idst);
    //     =>
    // floor_tile(idst);
    return _make_math_eltwise_unary_approx_rule("floor", "floor_tile");
}

RewriteRule RuleFactory::make_math_gelu_rule() {
    // self.gelu(idst);
    //     =>
    // gelu_tile(idst, FAST_AND_APPROX);
    return _make_math_eltwise_unary_approx_rule("gelu", "gelu_tile");
}

RewriteRule RuleFactory::make_math_gez_rule() {
    // self.gez(idst);
    //     =>
    // gez_tile(idst);
    return _make_math_eltwise_unary_rule("gez", "gez_tile");
}

RewriteRule RuleFactory::make_math_gtz_rule() {
    // self.gtz(idst);
    //     =>
    // gtz_tile(idst);
    return _make_math_eltwise_unary_rule("gtz", "gtz_tile");
}

RewriteRule RuleFactory::make_math_heaviside_rule() {
    // self.heaviside(idst, param);
    //     =>
    // heaviside_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("heaviside", "heaviside_tile");
}

RewriteRule RuleFactory::make_math_i0_rule() {
    // self.i0(idst);
    //     =>
    // i0_tile(idst);
    return _make_math_eltwise_unary_rule("i0", "i0_tile");
}

RewriteRule RuleFactory::make_math_isfinite_rule() {
    // self.isfinite(idst);
    //     =>
    // isfinite_tile(idst);
    return _make_math_eltwise_unary_rule("isfinite", "isfinite_tile");
}

RewriteRule RuleFactory::make_math_isinf_rule() {
    // self.isinf(idst);
    //     =>
    // isinf_tile(idst);
    return _make_math_eltwise_unary_rule("isinf", "isinf_tile");
}

RewriteRule RuleFactory::make_math_isnan_rule() {
    // self.isnan(idst);
    //     =>
    // isnan_tile(idst);
    return _make_math_eltwise_unary_rule("isnan", "isnan_tile");
}

RewriteRule RuleFactory::make_math_isneginf_rule() {
    // self.isneginf(idst);
    //     =>
    // isneginf_tile(idst);
    return _make_math_eltwise_unary_rule("isneginf", "isneginf_tile");
}

RewriteRule RuleFactory::make_math_isposinf_rule() {
    // self.isposinf(idst);
    //     =>
    // isposinf_tile(idst);
    return _make_math_eltwise_unary_rule("isposinf", "isposinf_tile");
}

RewriteRule RuleFactory::make_math_leaky_relu_rule() {
    // self.leaky_relu(idst, param);
    //     =>
    // leaky_relu_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("leaky_relu", "leaky_relu_tile");
}

RewriteRule RuleFactory::make_math_lez_rule() {
    // self.lez(idst);
    //     =>
    // lez_tile(idst);
    return _make_math_eltwise_unary_rule("lez", "lez_tile");
}

RewriteRule RuleFactory::make_math_log_rule() {
    // self.log(idst);
    //     =>
    // log_tile(idst);
    return _make_math_eltwise_unary_rule("log", "log_tile");
}

RewriteRule RuleFactory::make_math_log_with_base_rule() {
    // self.log_with_base(idst, param);
    //     =>
    // log_with_base_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("log_with_base", "log_with_base_tile");
}

RewriteRule RuleFactory::make_math_logical_not_rule() {
    // self.logical_not(idst);
    //     =>
    // logical_not_tile(idst);
    return _make_math_eltwise_unary_rule("logical_not", "logical_not_tile");
}

RewriteRule RuleFactory::make_math_ltz_rule() {
    // self.ltz(idst);
    //     =>
    // ltz_tile(idst);
    return _make_math_eltwise_unary_rule("ltz", "ltz_tile");
}

RewriteRule RuleFactory::make_math_max_rule() {
    // self.max(idst);
    //     =>
    // tanto_max_tile(idst);
    return _make_math_eltwise_unary_rule("max", "tanto_max_tile");
}

RewriteRule RuleFactory::make_math_mul_scalar_rule() {
    // self.mul_scalar(idst, param);
    //     =>
    // mul_unary_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("mul_scalar", "mul_unary_tile");
}

RewriteRule RuleFactory::make_math_nez_rule() {
    // self.nez(idst);
    //     =>
    // nez_tile(idst);
    return _make_math_eltwise_unary_rule("nez", "nez_tile");
}

RewriteRule RuleFactory::make_math_power_rule() {
    // self.power(idst, param);
    //     =>
    // power_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("power", "power_tile");
}

RewriteRule RuleFactory::make_math_recip_rule() {
    // self.recip(idst);
    //     =>
    // recip_tile(idst);
    return _make_math_eltwise_unary_rule("recip", "recip_tile");
}

RewriteRule RuleFactory::make_math_relu_rule() {
    // self.relu(idst);
    //     =>
    // relu_tile(idst);
    return _make_math_eltwise_unary_rule("relu", "relu_tile");
}

RewriteRule RuleFactory::make_math_relu_max_rule() {
    // self.relu_max(idst, param);
    //     =>
    // relu_max_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("relu_max", "relu_max_tile");
}

RewriteRule RuleFactory::make_math_relu_min_rule() {
    // self.relu_min(idst, param);
    //     =>
    // relu_min_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("relu_min", "relu_min_tile");
}

RewriteRule RuleFactory::make_math_rsqrt_rule() {
    // self.rsqrt(idst);
    //     =>
    // rsqrt_tile(idst);
    return _make_math_eltwise_unary_approx_rule("rsqrt", "rsqrt_tile");
}

RewriteRule RuleFactory::make_math_rsub_scalar_rule() {
    // self.rsub_scalar(idst, param);
    //     =>
    // rsub_unary_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("rsub_scalar", "rsub_unary_tile");
}

RewriteRule RuleFactory::make_math_sigmoid_rule() {
    // self.sigmoid(idst);
    //     =>
    // sigmoid_tile(idst);
    return _make_math_eltwise_unary_rule("sigmoid", "sigmoid_tile");
}

RewriteRule RuleFactory::make_math_sign_rule() {
    // self.sign(idst);
    //     =>
    // sign_tile(idst);
    return _make_math_eltwise_unary_rule("sign", "sign_tile");
}

RewriteRule RuleFactory::make_math_signbit_rule() {
    // self.signbit(idst);
    //     =>
    // signbit_tile(idst);
    return _make_math_eltwise_unary_rule("signbit", "signbit_tile");
}

RewriteRule RuleFactory::make_math_sin_rule() {
    // self.sin(idst);
    //     =>
    // sin_tile(idst);
    return _make_math_eltwise_unary_rule("sin", "sin_tile");
}

RewriteRule RuleFactory::make_math_sqrt_rule() {
    // self.sqrt(idst);
    //     =>
    // sqrt_tile(idst);
    return _make_math_eltwise_unary_rule("sqrt", "sqrt_tile");
}

RewriteRule RuleFactory::make_math_square_rule() {
    // self.square(idst);
    //     =>
    // square_tile(idst);
    return _make_math_eltwise_unary_rule("square", "square_tile");
}

RewriteRule RuleFactory::make_math_sub_scalar_rule() {
    // self.sub_scalar(idst, param);
    //     =>
    // sub_unary_tile(idst, param);
    return _make_math_eltwise_unary_param_rule("sub_scalar", "sub_unary_tile");
}

RewriteRule RuleFactory::make_math_tan_rule() {
    // self.tan(idst);
    //     =>
    // tan_tile(idst);
    return _make_math_eltwise_unary_rule("tan", "tan_tile");
}

RewriteRule RuleFactory::make_math_tanh_rule() {
    // self.tanh(idst);
    //     =>
    // tanh_tile(idst);
    return _make_math_eltwise_unary_rule("tanh", "tanh_tile");
}

RewriteRule RuleFactory::make_math_arg_rule() {
    // remove call arguments of type "math"
    // (mast follow all rules for math method calls)
    return makeRule(
        expr(
            hasType(cxxRecordDecl(hasName("math"))),
            hasParent(callExpr())).bind("arg"),
        remove(node("arg")));
}

// functions

RewriteRule RuleFactory::make_func_tilize_block_rule() {
    // tilize_block(src, block, dst);
    //     =>
    // tilize_block(src.cb_id, block, dst.cb_id);
    return makeRule(
        make_func_call_3_matcher("tilize_block"),
        changeTo(
            statement("stmt"), 
            cat(
                "tilize_block(", 
                    access("arg0", "cb_id"), ", ", 
                    node("arg1"), ", ",
                    access("arg2", "cb_id"), ");")));
}

RewriteRule RuleFactory::make_func_untilize_block_rule() {
    // untilize_block(src, block, dst);
    //     =>
    // untilize_block<1>(src.cb_id, block, dst.cb_id);
    return makeRule(
        make_func_call_3_matcher("untilize_block"),
        changeTo(
            statement("stmt"), 
            cat(
                "untilize_block<1>(", 
                    access("arg0", "cb_id"), ", ", 
                    node("arg1"), ", ",
                    access("arg2", "cb_id"), ");")));
}

} // namespace front
} // namespace tanto
} // namespace ronin

