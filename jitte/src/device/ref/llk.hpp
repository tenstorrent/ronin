// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "core/llk_defs.hpp"
#include "core/memory.hpp"
#include "core/cb_api.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

enum class SfpuBinaryOp {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    RSUB = 4,
    POW = 5
};

class LLK {
public:
    LLK(Memory *l1, CB *cb);
    ~LLK();
public:
    void reset();
public:
    void acquire_dst();
public:
    // pack
    void pack(uint32_t tile_index, uint32_t output);
    void matmul_pack(uint32_t tile_index, uint32_t output, uint32_t ntiles);
    void pack_relu_config(uint32_t config);
    void pack_block(uint32_t block_index, uint32_t output);
    void pack_block_raw(uint32_t block_index, uint32_t output);
    void pack_untilize(
        uint32_t block_rt_dim, 
        uint32_t output, 
        uint32_t block_c_index,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim);
public:
    // unpack
    void unpack_A(uint32_t operand, uint32_t tile_index, bool transpose_xy);
    void unpack_AB(
        BroadcastType BType,
        uint32_t operandA, 
        uint32_t operandB, 
        uint32_t tile_index_a, 
        uint32_t tile_index_b);
    void unpack_AB_matmul(
        uint32_t operandA, 
        uint32_t operandB, 
        uint32_t tile_index_a, 
        uint32_t tile_index_b);
    void unpack_tilize(uint32_t icb, uint32_t block);
    void unpack_untilize(uint32_t icb, uint32_t block); // DEPRECATED
public:
    // math
    void math_eltwise_binary(
        EltwiseBinaryType eltwise_binary_type,
        BroadcastType src_b_bcast_type,
        uint32_t dst_index);
    void math_eltwise_unary_datacopy(uint32_t dst_index);
    void math_matmul(uint32_t dst_index, bool transpose);
    void math_reduce(PoolType type, ReduceDim dim, uint32_t dst_index);
public:
    // math/sfpu
    void math_eltwise_binary_sfpu_copy_dest_values(uint32_t dst_index0, uint32_t dst_index1);
    void math_eltwise_binary_sfpu_binop(
        SfpuBinaryOp binop, 
        uint32_t dst_index0, 
        uint32_t dst_index1);
    void math_eltwise_unary_sfpu_rsqrt(uint32_t dst_index, bool approx);
    void math_eltwise_unary_sfpu_sigmoid(uint32_t dst_index);
    void math_eltwise_unary_sfpu_log(uint32_t dst_index);
    void math_eltwise_unary_sfpu_log_with_base(uint32_t dst_index, uint32_t base_scale);
    void math_eltwise_unary_sfpu_tanh(uint32_t dst_index);
    void math_eltwise_unary_sfpu_signbit(uint32_t dst_index);
    void math_eltwise_unary_sfpu_abs(uint32_t dst_index);
    void math_eltwise_unary_sfpu_sign(uint32_t dst_index);
    void math_eltwise_unary_sfpu_square(uint32_t dst_index);
    void math_eltwise_unary_sfpu_ltz(uint32_t dst_index);
    void math_eltwise_unary_sfpu_eqz(uint32_t dst_index);
    void math_eltwise_unary_sfpu_lez(uint32_t dst_index);
    void math_eltwise_unary_sfpu_gtz(uint32_t dst_index);
    void math_eltwise_unary_sfpu_nez(uint32_t dst_index);
    void math_eltwise_unary_sfpu_gez(uint32_t dst_index);
    void math_eltwise_unary_sfpu_power(uint32_t dst_index, uint32_t pow);
    void math_eltwise_unary_sfpu_max(uint32_t dst0_index, uint32_t dst1_index);
    void math_eltwise_unary_sfpu_exp2(uint32_t dst_index);
    void math_eltwise_unary_sfpu_heaviside(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_expm1(uint32_t dst_index);
    void math_eltwise_unary_sfpu_asin(uint32_t dst_index);
    void math_eltwise_unary_sfpu_atan(uint32_t dst_index);
    void math_eltwise_unary_sfpu_acos(uint32_t dst_index);
    void math_eltwise_unary_sfpu_add_scalar(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_sub_scalar(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_mul_scalar(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_div_scalar(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_rsub_scalar(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_ceil(uint32_t dst_index);
    void math_eltwise_unary_sfpu_ceil_float32(uint32_t dst_index);
    void math_eltwise_unary_sfpu_elu(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_erf(uint32_t dst_index, bool approx);
    void math_eltwise_unary_sfpu_erfc(uint32_t dst_index, bool approx);
    void math_eltwise_unary_sfpu_erfinv(uint32_t dst_index);
    void math_eltwise_unary_sfpu_exponential(uint32_t dst_index);
    void math_eltwise_unary_sfpu_fill_bitcast(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_floor(uint32_t dst_index);
    void math_eltwise_unary_sfpu_floor_float32(uint32_t dst_index);
    void math_eltwise_unary_sfpu_gelu(uint32_t dst_index, bool approx);
    void math_eltwise_unary_sfpu_i0(uint32_t dst_index);
    void math_eltwise_unary_sfpu_isinf(uint32_t dst_index);
    void math_eltwise_unary_sfpu_isposinf(uint32_t dst_index);
    void math_eltwise_unary_sfpu_isneginf(uint32_t dst_index);
    void math_eltwise_unary_sfpu_isnan(uint32_t dst_index);
    void math_eltwise_unary_sfpu_isfinite(uint32_t dst_index);
    void math_eltwise_unary_sfpu_logical_not_unary(uint32_t dst_index);
    void math_eltwise_unary_sfpu_reciprocal(uint32_t dst_index);
    void math_eltwise_unary_sfpu_relu_max(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_relu_min(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_relu(uint32_t dst_index);
    void math_eltwise_unary_sfpu_lrelu(uint32_t dst_index, uint32_t param0);
    void math_eltwise_unary_sfpu_sqrt(uint32_t dst_index);
    void math_eltwise_unary_sfpu_sine(uint32_t dst_index);
    void math_eltwise_unary_sfpu_cosine(uint32_t dst_index);
    void math_eltwise_unary_sfpu_tan(uint32_t dst_index);
    void math_eltwise_unary_sfpu_typecast(
        uint32_t in_dtype, 
        uint32_t out_dtype, 
        uint32_t dst_index);
private:
    void reserve_block(uint32_t tiles);
    DataFormat get_cb_data_format(uint32_t operand);
    uint32_t get_cb_tile_offset(uint32_t operand, uint32_t tile_index);
    uint8_t *get_cb_read_ptr(uint32_t operand);
    uint8_t *get_cb_write_ptr(uint32_t operand);
    float *get_dst_ptr(uint32_t dst_index);
    static void copy_tile(float *dst, const float *src);
    static void fill_tile(float *dst, float value);
private:
    static const uint32_t TILE_DIM = 32;
    static const uint32_t TILE_SIZE = TILE_DIM * TILE_DIM;
#if 0 // TODO: Revise this
    static const uint32_t DST_COUNT = 16;
#endif
    static const uint32_t DST_COUNT = 8;
private:
    Memory *m_l1;
    CB *m_cb;
    std::vector<float> m_src_a;
    std::vector<float> m_src_b;
    std::vector<float> m_dst;
    std::vector<float> m_tile;
    std::vector<float> m_temp;
    ReluType m_relu_mode;
    float m_relu_threshold;
    std::vector<bool> m_dst_valid;
    std::vector<float> m_block; // for unpack_tilize/untilize
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

