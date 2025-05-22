// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    SFPU_OP_ABS = 0,
    SFPU_OP_ACOS = 1,
    SFPU_OP_ASIN = 2,
    SFPU_OP_ATAN = 3,
    SFPU_OP_COS = 4,
    SFPU_OP_ELU = 5,
    SFPU_OP_EQZ = 6,
    SFPU_OP_ERF = 7,
    SFPU_OP_ERFC = 8,
    SFPU_OP_ERFINV = 9,
    SFPU_OP_EXP = 10,
    SFPU_OP_EXP2 = 11,
    SFPU_OP_EXPM1 = 12,
    SFPU_OP_GELU = 13,
    SFPU_OP_GEZ = 14,
    SFPU_OP_GTZ = 15,
    SFPU_OP_HEAVISIDE = 16,
    SFPU_OP_I0 = 17,
    SFPU_OP_ISFINITE = 18,
    SFPU_OP_ISINF = 19,
    SFPU_OP_ISNAN = 20,
    SFPU_OP_ISNEGINF = 21,
    SFPU_OP_ISPOSINF = 22,
    SFPU_OP_LEAKY_RELU = 23,
    SFPU_OP_LEZ = 24,
    SFPU_OP_LOG = 25,
    SFPU_OP_LOG_WITH_BASE = 26,
    SFPU_OP_LOGICAL_NOT = 27,
    SFPU_OP_LTZ = 28,
    SFPU_OP_NEZ = 29,
    SFPU_OP_POWER = 30,
    SFPU_OP_RECIP = 31,
    SFPU_OP_RELU = 32,
    SFPU_OP_RELU_MAX = 33,
    SFPU_OP_RELU_MIN = 34,
    SFPU_OP_RSQRT = 35,
    SFPU_OP_SIGMOID = 36,
    SFPU_OP_SIGN = 37,
    SFPU_OP_SIGNBIT = 38,
    SFPU_OP_SIN = 39,
    SFPU_OP_SQRT = 40,
    SFPU_OP_SQUARE = 41,
    SFPU_OP_TAN = 42,
    SFPU_OP_TANH = 43;

param<uint32> sfpu_op_code;

void sfpu_op(math<T> acc, uint32 iparam, float fparam) {
    if (sfpu_op_code == SFPU_OP_ABS) {
        acc.abs(0);
    } else if (sfpu_op_code == SFPU_OP_ACOS) {
        acc.acos(0);
    } else if (sfpu_op_code == SFPU_OP_ASIN) {
        acc.asin(0);
    } else if (sfpu_op_code == SFPU_OP_ATAN) {
        acc.atan(0);
    } else if (sfpu_op_code == SFPU_OP_COS) {
        acc.cos(0);
    } else if (sfpu_op_code == SFPU_OP_ELU) {
        acc.elu(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_EQZ) {
        acc.eqz(0);
    } else if (sfpu_op_code == SFPU_OP_ERF) {
        acc.erf(0);
    } else if (sfpu_op_code == SFPU_OP_ERFC) {
        acc.erfc(0);
    } else if (sfpu_op_code == SFPU_OP_ERFINV) {
        acc.erfinv(0);
    } else if (sfpu_op_code == SFPU_OP_EXP) {
        acc.exp(0);
    } else if (sfpu_op_code == SFPU_OP_EXP2) {
        acc.exp2(0);
    } else if (sfpu_op_code == SFPU_OP_EXPM1) {
        acc.expm1(0);
    } else if (sfpu_op_code == SFPU_OP_GELU) {
        acc.gelu(0);
    } else if (sfpu_op_code == SFPU_OP_GEZ) {
        acc.gez(0);
    } else if (sfpu_op_code == SFPU_OP_GTZ) {
        acc.gtz(0);
    } else if (sfpu_op_code == SFPU_OP_HEAVISIDE) {
        acc.heaviside(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_I0) {
        acc.i0(0);
    } else if (sfpu_op_code == SFPU_OP_ISFINITE) {
        acc.isfinite(0);
    } else if (sfpu_op_code == SFPU_OP_ISINF) {
        acc.isinf(0);
    } else if (sfpu_op_code == SFPU_OP_ISNAN) {
        acc.isnan(0);
    } else if (sfpu_op_code == SFPU_OP_ISNEGINF) {
        acc.isneginf(0);
    } else if (sfpu_op_code == SFPU_OP_ISPOSINF) {
        acc.isposinf(0);
    } else if (sfpu_op_code == SFPU_OP_LEAKY_RELU) {
        acc.leaky_relu(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_LEZ) {
        acc.lez(0);
    } else if (sfpu_op_code == SFPU_OP_LOG) {
        acc.log(0);
    } else if (sfpu_op_code == SFPU_OP_LOG_WITH_BASE) {
        acc.log_with_base(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_LOGICAL_NOT) {
        acc.logical_not(0);
    } else if (sfpu_op_code == SFPU_OP_LTZ) {
        acc.ltz(0);
    } else if (sfpu_op_code == SFPU_OP_NEZ) {
        acc.nez(0);
    } else if (sfpu_op_code == SFPU_OP_POWER) {
        acc.power(0, iparam);
    } else if (sfpu_op_code == SFPU_OP_RECIP) {
        acc.recip(0);
    } else if (sfpu_op_code == SFPU_OP_RELU) {
        acc.relu(0);
    } else if (sfpu_op_code == SFPU_OP_RELU_MAX) {
        acc.relu_max(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_RELU_MIN) {
        acc.relu_min(0, fparam);
    } else if (sfpu_op_code == SFPU_OP_RSQRT) {
        acc.rsqrt(0);
    } else if (sfpu_op_code == SFPU_OP_SIGMOID) {
        acc.sigmoid(0);
    } else if (sfpu_op_code == SFPU_OP_SIGN) {
        acc.sign(0);
    } else if (sfpu_op_code == SFPU_OP_SIGNBIT) {
        acc.signbit(0);
    } else if (sfpu_op_code == SFPU_OP_SIN) {
        acc.sin(0);
    } else if (sfpu_op_code == SFPU_OP_SQRT) {
        acc.sqrt(0);
    } else if (sfpu_op_code == SFPU_OP_SQUARE) {
        acc.square(0);
    } else if (sfpu_op_code == SFPU_OP_TAN) {
        acc.tan(0);
    } else if (sfpu_op_code == SFPU_OP_TANH) {
        acc.tanh(0);
    }
}

void kernel(
        pipe<T> px, 
        pipe<T> py, 
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 iparam, 
        float fparam) {
    for (uint32 block = 0; block < num_blocks; block++) {
        py.reserve_back();
        px.wait_front();
        for (uint32 i = 0; i < block_tiles; i++) {
            math<T> acc;
            acc.copy(px, i, 0);
            sfpu_op(acc, iparam, fparam);
            acc.pack(0, py);
        }
        px.pop_front();
        py.push_back();
    }
}

