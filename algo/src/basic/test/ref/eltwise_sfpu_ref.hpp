// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

enum class EltwiseSfpuRefOp {
    Abs,
    Acos,
    Asin,
    Atan,
    Cos,
    Elu,
    Eqz,
    Erf,
    Erfc,
    Erfinv,
    Exp,
    Exp2,
    Expm1,
    Gelu,
    Gez,
    Gtz,
    Heaviside,
    I0,
    Isfinite,
    Isinf,
    Isnan,
    Isneginf,
    Isposinf,
    LeakyRelu,
    Lez,
    Log,
    LogWithBase,
    LogicalNot,
    Ltz,
    Nez,
    Power,
    Recip,
    Relu,
    ReluMax,
    ReluMin,
    Rsqrt,
    Sigmoid,
    Sign,
    Signbit,
    Sin,
    Sqrt,
    Square,
    Tan,
    Tanh
};

class EltwiseSfpuRef {
public:
    EltwiseSfpuRef();
    ~EltwiseSfpuRef();
public:
    void init(
        EltwiseSfpuRefOp op, 
        uint32_t iparam,
        float fparam,
        int N);
    void run(const float *x, float *y);
private:
    void abs(const float *x, float *y);
    void acos(const float *x, float *y);
    void asin(const float *x, float *y);
    void atan(const float *x, float *y);
    void cos(const float *x, float *y);
    void elu(const float *x, float *y);
    void eqz(const float *x, float *y);
    void erf(const float *x, float *y);
    void erfc(const float *x, float *y);
    void erfinv(const float *x, float *y);
    void exp(const float *x, float *y);
    void exp2(const float *x, float *y);
    void expm1(const float *x, float *y);
    void gelu(const float *x, float *y);
    void gez(const float *x, float *y);
    void gtz(const float *x, float *y);
    void heaviside(const float *x, float *y);
    void i0(const float *x, float *y);
    void isfinite(const float *x, float *y);
    void isinf(const float *x, float *y);
    void isnan(const float *x, float *y);
    void isneginf(const float *x, float *y);
    void isposinf(const float *x, float *y);
    void leaky_relu(const float *x, float *y);
    void lez(const float *x, float *y);
    void log(const float *x, float *y);
    void log_with_base(const float *x, float *y);
    void logical_not(const float *x, float *y);
    void ltz(const float *x, float *y);
    void nez(const float *x, float *y);
    void power(const float *x, float *y);
    void recip(const float *x, float *y);
    void relu(const float *x, float *y);
    void relu_max(const float *x, float *y);
    void relu_min(const float *x, float *y);
    void rsqrt(const float *x, float *y);
    void sigmoid(const float *x, float *y);
    void sign(const float *x, float *y);
    void signbit(const float *x, float *y);
    void sin(const float *x, float *y);
    void sqrt(const float *x, float *y);
    void square(const float *x, float *y);
    void tan(const float *x, float *y);
    void tanh(const float *x, float *y);
private:
    EltwiseSfpuRefOp m_op = EltwiseSfpuRefOp(0);
    uint32_t m_iparam = 0;
    float m_fparam = 0.0f;
    int m_N = 0;
};

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

