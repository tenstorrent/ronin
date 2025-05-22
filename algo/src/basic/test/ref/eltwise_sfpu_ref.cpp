// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cmath>
#include <cassert>

#include "test/ref/eltwise_sfpu_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

namespace {

inline float f_elu(float x, float slope) {
    return (x <= 0.0) ? slope * (std::exp(x) - 1) : x;
}

inline float f_gelu(float x) {
    float SQRT_2_DIV_PI = 0.7978845608028654f; // sqrt(2.0 / PI);
    return 0.5f * x * (1.0f + std::tanh(SQRT_2_DIV_PI * (x + 0.044715f * x * x * x)));
}

inline float f_leaky_relu(float x, float slope) {
    return (x <= 0.0f) ? slope * x : x;
}

inline float f_relu(float x) {
    return (x <= 0.0f) ? 0.0f : x;
}

inline float f_relu_max(float x, float threshold) {
    return (x > threshold) ? threshold : (x < 0.0f) ? 0.0f : x;
}

inline float f_relu_min(float x, float threshold) {
    return (x <= threshold) ? 0.0f : (x < 0.0f) ? 0.0f : x;
}

inline float f_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace

//
//    EltwiseSfpuRef
//

EltwiseSfpuRef::EltwiseSfpuRef() { }

EltwiseSfpuRef::~EltwiseSfpuRef() { }

void EltwiseSfpuRef::init(
        EltwiseSfpuRefOp op, 
        uint32_t iparam,
        float fparam,
        int N) {
    m_op = op;
    m_iparam = iparam;
    m_fparam = fparam;
    m_N = N;
}

void EltwiseSfpuRef::run(const float *x, float *y) {
    switch (m_op) {
    case EltwiseSfpuRefOp::Abs:
        abs(x, y);
        break;
    case EltwiseSfpuRefOp::Acos:
        acos(x, y);
        break;
    case EltwiseSfpuRefOp::Asin:
        asin(x, y);
        break;
    case EltwiseSfpuRefOp::Atan:
        atan(x, y);
        break;
    case EltwiseSfpuRefOp::Cos:
        cos(x, y);
        break;
    case EltwiseSfpuRefOp::Elu:
        elu(x, y);
        break;
    case EltwiseSfpuRefOp::Eqz:
        eqz(x, y);
        break;
    case EltwiseSfpuRefOp::Erf:
        erf(x, y);
        break;
    case EltwiseSfpuRefOp::Erfc:
        erfc(x, y);
        break;
    case EltwiseSfpuRefOp::Erfinv:
        erfinv(x, y);
        break;
    case EltwiseSfpuRefOp::Exp:
        exp(x, y);
        break;
    case EltwiseSfpuRefOp::Exp2:
        exp2(x, y);
        break;
    case EltwiseSfpuRefOp::Expm1:
        expm1(x, y);
        break;
    case EltwiseSfpuRefOp::Gelu:
        gelu(x, y);
        break;
    case EltwiseSfpuRefOp::Gez:
        gez(x, y);
        break;
    case EltwiseSfpuRefOp::Gtz:
        gtz(x, y);
        break;
    case EltwiseSfpuRefOp::Heaviside:
        heaviside(x, y);
        break;
    case EltwiseSfpuRefOp::I0:
        i0(x, y);
        break;
    case EltwiseSfpuRefOp::Isfinite:
        isfinite(x, y);
        break;
    case EltwiseSfpuRefOp::Isinf:
        isinf(x, y);
        break;
    case EltwiseSfpuRefOp::Isnan:
        isnan(x, y);
        break;
    case EltwiseSfpuRefOp::Isneginf:
        isneginf(x, y);
        break;
    case EltwiseSfpuRefOp::Isposinf:
        isposinf(x, y);
        break;
    case EltwiseSfpuRefOp::LeakyRelu:
        leaky_relu(x, y);
        break;
    case EltwiseSfpuRefOp::Lez:
        lez(x, y);
        break;
    case EltwiseSfpuRefOp::Log:
        log(x, y);
        break;
    case EltwiseSfpuRefOp::LogWithBase:
        log_with_base(x, y);
        break;
    case EltwiseSfpuRefOp::LogicalNot:
        logical_not(x, y);
        break;
    case EltwiseSfpuRefOp::Ltz:
        ltz(x, y);
        break;
    case EltwiseSfpuRefOp::Nez:
        nez(x, y);
        break;
    case EltwiseSfpuRefOp::Power:
        power(x, y);
        break;
    case EltwiseSfpuRefOp::Recip:
        recip(x, y);
        break;
    case EltwiseSfpuRefOp::Relu:
        relu(x, y);
        break;
    case EltwiseSfpuRefOp::ReluMax:
        relu_max(x, y);
        break;
    case EltwiseSfpuRefOp::ReluMin:
        relu_min(x, y);
        break;
    case EltwiseSfpuRefOp::Rsqrt:
        rsqrt(x, y);
        break;
    case EltwiseSfpuRefOp::Sigmoid:
        sigmoid(x, y);
        break;
    case EltwiseSfpuRefOp::Sign:
        sign(x, y);
        break;
    case EltwiseSfpuRefOp::Signbit:
        signbit(x, y);
        break;
    case EltwiseSfpuRefOp::Sin:
        sin(x, y);
        break;
    case EltwiseSfpuRefOp::Sqrt:
        sqrt(x, y);
        break;
    case EltwiseSfpuRefOp::Square:
        square(x, y);
        break;
    case EltwiseSfpuRefOp::Tan:
        tan(x, y);
        break;
    case EltwiseSfpuRefOp::Tanh:
        tanh(x, y);
        break;
    default:
        assert(false);
        break;
    }
}

void EltwiseSfpuRef::abs(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::abs(x[i]);
    }
}

void EltwiseSfpuRef::acos(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::acos(x[i]);
    }
}

void EltwiseSfpuRef::asin(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::asin(x[i]);
    }
}

void EltwiseSfpuRef::atan(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::atan(x[i]);
    }
}

void EltwiseSfpuRef::cos(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::cos(x[i]);
    }
}

void EltwiseSfpuRef::elu(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_elu(x[i], m_fparam);
    }
}

void EltwiseSfpuRef::eqz(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] == 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::erf(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::erf(x[i]);
    }
}

void EltwiseSfpuRef::erfc(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::erfc(x[i]);
    }
}

void EltwiseSfpuRef::erfinv(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::exp(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::exp(x[i]);
    }
}

void EltwiseSfpuRef::exp2(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::exp2(x[i]);
    }
}

void EltwiseSfpuRef::expm1(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::expm1(x[i]);
    }
}

void EltwiseSfpuRef::gelu(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_gelu(x[i]);
    }
}

void EltwiseSfpuRef::gez(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] >= 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::gtz(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::heaviside(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::i0(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::isfinite(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::isfinite(x[i]);
    }
}

void EltwiseSfpuRef::isinf(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::isinf(x[i]);
    }
}

void EltwiseSfpuRef::isnan(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::isnan(x[i]);
    }
}

void EltwiseSfpuRef::isneginf(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::isposinf(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::leaky_relu(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_leaky_relu(x[i], m_fparam);
    }
}

void EltwiseSfpuRef::lez(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] <= 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::log(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::log(x[i]);
    }
}

void EltwiseSfpuRef::log_with_base(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::logical_not(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::ltz(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] < 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::nez(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = (x[i] != 0.0f) ? 1.0f : 0.0f;
    }
}

void EltwiseSfpuRef::power(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::recip(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = 1.0f / x[i];
    }
}

void EltwiseSfpuRef::relu(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_relu(x[i]);
    }
}

void EltwiseSfpuRef::relu_max(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_relu_max(x[i], m_fparam);
    }
}

void EltwiseSfpuRef::relu_min(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_relu_min(x[i], m_fparam);
    }
}

void EltwiseSfpuRef::rsqrt(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::sigmoid(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = f_sigmoid(x[i]);
    }
}

void EltwiseSfpuRef::sign(const float *x, float *y) {
    // TODO
}

void EltwiseSfpuRef::signbit(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::signbit(x[i]);
    }
}

void EltwiseSfpuRef::sin(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::sin(x[i]);
    }
}

void EltwiseSfpuRef::sqrt(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::sqrt(x[i]);
    }
}

void EltwiseSfpuRef::square(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        float t = x[i];
        y[i] = t * t;
    }
}

void EltwiseSfpuRef::tan(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::tan(x[i]);
    }
}

void EltwiseSfpuRef::tanh(const float *x, float *y) {
    for (int i = 0; i < m_N; i++) {
        y[i] = std::tanh(x[i]);
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

