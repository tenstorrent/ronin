// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "test/ref/eltwise_binary_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    EltwiseBinaryRef
//

EltwiseBinaryRef::EltwiseBinaryRef() { }

EltwiseBinaryRef::~EltwiseBinaryRef() { }

void EltwiseBinaryRef::init(EltwiseBinaryRefOp op, int N) {
    m_op = op;
    m_N = N;
}

void EltwiseBinaryRef::run(
        const float *a,
        const float *b,
        float *c) {
    switch (m_op) {
    case EltwiseBinaryRefOp::Add:
        add(a, b, c);
        break;
    case EltwiseBinaryRefOp::Sub:
        sub(a, b, c);
        break;
    case EltwiseBinaryRefOp::Mul:
        mul(a, b, c);
        break;
    default:
        assert(false);
        break;
    }
}

void EltwiseBinaryRef::add(
        const float *a,
        const float *b,
        float *c) {
    for (int i = 0; i < m_N; i++) {
        c[i] = a[i] + b[i];
    }
}

void EltwiseBinaryRef::sub(
        const float *a,
        const float *b,
        float *c) {
    for (int i = 0; i < m_N; i++) {
        c[i] = a[i] - b[i];
    }
}

void EltwiseBinaryRef::mul(
        const float *a,
        const float *b,
        float *c) {
    for (int i = 0; i < m_N; i++) {
        c[i] = a[i] * b[i];
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

