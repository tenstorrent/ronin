// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <algorithm>

#include "host/base/post_op.hpp"

#include "host/ref/binary_ref.hpp"

namespace ronin {
namespace op {
namespace binary {
namespace ref {

namespace base = ronin::op::common::base;

//
//    AddRef
//

AddRef::AddRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op/* = base::PostOpSpec()*/):
            m_N(N),
            m_H(H),
            m_C(C),
            m_post_op(post_op) { }

AddRef::~AddRef() { }

void AddRef::init(const float *a, const float *b, float *c) {
    m_a = a;
    m_b = b;
    m_c = c;
}

void AddRef::run() {
    int NHC = m_N * m_H * m_C;
    for (int nhc = 0; nhc < NHC; nhc++) {
        float v = m_a[nhc] + m_b[nhc];
        v = m_post_op.eval(v);
        m_c[nhc] = v;
    }
}

int AddRef::input_volume(int index) {
    assert(index == 0 || index == 1);
    return m_N * m_H * m_C;
}

int AddRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

//
//    SubRef
//

SubRef::SubRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op/* = base::PostOpSpec()*/):
            m_N(N),
            m_H(H),
            m_C(C),
            m_post_op(post_op) { }

SubRef::~SubRef() { }

void SubRef::init(const float *a, const float *b, float *c) {
    m_a = a;
    m_b = b;
    m_c = c;
}

void SubRef::run() {
    int NHC = m_N * m_H * m_C;
    for (int nhc = 0; nhc < NHC; nhc++) {
        float v = m_a[nhc] - m_b[nhc];
        v = m_post_op.eval(v);
        m_c[nhc] = v;
    }
}

int SubRef::input_volume(int index) {
    assert(index == 0 || index == 1);
    return m_N * m_H * m_C;
}

int SubRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

//
//    MulRef
//

MulRef::MulRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op/* = base::PostOpSpec()*/):
            m_N(N),
            m_H(H),
            m_C(C),
            m_post_op(post_op) { }

MulRef::~MulRef() { }

void MulRef::init(const float *a, const float *b, float *c) {
    m_a = a;
    m_b = b;
    m_c = c;
}

void MulRef::run() {
    int NHC = m_N * m_H * m_C;
    for (int nhc = 0; nhc < NHC; nhc++) {
        float v = m_a[nhc] * m_b[nhc];
        v = m_post_op.eval(v);
        m_c[nhc] = v;
    }
}

int MulRef::input_volume(int index) {
    assert(index == 0 || index == 1);
    return m_N * m_H * m_C;
}

int MulRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

} // namespace ref
} // namespace binary
} // namespace op
} // namespace ronin

