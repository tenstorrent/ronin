// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <limits>
#include <algorithm>

#include "host/ref/reduce_ref.hpp"

namespace ronin {
namespace op {
namespace reduce {
namespace ref {

//
//    ReduceMaxRef
//

ReduceMaxRef::ReduceMaxRef(
        int N,
        int H,
        int W,
        int axis):
            m_N(N),
            m_H(H),
            m_W(W),
            m_axis(axis) { }

ReduceMaxRef::~ReduceMaxRef() { }

void ReduceMaxRef::init(const float *x, float *y) {
    assert(m_axis == 1 || m_axis == 2);
    m_x = x;
    m_y = y;
}

void ReduceMaxRef::run() {
    constexpr float init_acc = std::numeric_limits<float>::lowest();
    if (m_axis == 1) {
        int HW = m_H * m_W;
        for (int n = 0; n < m_N; n++) {
            for (int w = 0; w < m_W; w++) {
                float acc = init_acc;
                for (int h = 0; h < m_H; h++) {
                    acc = std::max(acc, m_x[n * HW + h * m_W + w]);
                }
                m_y[n * m_W + w] = acc;
            }
        }
    } else {
        int NH = m_N * m_H;
        for (int nh = 0; nh < NH; nh++) {
            float acc = init_acc;
            for (int w = 0; w < m_W; w++) {
                acc = std::max(acc, m_x[nh * m_W + w]);
            }
            m_y[nh] = acc;
        }
    }
}

int ReduceMaxRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W;
}

int ReduceMaxRef::output_volume(int index) {
    assert(index == 0);
    return (m_axis == 1) ? m_N * m_W : m_N * m_H;
}

//
//    ReduceMeanRef
//

ReduceMeanRef::ReduceMeanRef(
        int N,
        int H,
        int W,
        int axis):
            m_N(N),
            m_H(H),
            m_W(W),
            m_axis(axis) { }

ReduceMeanRef::~ReduceMeanRef() { }

void ReduceMeanRef::init(const float *x, float *y) {
    assert(m_axis == 1 || m_axis == 2);
    m_x = x;
    m_y = y;
}

void ReduceMeanRef::run() {
    if (m_axis == 1) {
        float scale = 1.0f / float(m_H);
        int HW = m_H * m_W;
        for (int n = 0; n < m_N; n++) {
            for (int w = 0; w < m_W; w++) {
                float acc = 0.0f;
                for (int h = 0; h < m_H; h++) {
                    acc += m_x[n * HW + h * m_W + w];
                }
                m_y[n * m_W + w] = acc * scale;
            }
        }
    } else {
        float scale = 1.0f / float(m_W);
        int NH = m_N * m_H;
        for (int nh = 0; nh < NH; nh++) {
            float acc = 0.0f;
            for (int w = 0; w < m_W; w++) {
                acc += m_x[nh * m_W + w];
            }
            m_y[nh] = acc * scale;
        }
    }
}

int ReduceMeanRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W;
}

int ReduceMeanRef::output_volume(int index) {
    assert(index == 0);
    return (m_axis == 1) ? m_N * m_W : m_N * m_H;
}

//
//    ReduceSumRef
//

ReduceSumRef::ReduceSumRef(
        int N,
        int H,
        int W,
        int axis):
            m_N(N),
            m_H(H),
            m_W(W),
            m_axis(axis) { }

ReduceSumRef::~ReduceSumRef() { }

void ReduceSumRef::init(const float *x, float *y) {
    assert(m_axis == 1 || m_axis == 2);
    m_x = x;
    m_y = y;
}

void ReduceSumRef::run() {
    if (m_axis == 1) {
        int HW = m_H * m_W;
        for (int n = 0; n < m_N; n++) {
            for (int w = 0; w < m_W; w++) {
                float acc = 0.0f;
                for (int h = 0; h < m_H; h++) {
                    acc += m_x[n * HW + h * m_W + w];
                }
                m_y[n * m_W + w] = acc;
            }
        }
    } else {
        int NH = m_N * m_H;
        for (int nh = 0; nh < NH; nh++) {
            float acc = 0.0f;
            for (int w = 0; w < m_W; w++) {
                acc += m_x[nh * m_W + w];
            }
            m_y[nh] = acc;
        }
    }
}

int ReduceSumRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W;
}

int ReduceSumRef::output_volume(int index) {
    assert(index == 0);
    return (m_axis == 1) ? m_N * m_W : m_N * m_H;
}

} // namespace ref
} // namespace reduce
} // namespace op
} // namespace ronin

