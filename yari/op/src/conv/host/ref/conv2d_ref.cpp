// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <algorithm>

#include "host/base/post_op.hpp"

#include "host/ref/conv2d_ref.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace ref {

namespace base = ronin::op::common::base;

//
//    Conv2dRef
//

Conv2dRef::Conv2dRef(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int K,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        const base::PostOpSpec &post_op):
            m_N(N),
            m_H(H),
            m_W(W),
            m_C(C),
            m_P(P),
            m_Q(Q),
            m_K(K),
            m_R(R),
            m_S(S),
            m_pad_h(pad_h),
            m_pad_w(pad_w),
            m_stride_h(stride_h),
            m_stride_w(stride_w),
            m_dilation_h(dilation_h),
            m_dilation_w(dilation_w),
            m_post_op(post_op) { }

Conv2dRef::~Conv2dRef() { }

void Conv2dRef::init(
        const float *x,
        const float *w,
        const float *b,
        const float *z,
        float *y) {
    m_x = x;
    m_w = w;
    m_b = b;
    m_y = y;
    m_z = z;
}

void Conv2dRef::run() { 
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int KC = m_K * m_C;
    int SKC = m_S * KC;
    int QK = m_Q * m_K;
    int PQK = m_P * QK;
    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
    for (int k = 0; k < m_K; k++) { 
        float acc = 0.0f;
        for (int r = 0; r < m_R; r++) {
        for (int s = 0; s < m_S; s++) {
            int th = p * m_stride_h - m_pad_h + r * m_dilation_h;
            if (th < 0 || th >= m_H) {
                continue;
            }
            int tw = q * m_stride_w - m_pad_w + s * m_dilation_w;
            if (tw < 0 || tw >= m_W) {
                continue;
            }
            int xpos = n * HWC + th * WC + tw * m_C;
            int wpos = r * SKC + s * KC + k * m_C;
            for (int c = 0; c < m_C; c++) {
                acc += m_x[xpos] * m_w[wpos];
                xpos++;
                wpos++;
            } // c
        } // s
        } // r
        if (m_b != nullptr) {
            acc += m_b[k];
        }
        int ypos = n * PQK + p * QK + q * m_K + k;
        if (m_z != nullptr) {
            acc += m_z[ypos];
        }
        acc = m_post_op.eval(acc);
        m_y[ypos] = acc;
    } // k
    } // q
    } // p
    } // n
}

int Conv2dRef::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N *  m_H * m_W * m_C;
    case 1:
        return m_R * m_S * m_K * m_C;
    case 2:
        return m_K;
    case 3:
        return m_N * m_P * m_Q * m_K;
    default:
        assert(false);
        return 0;
    }
}

int Conv2dRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_K;
}

} // namespace ref
} // namespace conv
} // namespace op
} // namespace ronin

