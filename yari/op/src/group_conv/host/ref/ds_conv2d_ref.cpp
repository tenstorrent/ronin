// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "host/base/post_op.hpp"

#include "host/ref/ds_conv2d_ref.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace ref {

namespace base = ronin::op::common::base;

//
//    DSConv2dRef
//

DSConv2dRef::DSConv2dRef(
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

DSConv2dRef::~DSConv2dRef() { }

void DSConv2dRef::init(
        const float *x,
        const float *w,
        const float *b,
        const float *w2,
        const float *b2,
        const float *z,
        float *y) {
    m_x = x;
    m_w = w;
    m_b = b;
    m_w2 = w2;
    m_b2 = b2;
    m_z = z;
    m_y = y;
    m_t.resize(m_C);
}

void DSConv2dRef::run() {
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int QC = m_Q * m_C;
    int PQC = m_P * QC;
    int QK = m_Q * m_K;
    int PQK = m_P * QK;
    int SC = m_S * m_C;
    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
        // depthwise
        for (int c = 0; c < m_C; c++) {
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
                int xpos = n * HWC + th * WC + tw * m_C + c;
                int wpos = r * SC + s * m_C + c;
                acc += m_x[xpos] * m_w[wpos];
            } // s
            } // r
            acc += m_b[c];
            acc = m_post_op.eval(acc);
            m_t[c] = acc;
        } // c
        // pointwise
        int wpos = 0;
        for (int k = 0; k < m_K; k++) { 
            float acc = 0.0f;
            for (int c = 0; c < m_C; c++) {
                acc += m_t[c] * m_w2[wpos];
                wpos++;
            } // c
            acc += m_b2[k];
            int ypos = n * PQK + p * QK + q * m_K + k;
            if (m_z != nullptr) {
                acc += m_z[ypos];
            }
            m_y[ypos] = acc;
        } // k
    } // q
    } // p
    } // n
}

int DSConv2dRef::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * m_H * m_W * m_C;
    case 1:
        return m_R * m_S * m_C;
    case 2:
        return m_C;
    case 3:
        return m_R * m_S * m_K * m_C;
    case 4:
        return m_K;
    case 5:
        return m_N * m_P * m_Q * m_K;
    default:
        assert(false);
        return 0;
    }
}

int DSConv2dRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_K;
}

} // namespace ref
} // namespace group_conv
} // namespace op
} // namespace ronin

