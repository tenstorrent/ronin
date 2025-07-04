// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cassert>
#include <algorithm>

#include "host/base/post_op.hpp"

#include "host/ref/deform_conv2d_ref.hpp"

namespace ronin {
namespace op {
namespace deform_conv {
namespace ref {

namespace base = ronin::op::common::base;

//
//    DeformConv2dRef
//

DeformConv2dRef::DeformConv2dRef(
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
        int deform_groups,
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
            m_deform_groups(deform_groups),
            m_post_op(post_op) { }

DeformConv2dRef::~DeformConv2dRef() { }

void DeformConv2dRef::init(
        const float *x,
        const float *d,
        const float *w,
        const float *b,
        const float *z,
        float *y) {
    assert(m_C % m_deform_groups == 0);
    m_x = x;
    m_d = d;
    m_w = w;
    m_b = b;
    m_y = y;
    m_z = z;
}

void DeformConv2dRef::run() { 
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int KC = m_K * m_C;
    int SKC = m_S * KC;
    int QK = m_Q * m_K;
    int PQK = m_P * QK;
    int D = 2 * m_deform_groups;
    int SD = m_S * D;
    int RSD = m_R * SD;
    int QRSD = m_Q * RSD;
    int PQRSD = m_P * QRSD;
    int c_stride = m_C / m_deform_groups;
    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
    for (int k = 0; k < m_K; k++) { 
        float acc = 0.0f;
        for (int r = 0; r < m_R; r++) {
        for (int s = 0; s < m_S; s++) {
            int th = p * m_stride_h - m_pad_h + r * m_dilation_h;
            int tw = q * m_stride_w - m_pad_w + s * m_dilation_w;
            int xpos = n * HWC;
            int wpos = r * SKC + s * KC + k * m_C;
            int dpos = n * PQRSD + p * QRSD + q * RSD + r * SD + s * D;
            for (int co = 0; co < m_C; co += c_stride) {
                float dh = m_d[dpos];
                float dw = m_d[dpos + 1];
                int ih, iw;
                float lh, lw, hh, hw;
                prepare_interp(dh, dw, ih, iw, lh, lw, hh, hw);
                for (int ci = 0; ci < c_stride; ci++) {
                    float x = interp(xpos, th + ih, tw + iw, lh, lw, hh, hw);
                    acc += x * m_w[wpos];
                    xpos++;
                    wpos++;
                } // ci
                dpos += 2;
            } // co
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

int DeformConv2dRef::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N *  m_H * m_W * m_C;
    case 1:
        return m_N * m_P * m_Q * m_deform_groups * m_R * m_S * 2;
    case 2:
        return m_R * m_S * m_K * m_C;
    case 3:
        return m_K;
    case 4:
        return m_N * m_P * m_Q * m_K;
    default:
        assert(false);
        return 0;
    }
}

int DeformConv2dRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_K;
}

void DeformConv2dRef::prepare_interp(
        float dh,
        float dw,
        int &ih,
        int &iw,
        float &lh,
        float &lw,
        float &hh,
        float &hw) {
    float fh = std::floor(dh);
    float fw = std::floor(dw);
    ih = int(fh);
    iw = int(fw);
    lh = dh - fh;
    lw = dw - fw; 
    hh = 1.0f - lh;
    hw = 1.0f - lw;
}

float DeformConv2dRef::interp(
        int xpos,
        int th,
        int tw,
        float lh, 
        float lw,
        float hh,
        float hw) {
    const float *px = m_x + xpos;
    int WC = m_W * m_C;
    float x0 = 0.0f;
    float x1 = 0.0f;
    float x2 = 0.0f;
    float x3 = 0.0f;
    int uh = th + 1;
    int uw = tw + 1;
    if (th >= 0 && th < m_H) {
        if (tw >= 0 && tw < m_W) {
            x0 = px[th * WC + tw * m_C];
        }
        if (uw >= 0 && uw < m_W) {
            x1 = px[th * WC + uw * m_C];
        }
    }
    if (uh >= 0 && uh < m_H) {
        if (tw >= 0 && tw < m_W) {
            x2 = px[uh * WC + tw * m_C];
        }
        if (uw >= 0 && uw < m_W) {
            x3 = px[uh * WC + uw * m_C];
        }
    }
    float c0 = hh * hw;
    float c1 = hh * lw;
    float c2 = lh * hw;
    float c3 = lh * lw;
    return c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
}

} // namespace ref
} // namespace deform_conv
} // namespace op
} // namespace ronin

