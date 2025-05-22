// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <limits>
#include <algorithm>

#include "host/ref/pool2d_ref.hpp"

namespace ronin {
namespace op {
namespace pool {
namespace ref {

//
//    AvgPool2dRef
//

AvgPool2dRef::AvgPool2dRef(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w):
            m_N(N),
            m_H(H),
            m_W(W),
            m_C(C),
            m_P(P),
            m_Q(Q),
            m_R(R),
            m_S(S),
            m_pad_h(pad_h),
            m_pad_w(pad_w),
            m_stride_h(stride_h),
            m_stride_w(stride_w),
            m_dilation_h(dilation_h),
            m_dilation_w(dilation_w) { }

AvgPool2dRef::~AvgPool2dRef() { }

void AvgPool2dRef::init(const float *x, float *y) {
    m_x = x;
    m_y = y;
}

void AvgPool2dRef::run() {
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int SC = m_S * m_C;
    int RSC = m_R * SC;
    int QC = m_Q * m_C;
    int PQC = m_P * QC;

    float scale = 1.0f / float(m_R * m_S);

    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
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
            acc += m_x[xpos];
        } // s
        } // r
        int ypos = n * PQC + p * QC + q * m_C + c;
        m_y[ypos] = acc * scale;
    } // c
    } // q
    } // p
    } // n
}

int AvgPool2dRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W * m_C;
}

int AvgPool2dRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_C;
}

//
//    MaxPool2dRef
//

MaxPool2dRef::MaxPool2dRef(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w):
            m_N(N),
            m_H(H),
            m_W(W),
            m_C(C),
            m_P(P),
            m_Q(Q),
            m_R(R),
            m_S(S),
            m_pad_h(pad_h),
            m_pad_w(pad_w),
            m_stride_h(stride_h),
            m_stride_w(stride_w),
            m_dilation_h(dilation_h),
            m_dilation_w(dilation_w) { }

MaxPool2dRef::~MaxPool2dRef() { }

void MaxPool2dRef::init(const float *x, float *y) {
    m_x = x;
    m_y = y;
}

void MaxPool2dRef::run() {
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int SC = m_S * m_C;
    int RSC = m_R * SC;
    int QC = m_Q * m_C;
    int PQC = m_P * QC;

    constexpr float init_acc = std::numeric_limits<float>::lowest();

    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
    for (int c = 0; c < m_C; c++) { 
        float acc = init_acc;
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
            acc = std::max(acc, m_x[xpos]);
        } // s
        } // r
        int ypos = n * PQC + p * QC + q * m_C + c;
        m_y[ypos] = acc;
    } // c
    } // q
    } // p
    } // n
}

int MaxPool2dRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W * m_C;
}

int MaxPool2dRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_C;
}

} // namespace ref
} // namespace pool
} // namespace op
} // namespace ronin

