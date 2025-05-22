// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <algorithm>

#include "host/ref/interp_common.hpp"
#include "host/ref/interp2d_linear_ref.hpp"

namespace ronin {
namespace op {
namespace interp {
namespace ref {

//
//     Interp2dLinearRef
//

Interp2dLinearRef::Interp2dLinearRef(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        float scale_h,
        float scale_w,
        CoordTransformMode coord_transform_mode):
            m_N(N),
            m_H(H),
            m_W(W),
            m_C(C),
            m_P(P),
            m_Q(Q),
            m_scale_h(scale_h),
            m_scale_w(scale_w),
            m_coord_transform_mode(coord_transform_mode) { }

Interp2dLinearRef::~Interp2dLinearRef() { }

void Interp2dLinearRef::init(const float *x, float *y) {
    m_x = x;
    m_y = y;

    m_in_h1.resize(m_P);
    m_in_h2.resize(m_P);
    m_in_w1.resize(m_Q);
    m_in_w2.resize(m_Q);
    m_dh1.resize(m_P);
    m_dh2.resize(m_P);
    m_dw1.resize(m_Q);
    m_dw2.resize(m_Q);

    for (int p = 0; p < m_P; p++) {
        float in_h = get_input_coord(m_coord_transform_mode, p, m_scale_h, m_P, m_H);
        in_h = std::max(0.0f, std::min(in_h, float(m_H - 1)));
        m_in_h1[p] = std::min(int(in_h), m_H - 1);
        m_in_h2[p] = std::min(m_in_h1[p] + 1, m_H - 1);
        m_dh1[p] = std::abs(in_h - m_in_h1[p]);
        m_dh2[p] = std::abs(in_h - m_in_h2[p]);
        if (m_in_h1[p] == m_in_h2[p]) {
            m_dh1[p] = 0.5f;
            m_dh2[p] = 0.5f;
        }
    }

    for (int q = 0; q < m_Q; q++) {
        float in_w = get_input_coord(m_coord_transform_mode, q, m_scale_w, m_Q, m_W);
        in_w = std::max(0.0f, std::min(in_w, float(m_W - 1)));
        m_in_w1[q] = std::min(int(in_w), m_W - 1);
        m_in_w2[q] = std::min(m_in_w1[q] + 1, m_W - 1);
        m_dw1[q] = std::abs(in_w - m_in_w1[q]);
        m_dw2[q] = std::abs(in_w - m_in_w2[q]);
        if (m_in_w1[q] == m_in_w2[q]) {
            m_dw1[q] = 0.5f;
            m_dw2[q] = 0.5f;
        }
    }
}

void Interp2dLinearRef::run() {
    int WC = m_W * m_C;
    int HWC = m_H * WC;
    int iy = 0;
    for (int n = 0; n < m_N; n++) {
    for (int p = 0; p < m_P; p++) {
    for (int q = 0; q < m_Q; q++) {
    for (int c = 0; c < m_C; c++) {
        int ix11 = n * HWC + m_in_h1[p] * WC + m_in_w1[q] * m_C + c;
        int ix21 = n * HWC + m_in_h1[p] * WC + m_in_w2[q] * m_C + c;
        int ix12 = n * HWC + m_in_h2[p] * WC + m_in_w1[q] * m_C + c;
        int ix22 = n * HWC + m_in_h2[p] * WC + m_in_w2[q] * m_C + c;
        m_y[iy] = 
            m_dh2[p] * m_dw2[q] * m_x[ix11] + 
            m_dh1[p] * m_dw2[q] * m_x[ix21] + 
            m_dh2[p] * m_dw1[q] * m_x[ix12] + 
            m_dh1[p] * m_dw1[q] * m_x[ix22];
        iy++;            
    } // c
    } // q
    } // p
    } // n
}

int Interp2dLinearRef::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_W * m_C;
}

int Interp2dLinearRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_P * m_Q * m_C;
}

} // namespace ref
} // namespace interp
} // namespace op
} // namespace ronin

