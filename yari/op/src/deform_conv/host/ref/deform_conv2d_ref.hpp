// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host/base/post_op.hpp"

namespace ronin {
namespace op {
namespace deform_conv {
namespace ref {

namespace base = ronin::op::common::base;

class DeformConv2dRef {
public:
    DeformConv2dRef(
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
        const base::PostOpSpec &post_op);
    ~DeformConv2dRef();
public:
    void init(
        const float *x,
        const float *d,
        const float *w,
        const float *b,
        const float *z,
        float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    static void prepare_interp(
        float dh,
        float dw,
        int &ih,
        int &iw,
        float &lh,
        float &lw,
        float &hh,
        float &hw);
    float interp(
        int xpos, 
        int th,
        int tw,
        float lh,
        float lw,
        float hh,
        float hw);
private:
    const float *m_x = nullptr;
    const float *m_d = nullptr;
    const float *m_w = nullptr;
    const float *m_b = nullptr;
    const float *m_z = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_C = 0;
    int m_P = 0;
    int m_Q = 0;
    int m_K = 0;
    int m_R = 0;
    int m_S = 0;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_stride_h = 0;
    int m_stride_w = 0;
    int m_dilation_h = 0;
    int m_dilation_w = 0;
    int m_deform_groups = 0;
    base::PostOpSpec m_post_op;
};

} // namespace ref
} // namespace deform_conv
} // namespace op
} // namespace ronin

