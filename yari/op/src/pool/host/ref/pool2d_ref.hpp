// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace op {
namespace pool {
namespace ref {

class AvgPool2dRef {
public:
    AvgPool2dRef(
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
        int dilation_w);
    ~AvgPool2dRef();
public:
    void init(const float *x, float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_C = 0;
    int m_P = 0;
    int m_Q = 0;
    int m_R = 0;
    int m_S = 0;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_stride_h = 0;
    int m_stride_w = 0;
    int m_dilation_h = 0;
    int m_dilation_w = 0;
};

class MaxPool2dRef {
public:
    MaxPool2dRef(
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
        int dilation_w);
    ~MaxPool2dRef();
public:
    void init(const float *x, float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_C = 0;
    int m_P = 0;
    int m_Q = 0;
    int m_R = 0;
    int m_S = 0;
    int m_pad_h = 0;
    int m_pad_w = 0;
    int m_stride_h = 0;
    int m_stride_w = 0;
    int m_dilation_h = 0;
    int m_dilation_w = 0;
};

} // namespace ref
} // namespace pool
} // namespace op
} // namespace ronin

