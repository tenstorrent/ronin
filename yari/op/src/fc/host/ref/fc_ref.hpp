// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace op {
namespace fc {
namespace ref {

class FCRef {
public:
    FCRef(
        int N,
        int H,
        int C,
        int K);
    ~FCRef();
public:
    void init(
        const float *x,
        const float *w,
        const float *b,
        float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    const float *m_w = nullptr;
    const float *m_b = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_C = 0;
    int m_K = 0;
};

} // namespace ref
} // namespace fc
} // namespace op
} // namespace ronin

