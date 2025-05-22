// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "host/ref/interp_common.hpp"

namespace ronin {
namespace op {
namespace interp {
namespace ref {

class Interp2dLinearRef {
public:
    Interp2dLinearRef(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        float scale_h,
        float scale_w,
        CoordTransformMode coord_transform_mode);
    ~Interp2dLinearRef();
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
    float m_scale_h = 0.0f;
    float m_scale_w = 0.0f;
    CoordTransformMode m_coord_transform_mode = CoordTransformMode(0);
    std::vector<int> m_in_h1;
    std::vector<int> m_in_h2;
    std::vector<int> m_in_w1;
    std::vector<int> m_in_w2;
    std::vector<float> m_dh1;
    std::vector<float> m_dh2;
    std::vector<float> m_dw1;
    std::vector<float> m_dw2;
};

} // namespace ref
} // namespace interp
} // namespace op
} // namespace ronin

