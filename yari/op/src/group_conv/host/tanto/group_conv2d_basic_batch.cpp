// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/tanto/conv2d_basic_batch.hpp"
#include "host/tanto/group_conv2d_basic_batch.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

using conv::tanto::Conv2dBasicBatch;

//
//    GroupConv2dBasicBatch
//

GroupConv2dBasicBatch::GroupConv2dBasicBatch(
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
        int groups,
        const base::PostOpSpec &post_op,
        int batch_size):
            Conv2dBasicBatch(
                N,
                H,
                W,
                C,
                P,
                Q,
                K,
                R,
                S,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                post_op,
                batch_size),
            m_C(uint32_t(C)),
            m_K(uint32_t(K)),
            m_R(uint32_t(R)),
            m_S(uint32_t(S)),
            m_groups(uint32_t(groups)) { }

GroupConv2dBasicBatch::~GroupConv2dBasicBatch() { }

void GroupConv2dBasicBatch::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gz,
        const core::Global &gy) {
    Conv2dBasicBatch::init(device, gx, gw, gb, gz, gy);
}

void GroupConv2dBasicBatch::run() {
    Conv2dBasicBatch::run();
}

int GroupConv2dBasicBatch::input_volume(int index) {
    return Conv2dBasicBatch::input_volume(index);
}

int GroupConv2dBasicBatch::output_volume(int index) {
    return Conv2dBasicBatch::output_volume(index);
}

std::vector<float> GroupConv2dBasicBatch::
        transform_input(int index, const std::vector<float> &x) {
    if (index == 1) {
        std::vector<float> y = expand_weights(x);
        return Conv2dBasicBatch::transform_input(index, y);
    }
    return Conv2dBasicBatch::transform_input(index, x);
}

std::vector<float> GroupConv2dBasicBatch::
        transform_output(int index, const std::vector<float> &x) {
    return Conv2dBasicBatch::transform_output(index, x);
}

std::vector<float> GroupConv2dBasicBatch::expand_weights(const std::vector<float> &x) {
    // expand weights to quasi-diagonal layout: RSGKgCg -> RSKC
    assert(m_C % m_groups == 0);
    assert(m_K % m_groups == 0);
    uint32_t Cg = m_C / m_groups;
    uint32_t Kg = m_K / m_groups;
    uint32_t RS = m_R * m_S;
    uint32_t KC = m_K * m_C;
    std::vector<float> y(RS * KC, 0.0f);
    uint32_t xpos = 0;
    for (uint32_t rs = 0; rs < RS; rs++) {
    for (uint32_t g = 0; g < m_groups; g++) {
    for (uint32_t kg = 0; kg < Kg; kg++) {
    for (uint32_t cg = 0; cg < Cg; cg++) {
        uint32_t k = g * Kg + kg;
        uint32_t c = g * Cg + cg;
        uint32_t ypos = rs * KC + k * m_C + c;
        y[ypos] = x[xpos];
        xpos++;
    } // cg
    } // kg
    } // g
    } // rs
    return y;
}

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

