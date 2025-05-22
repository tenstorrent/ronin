// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/tanto/conv2d_basic_batch.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

using conv::tanto::Conv2dBasicBatch;

class GroupConv2dBasicBatch: public Conv2dBasicBatch {
public:
    GroupConv2dBasicBatch(
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
        int batch_size);
    ~GroupConv2dBasicBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gz,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    std::vector<float> expand_weights(const std::vector<float> &x);
private:
    uint32_t m_C = 0;
    uint32_t m_K = 0;
    uint32_t m_R = 0;
    uint32_t m_S = 0;
    uint32_t m_groups = 0;
};

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

