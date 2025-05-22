// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "host/ref/net_ref.hpp"

namespace ronin {
namespace nn {
namespace mobilenetv2_050 {
namespace ref {

namespace base = common::ref;

class MobileNetV2_050_Ref: public base::NetRef {
public:
    MobileNetV2_050_Ref(int N);
    ~MobileNetV2_050_Ref();
public:
    void init(const std::string &data_dir);
    int input_count();
    void set_input(int index, const std::vector<float> &data);
    int output_count();
    void get_output(int index, std::vector<float> &data);
    void run();
private:
    void init_layers();
    void load_buffers();
    void init_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const base::Conv2dParam &param);
    void init_fc(
        int ix,
        int iw,
        int ib,
        int iy,
        const base::FCParam &param);
    void init_group_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const base::GroupConv2dParam &param);
    void init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param);
};

} // namespace ref
} // namespace mobilenetv2_050
} // namespace nn
} // namespace ronin

