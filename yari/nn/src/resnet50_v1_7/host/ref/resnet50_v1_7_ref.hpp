// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "host/ref/net_ref.hpp"

namespace ronin {
namespace nn {
namespace resnet50_v1_7 {
namespace ref {

namespace base = common::ref;

class ResNet50V17Ref: public base::NetRef {
public:
    ResNet50V17Ref(int N);
    ~ResNet50V17Ref();
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
    void init_add(
        int ia,
        int ib,
        int ic,
        const base::BinaryParam &param);
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
    void init_max_pool2d(
        int ix,
        int iy,
        const base::Pool2dParam &param);
    void init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param);
};

} // namespace ref
} // namespace resnet50_v1_7
} // namespace nn
} // namespace ronin

