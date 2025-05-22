// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "host/core/api.hpp"

#include "host/tanto/net_global.hpp"

namespace ronin {
namespace nn {
namespace resnet18 {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = common::tanto;

class ResNet18Global: public base::NetGlobal {
public:
    ResNet18Global(
        const core::Device &device, 
        int N,
        int batch_size);
    ~ResNet18Global();
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
    void init_max_pool2d(
        int ix,
        int iy,
        const base::Pool2dParam &param);
    void init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param);
private:
    int m_batch_size = 0;
};

} // namespace tanto
} // namespace resnet18
} // namespace nn
} // namespace ronin

