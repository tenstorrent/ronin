// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "host/ref/layer_base.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace ref {

//
//    NetRef
//

class NetRef {
public:
    NetRef(int N);
    ~NetRef();
public:
    void set_data_dir(const std::string &data_dir);
    void run();
    int N() {
        return m_N;
    }
    void init_input(int buffer, Layer *layer, int input); 
    void init_output(int buffer, Layer *layer, int output); 
    void init_buffer(int index, int volume);
    void load_buffer(int index, const std::string &fn);
    Buffer *get_buffer(int index);
    float *buffer_data(int index);
    void add_layer(std::unique_ptr<Layer> &&layer);
    std::vector<float> read_buffer(int index);
    std::string diag_buffer_stats(int index);
protected:
    int m_N = 0;
    std::string m_data_dir;
    std::vector<std::unique_ptr<Buffer>> m_buffers;
    std::vector<std::unique_ptr<Layer>> m_layers;
};

//
//    Binary
//

void init_add(
    NetRef *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param);
void init_sub(
    NetRef *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param);
void init_mul(
    NetRef *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param);

//
//    Conv2d
//

void init_conv2d(
    NetRef *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param);

//
//    FC
//

void init_fc(
    NetRef *net,
    int ix,
    int iw,
    int ib,
    int iy,
    const FCParam &param);

//
//    GroupConv2d
//

void init_group_conv2d(
    NetRef *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const GroupConv2dParam &param);

//
//    Pool2d
//

void init_avg_pool2d(
    NetRef *net,
    int ix,
    int iy,
    const Pool2dParam &param);
void init_max_pool2d(
    NetRef *net,
    int ix,
    int iy,
    const Pool2dParam &param);

//
//    Reduce
//

void init_reduce_max(
    NetRef *net,
    int ix,
    int iy,
    const ReduceParam &param);
void init_reduce_mean(
    NetRef *net,
    int ix,
    int iy,
    const ReduceParam &param);
void init_reduce_sum(
    NetRef *net,
    int ix,
    int iy,
    const ReduceParam &param);

} // namespace ref
} // namespace common
} // namespace nn
} // namespace ronin

