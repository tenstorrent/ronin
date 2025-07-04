// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "host/core/api.hpp"

#include "host/tanto/layer_base.hpp"
#include "host/tanto/conv2d_perf_db.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

namespace core = ronin::tanto::host;

//
//    NetGlobal
//

class NetGlobal {
public:
    NetGlobal(const core::Device &device, int N);
    ~NetGlobal();
public:
    void set_data_dir(const std::string &data_dir);
    void run();
    Conv2dPerfDb &conv2d_perf_db() {
        return m_conv2d_perf_db;
    }
    const core::Device &device() {
        return m_device;
    }
    int N() {
        return m_N;
    }
    void init_input(
        int buffer, 
        Layer *layer, 
        int input); 
    void init_output(
        int buffer, 
        Layer *layer, 
        int output);
    void init_buffer(
        int buffer, 
        Layer *layer, 
        int input, 
        int output);
    void load_buffer(int index, const std::string &fn);
    const core::Global &get_buffer(int index);
    void add_layer(std::unique_ptr<Layer> &&layer);
    int layer_count();
    Layer *layer_at(int index);
    std::vector<float> read_buffer(int index);
    std::vector<float> read_buffer_raw(int index);
    std::string diag_buffer_stats(int index);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
protected:
    struct BufferInfo {
        Layer *layer = nullptr;
        int input = 0;
        int output = 0;
    };
protected:
    Conv2dPerfDb m_conv2d_perf_db;
    core::Device m_device;
    int m_N = 0;
    std::string m_data_dir;
    std::vector<core::Global> m_buffers;
    std::vector<BufferInfo> m_buffer_infos;
    std::vector<std::unique_ptr<Layer>> m_layers;
};

//
//    Binary
//

void init_add(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);
void init_add_batch(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);

void init_sub(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);
void init_sub_batch(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);

void init_mul(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);
void init_mul_batch(
    NetGlobal *net,
    int ia,
    int ib,
    int ic,
    const BinaryParam &param,
    int batch_size);

//
//    Conv2d
//

void init_conv2d(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param,
    int batch_size);
void init_conv2d_basic_batch(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param,
    int batch_size);
void init_conv2d_basic_split(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param,
    int batch_size);
void init_conv2d_basic_spatial(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param,
    int batch_size);
void init_conv2d_image_batch(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const Conv2dParam &param,
    int batch_size);

//
//    FC
//

void init_fc(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iy,
    const FCParam &param,
    int batch_size);
void init_fc_batch(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iy,
    const FCParam &param,
    int batch_size);

//
//    GroupConv2d
//

void init_group_conv2d(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const GroupConv2dParam &param,
    int batch_size);
void init_group_conv2d_basic_batch(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const GroupConv2dParam &param,
    int batch_size);
void init_group_conv2d_dw_batch(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const GroupConv2dParam &param,
    int batch_size);
void init_group_conv2d_dw_spatial(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iz,
    int iy,
    const GroupConv2dParam &param,
    int batch_size);

void init_ds_conv2d(
    NetGlobal *net,
    int ix,
    int iw,
    int ib,
    int iw2,
    int ib2,
    int iz,
    int iy,
    const DSConv2dParam &param,
    int batch_size);

//
//    Pool2d
//

void init_avg_pool2d(
    NetGlobal *net,
    int ix,
    int iy,
    const Pool2dParam &param,
    int batch_size);
void init_avg_pool2d_batch(
    NetGlobal *net,
    int ix,
    int iy,
    const Pool2dParam &param,
    int batch_size);

void init_max_pool2d(
    NetGlobal *net,
    int ix,
    int iy,
    const Pool2dParam &param,
    int batch_size);
void init_max_pool2d_batch(
    NetGlobal *net,
    int ix,
    int iy,
    const Pool2dParam &param,
    int batch_size);

//
//    Reduce
//

void init_reduce_max(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);
void init_reduce_max_batch(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);

void init_reduce_mean(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);
void init_reduce_mean_batch(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);

void init_reduce_sum(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);
void init_reduce_sum_batch(
    NetGlobal *net,
    int ix,
    int iy,
    const ReduceParam &param,
    int batch_size);

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

