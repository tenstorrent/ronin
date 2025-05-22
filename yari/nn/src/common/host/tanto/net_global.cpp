// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "arhat/runtime/arhat.hpp"

#include "host/core/api.hpp"

#include "common/host/util/transform.hpp"
#include "host/util/diag.hpp"

#include "host/tanto/layer_base.hpp"
#include "host/tanto/layer_global.hpp"
#include "host/tanto/net_global.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

namespace core = ronin::tanto::host;

namespace {

using namespace ronin::op::common;

constexpr bool ENABLE_CONV2D_BASIC_SPATIAL = true;
constexpr bool ENABLE_GROUP_CONV2D_DW_SPATIAL = true;

core::Global null_global;

void copy(float *dst, const float *src, int count) {
    memcpy(dst, src, count * sizeof(float));
}

std::vector<float> reorder_kcrs_to_rskc(
        const std::vector<float> &x,
        int C,
        int K,
        int R, 
        int S) {
    int KC = K * C;
    int RS = R * S;
    std::vector<float> y(RS * KC);
    int iy = 0;
    for (int rs = 0; rs < RS; rs++) {
        for (int kc = 0; kc < KC; kc++) {
            int ix = kc * RS + rs;
            y[iy] = x[ix];
            iy++;
        }
    }
    return y;
}

std::vector<float> reorder_kcrs_to_krsc(
        const std::vector<float> &x,
        int C,
        int K,
        int R, 
        int S) {
    int RS = R * S;
    int CRS = C * R * S;
    std::vector<float> y(K * CRS);
    int iy = 0;
    for (int k = 0; k < K; k++) {
        for (int rs = 0; rs < RS; rs++) {
            for (int c = 0; c < C; c++) {
                int ix = k * CRS + c * RS + rs;
                y[iy] = x[ix];
                iy++;
            }
        }
    }
    return y;
}

const float *get_tensor_data(const arhat::Tensor &tensor) {
    assert(tensor.Type() == arhat::Tensor::Dtype::Float);
    return static_cast<const float *>(tensor.Data());
}

std::vector<uint16_t> tensor_to_vector(
        Layer *layer, 
        int input, 
        const arhat::Tensor &tensor) {
    int volume = tensor.Volume();
    std::vector<float> result(volume);
    copy(result.data(), get_tensor_data(tensor), volume);
    // Temporary solution to reorder convolution weights
    if (tensor.Rank() == 4) {
        // otherwise assume convoution
        //     image: reorder KCRS -> KRSC
        //     basic: reorder KCRS -> RSKC
        const int *shape = tensor.Shape();
        int K = shape[0];
        int C = shape[1];
        int R = shape[2];
        int S = shape[3];
        // may have C == 1 for DW convolutions; >=8 otherwise
        if (C == 3) {
            result = reorder_kcrs_to_krsc(result, C, K, R, S);
        } else {
            result = reorder_kcrs_to_rskc(result, C, K, R, S);
        }
    }
    result = layer->transform_input(input, result);
    return util::float_to_u16b(result);
}

} // namespace

//
//    NetGlobal
//

NetGlobal::NetGlobal(const core::Device &device, int N):
        m_device(device), m_N(N) { 
    m_buffers.reserve(16 * 1024);
    m_buffer_infos.reserve(16 * 1024);
}

NetGlobal::~NetGlobal() { }

void NetGlobal::set_data_dir(const std::string &data_dir) {
    m_data_dir = data_dir;
    size_t n = m_data_dir.size();
    if (n > 0 && (m_data_dir[n - 1] == '/' || m_data_dir[n - 1] == '\\')) {
        m_data_dir.resize(n - 1);
    }
}

void NetGlobal::run() {
    for (auto &layer: m_layers) {
        layer->run();
    }
}

void NetGlobal::init_input(
        int buffer, 
        Layer *layer, 
        int input) {
    init_buffer(buffer, layer, input, -1);
}

void NetGlobal::init_output(
        int buffer, 
        Layer *layer, 
        int output) {
    init_buffer(buffer, layer, -1, output);
}

void NetGlobal::init_buffer(
        int buffer, 
        Layer *layer, 
        int input, 
        int output) {
    if (buffer < 0) {
        return;
    }
    int size = 
        (input >= 0) ? 
            layer->input_volume(input) : 
            layer->output_volume(output);
    if (buffer >= m_buffers.size()) {
        m_buffers.resize(buffer + 1);
        m_buffer_infos.resize(buffer + 1);
    }
    if (!m_buffers[buffer].is_null()) {
        // throw exception instead?
        // using >= to support pre-allocated reused buffers
        assert(m_buffers[buffer].size() >= uint32_t(size));
    } else {
        uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
        m_buffers[buffer] = core::Global(m_device, T, uint32_t(size), log2_page_size);
        BufferInfo &info = m_buffer_infos[buffer];
        info.layer = layer;
        info.input = input;
        info.output = output;
    }
}

void NetGlobal::load_buffer(int index, const std::string &fn) {
    core::Global &global = m_buffers[index];
    BufferInfo &info = m_buffer_infos[index];
    assert(!global.is_null());
    assert(info.layer != nullptr && info.input >= 0);
    std::string path = m_data_dir + "/" + fn;
    arhat::Tensor tensor;
    tensor.Read(path);
    std::vector<uint16_t> buffer = tensor_to_vector(info.layer, info.input, tensor);
    assert(buffer.size() * sizeof(uint16_t) == global.bytes());
    core::Queue queue(m_device, 0);
    queue.enqueue_write(global, buffer.data(), false);
}

const core::Global &NetGlobal::get_buffer(int index) {
    return (index >= 0) ? m_buffers[index] : null_global;
}

void NetGlobal::add_layer(std::unique_ptr<Layer> &&layer) {
    m_layers.push_back(std::move(layer));
}

int NetGlobal::layer_count() {
    return int(m_layers.size());
}

Layer *NetGlobal::layer_at(int index) {
    return m_layers[index].get();
}

std::vector<float> NetGlobal::read_buffer(int index) {
    // introduced mainly for diagnostics
    // not recommended for regular use
    core::Global &global = m_buffers[index];
    assert(!global.is_null());
    BufferInfo &info = m_buffer_infos[index];
    Layer *layer = info.layer;
    int output = info.output;
    assert(layer != nullptr);
    assert(output >= 0);
    int volume = layer->output_volume(output);
    std::vector<uint16_t> data(volume);
    core::Queue queue(m_device, 0);
    // blocking read
    queue.enqueue_read(global, data.data(), true);
    std::vector<float> temp = util::u16b_to_float(data);
    return layer->transform_output(output, temp);
}

std::vector<float> NetGlobal::read_buffer_raw(int index) {
    // introduced mainly for diagnostics
    // not recommended for regular use
    core::Global &global = m_buffers[index];
    assert(!global.is_null());
    BufferInfo &info = m_buffer_infos[index];
    Layer *layer = info.layer;
    int output = info.output;
    assert(layer != nullptr);
    assert(output >= 0);
    int volume = layer->output_volume(output);
    std::vector<uint16_t> data(volume);
    core::Queue queue(m_device, 0);
    // blocking read
    queue.enqueue_read(global, data.data(), true);
    return util::u16b_to_float(data);
}

std::string NetGlobal::diag_buffer_stats(int index) {
    std::vector<float> data = read_buffer(index);
    // Temporary solution: restrict to first batch item, assume batch size 16
    data.resize(data.size() / 16);
    return ronin::op::common::util::diag_data_stats(data);
}

//
//    Binary
//

namespace {

template<typename LAYER>
void init_binary(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ia, layer, 0);
    net->init_input(ib, layer, 1);
    net->init_output(ic, layer, 0);
    layer->init(
        net->device(),
        net->get_buffer(ia), 
        net->get_buffer(ib), 
        net->get_buffer(ic));
}

} // namespace

void init_add(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_add_batch(net, ia, ib, ic, param, batch_size);
}

void init_add_batch(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_binary<AddBatchLayer>(net, ia, ib, ic, param, batch_size);
}

void init_sub(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_sub_batch(net, ia, ib, ic, param, batch_size);
}

void init_sub_batch(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_binary<SubBatchLayer>(net, ia, ib, ic, param, batch_size);
}

void init_mul(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_mul_batch(net, ia, ib, ic, param, batch_size);
}

void init_mul_batch(
        NetGlobal *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param,
        int batch_size) {
    init_binary<MulBatchLayer>(net, ia, ib, ic, param, batch_size);
}

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
        int batch_size) {
    // placeholder for generic version with
    // automated implementation choice based on param
    if (param.C <= 4) {
        init_conv2d_image_batch(
            net,
            ix,
            iw,
            ib,
            iz,
            iy,
            param,
            batch_size);
    } else if (ENABLE_CONV2D_BASIC_SPATIAL && batch_size == 1) {
        // EXPERIMENTAL
        init_conv2d_basic_spatial(
            net,
            ix,
            iw,
            ib,
            iz,
            iy,
            param,
            batch_size);
    } else {
        init_conv2d_basic_batch(
            net,
            ix,
            iw,
            ib,
            iz,
            iy,
            param,
            batch_size);
    }
}

void init_conv2d_basic_batch(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const Conv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<Conv2dBasicBatchLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

void init_conv2d_basic_spatial(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const Conv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<Conv2dBasicSpatialLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

void init_conv2d_image_batch(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const Conv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<Conv2dImageBatchLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

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
        int batch_size) {
    init_fc_batch(
        net,
        ix,
        iw,
        ib,
        iy,
        param,
        batch_size);
}

void init_fc_batch(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iy,
        const FCParam &param,
        int batch_size) {
    auto layer_unique = std::make_unique<FCBatchLayer>(net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(),
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iy));
}

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
        int batch_size) {
    if (param.C == param.groups && param.K == param.groups) {
        if (ENABLE_GROUP_CONV2D_DW_SPATIAL && batch_size == 1) {
            // EXPERIMENTAL
            init_group_conv2d_dw_spatial(
                net,
                ix,
                iw,
                ib,
                iz,
                iy,
                param,
                batch_size);
        } else {
            init_group_conv2d_dw_batch(
                net,
                ix,
                iw,
                ib,
                iz,
                iy,
                param,
                batch_size);
        }
    } else {
        init_group_conv2d_basic_batch(
            net,
            ix,
            iw,
            ib,
            iz,
            iy,
            param,
            batch_size);
    }
}

void init_group_conv2d_basic_batch(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const GroupConv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<GroupConv2dBasicBatchLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

void init_group_conv2d_dw_batch(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const GroupConv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<GroupConv2dDwBatchLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}


void init_group_conv2d_dw_spatial(
        NetGlobal *net,
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const GroupConv2dParam &param,
        int batch_size) {
    auto layer_unique = 
        std::make_unique<GroupConv2dDwSpatialLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

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
        int batch_size) {
    auto layer_unique = 
        std::make_unique<DSConv2dBatchLayer>(
            net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iw2, layer, 3);
    net->init_input(ib2, layer, 4);
    net->init_input(iz, layer, 5);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(), 
        net->get_buffer(ix), 
        net->get_buffer(iw), 
        net->get_buffer(ib), 
        net->get_buffer(iw2), 
        net->get_buffer(ib2), 
        net->get_buffer(iz), 
        net->get_buffer(iy));
}

//
//    Pool2d
//

namespace {

template<typename LAYER>
void init_pool2d(
        NetGlobal *net,
        int ix,
        int iy,
        const Pool2dParam &param,
        int batch_size) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(),
        net->get_buffer(ix), 
        net->get_buffer(iy));
}

} // namespace

void init_avg_pool2d(
        NetGlobal *net,
        int ix,
        int iy,
        const Pool2dParam &param,
        int batch_size) {
    init_avg_pool2d_batch(net, ix, iy, param, batch_size);
}

void init_avg_pool2d_batch(
        NetGlobal *net,
        int ix,
        int iy,
        const Pool2dParam &param,
        int batch_size) {
    init_pool2d<AvgPool2dBatchLayer>(net, ix, iy, param, batch_size);
}

void init_max_pool2d(
        NetGlobal *net,
        int ix,
        int iy,
        const Pool2dParam &param,
        int batch_size) {
    init_max_pool2d_batch(net, ix, iy, param, batch_size);
}

void init_max_pool2d_batch(
        NetGlobal *net,
        int ix,
        int iy,
        const Pool2dParam &param,
        int batch_size) {
    init_pool2d<MaxPool2dBatchLayer>(net, ix, iy, param, batch_size);
}

//
//    Reduce
//

namespace {

template<typename LAYER>
void init_reduce(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param, batch_size);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_output(iy, layer, 0);
    layer->init(
        net->device(),
        net->get_buffer(ix), 
        net->get_buffer(iy));
}

} // namespace

void init_reduce_max(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce_max_batch(net, ix, iy, param, batch_size);
}

void init_reduce_max_batch(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce<ReduceMaxBatchLayer>(net, ix, iy, param, batch_size);
}

void init_reduce_mean(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce_mean_batch(net, ix, iy, param, batch_size);
}

void init_reduce_mean_batch(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce<ReduceMeanBatchLayer>(net, ix, iy, param, batch_size);
}

void init_reduce_sum(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce_sum_batch(net, ix, iy, param, batch_size);
}

void init_reduce_sum_batch(
        NetGlobal *net,
        int ix,
        int iy,
        const ReduceParam &param,
        int batch_size) {
    init_reduce<ReduceSumBatchLayer>(net, ix, iy, param, batch_size);
}

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

