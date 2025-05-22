// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "arhat/runtime/arhat.hpp"

#include "host/util/diag.hpp"

#include "host/ref/layer_base.hpp"
#include "host/ref/layer_ref.hpp"
#include "host/ref/net_ref.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace ref {

namespace {

void tensor_to_buffer(const arhat::Tensor &tensor, Buffer *buffer) {
    // Temporary solution to reorder convolution weights
    int volume = buffer->volume();
    assert(tensor.Volume() == volume);
    if (tensor.Rank() != 4) {
        memcpy(buffer->data(), tensor.Data(), volume * sizeof(float));
    } else {
        // otherwise assume convoution, reorder KCRS -> RSKC
        const int *shape = tensor.Shape();
        int KC = shape[0] * shape[1];
        int RS = shape[2] * shape[3];
        assert(tensor.Type() == arhat::Tensor::Dtype::Float);
        const float *tensor_data = static_cast<const float *>(tensor.Data());
        float *buffer_data = buffer->data();
        int buffer_index = 0;
        for (int rs = 0; rs < RS; rs++) {
            for (int kc = 0; kc < KC; kc++) {
                int tensor_index = kc * RS + rs;
                buffer_data[buffer_index] = tensor_data[tensor_index];
                buffer_index++;
            }
        }
    }
}

} // namespace

//
//    NetRef
//

NetRef::NetRef(int N): m_N(N) { 
    m_buffers.reserve(16 * 1024);
}

NetRef::~NetRef() { }

void NetRef::set_data_dir(const std::string &data_dir) {
    m_data_dir = data_dir;
    size_t n = m_data_dir.size();
    if (n > 0 && (m_data_dir[n - 1] == '/' || m_data_dir[n - 1] == '\\')) {
        m_data_dir.resize(n - 1);
    }
}

void NetRef::run() {
    for (auto &layer: m_layers) {
        layer->run();
    }
}

void NetRef::init_input(int buffer, Layer *layer, int input) {
    if (buffer >= 0) {
        init_buffer(buffer, layer->input_volume(input));
    }
}

void NetRef::init_output(int buffer, Layer *layer, int output) {
    if (buffer >= 0) {
        init_buffer(buffer, layer->output_volume(output));
    }
}

void NetRef::init_buffer(int index, int volume) {
    if (index >= m_buffers.size()) {
        m_buffers.resize(index + 1);
    }
    if (m_buffers[index] != nullptr) {
        // throw exception instead?
        // using >= to support pre-allocated reused buffers
        assert(m_buffers[index]->volume() >= volume);
    } else {
        m_buffers[index] = std::make_unique<Buffer>(volume);
    }
}

void NetRef::load_buffer(int index, const std::string &fn) {
    Buffer *buffer = m_buffers[index].get();
    assert(buffer != nullptr);
    std::string path = m_data_dir + "/" + fn;
    arhat::Tensor tensor;
    tensor.Read(path);
    tensor_to_buffer(tensor, buffer);
}

Buffer *NetRef::get_buffer(int index) {
    return (index >= 0) ? m_buffers[index].get() : nullptr;
}

float *NetRef::buffer_data(int index) {
    Buffer *buf = get_buffer(index);
    return (buf != nullptr) ? buf->data() : nullptr;
}

void NetRef::add_layer(std::unique_ptr<Layer> &&layer) {
    m_layers.push_back(std::move(layer));
}

std::vector<float> NetRef::read_buffer(int index) {
    Buffer *buffer = m_buffers[index].get();
    assert(buffer != nullptr);
    int volume = buffer->volume();
    std::vector<float> result(volume);
    memcpy(result.data(), buffer->data(), volume * sizeof(float));
    return result;
}

std::string NetRef::diag_buffer_stats(int index) {
    std::vector<float> data = read_buffer(index);
    return ronin::op::common::util::diag_data_stats(data);
}

//
//    Binary
//

namespace {

template<typename LAYER>
void init_binary(
        NetRef *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ia, layer, 0);
    net->init_input(ib, layer, 1);
    net->init_output(ic, layer, 0);
    layer->init(
        net->buffer_data(ia), 
        net->buffer_data(ib), 
        net->buffer_data(ic));
}

} // namespace

void init_add(
        NetRef *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param) {
    init_binary<AddRefLayer>(net, ia, ib, ic, param);
}

void init_sub(
        NetRef *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param) {
    init_binary<SubRefLayer>(net, ia, ib, ic, param);
}

void init_mul(
        NetRef *net,
        int ia,
        int ib,
        int ic,
        const BinaryParam &param) {
    init_binary<MulRefLayer>(net, ia, ib, ic, param);
}

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
        const Conv2dParam &param) {
    auto layer_unique = std::make_unique<Conv2dRefLayer>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->buffer_data(ix), 
        net->buffer_data(iw), 
        net->buffer_data(ib), 
        net->buffer_data(iz), 
        net->buffer_data(iy));
}

//
//    FC
//

void init_fc(
        NetRef *net,
        int ix,
        int iw,
        int ib,
        int iy,
        const FCParam &param) {
    auto layer_unique = std::make_unique<FCRefLayer>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_output(iy, layer, 0);
    layer->init(
        net->buffer_data(ix), 
        net->buffer_data(iw), 
        net->buffer_data(ib), 
        net->buffer_data(iy));
}

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
        const GroupConv2dParam &param) {
    auto layer_unique = std::make_unique<GroupConv2dRefLayer>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_input(iw, layer, 1);
    net->init_input(ib, layer, 2);
    net->init_input(iz, layer, 3);
    net->init_output(iy, layer, 0);
    layer->init(
        net->buffer_data(ix), 
        net->buffer_data(iw), 
        net->buffer_data(ib), 
        net->buffer_data(iz), 
        net->buffer_data(iy));
}

//
//    Pool2d
//

namespace {

template<typename LAYER>
void init_pool2d(
        NetRef *net,
        int ix,
        int iy,
        const Pool2dParam &param) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_output(iy, layer, 0);
    layer->init(
        net->buffer_data(ix), 
        net->buffer_data(iy));
}

} // namespace

void init_avg_pool2d(
        NetRef *net,
        int ix,
        int iy,
        const Pool2dParam &param) {
    init_pool2d<AvgPool2dRefLayer>(net, ix, iy, param);
}

void init_max_pool2d(
        NetRef *net,
        int ix,
        int iy,
        const Pool2dParam &param) {
    init_pool2d<MaxPool2dRefLayer>(net, ix, iy, param);
}

//
//    Reduce
//

namespace {

template<typename LAYER>
void init_reduce(
        NetRef *net,
        int ix,
        int iy,
        const ReduceParam &param) {
    auto layer_unique = std::make_unique<LAYER>(net->N(), param);
    auto layer = layer_unique.get();
    net->add_layer(std::move(layer_unique));
    net->init_input(ix, layer, 0);
    net->init_output(iy, layer, 0);
    layer->init(
        net->buffer_data(ix), 
        net->buffer_data(iy));
}

} // namespace

void init_reduce_max(
        NetRef *net,
        int ix,
        int iy,
        const ReduceParam &param) {
    init_reduce<ReduceMaxRefLayer>(net, ix, iy, param);
}

void init_reduce_mean(
        NetRef *net,
        int ix,
        int iy,
        const ReduceParam &param) {
    init_reduce<ReduceMeanRefLayer>(net, ix, iy, param);
}

void init_reduce_sum(
        NetRef *net,
        int ix,
        int iy,
        const ReduceParam &param) {
    init_reduce<ReduceSumRefLayer>(net, ix, iy, param);
}

} // namespace ref
} // namespace common
} // namespace nn
} // namespace ronin

