// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/util/transform.hpp"

#include "host/base/post_op.hpp"

#include "host/tanto/net_global.hpp"

#include "host/tanto/resnet18_global.hpp"

namespace ronin {
namespace nn {
namespace resnet18 {
namespace tanto {

using ronin::op::common::base::PostOp;
using ronin::op::common::base::PostOpSpec;

namespace core = ronin::tanto::host;
namespace base = common::tanto;

using namespace ronin::op::common;

namespace {

using base::Conv2dParam;
using base::FCParam;
using base::Pool2dParam;
using base::ReduceParam;

PostOpSpec noop(PostOp::NONE);
PostOpSpec relu(PostOp::RELU);

// Conv2dParam: H W C P Q K R S pad_h pad_w stride_h stride_w dilation_h dilation_w post_op
// FCParam: H C K
// Pool2dParam: H W C P Q R S pad_h pad_w stride_h stride_w dilation_h dilation_w
// ReduceParam: H W axis

Conv2dParam conv1 = {224, 224, 3, 112, 112, 64, 7, 7, 3, 3, 2, 2, 1, 1, relu};
Pool2dParam pool3 = {112, 112, 64, 56, 56, 3, 3, 1, 1, 2, 2, 1, 1};
Conv2dParam conv4 = {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv6 = {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv14 = {56, 56, 64, 28, 28, 128, 1, 1, 0, 0, 2, 2, 1, 1, noop};
Conv2dParam conv15 = {56, 56, 64, 28, 28, 128, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv17 = {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv20 = {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv25 = {28, 28, 128, 14, 14, 256, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv27 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, noop};
Conv2dParam conv28 = {28, 28, 128, 14, 14, 256, 1, 1, 0, 0, 2, 2, 1, 1, relu};
Conv2dParam conv31 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv33 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv36 = {14, 14, 256, 7, 7, 512, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv38 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, noop};
Conv2dParam conv39 = {14, 14, 256, 7, 7, 512, 1, 1, 0, 0, 2, 2, 1, 1, relu};
Conv2dParam conv42 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv44 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, relu};
ReduceParam reduce47 = {7 * 7, 512, 1};
FCParam fc49 = {1, 512, 1000};

} // namespace

//
//    ResNet18Global
//

ResNet18Global::ResNet18Global(
        const core::Device &device, 
        int N,
        int batch_size):
            NetGlobal(device, N),
            m_batch_size(batch_size) { }

ResNet18Global::~ResNet18Global() { }

void ResNet18Global::init(const std::string &data_dir) {
    NetGlobal::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int ResNet18Global::input_count() {
    return 1;
}

void ResNet18Global::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    base::Layer *bottom = layer_at(0);
    std::vector<float> temp = bottom->transform_input(0, data);
    std::vector<uint16_t> input = util::float_to_u16b(temp);
    int buffer_index = 6;
    const core::Global &global = get_buffer(buffer_index);
    assert(!global.is_null());
    assert(input.size() * sizeof(uint16_t) == global.bytes());
    core::Queue queue(device(), 0);
    queue.enqueue_write(global, input.data(), false);
}

int ResNet18Global::output_count() {
    return 1;
}

void ResNet18Global::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    base::Layer *top = layer_at(layer_count() - 1);
    int volume = top->output_volume(0);
    std::vector<uint16_t> output(volume);
    int buffer_index = 91;
    const core::Global &global = get_buffer(buffer_index);
    assert(!global.is_null());
    assert(output.size() * sizeof(uint16_t) == global.bytes());
    core::Queue queue(device(), 0);
    // blocking read
    queue.enqueue_read(global, output.data(), true);
    std::vector<float> temp = util::u16b_to_float(output);
    data = top->transform_output(0, temp);
}

void ResNet18Global::run() {
    NetGlobal::run();
}

void ResNet18Global::init_layers() {
    init_conv2d(6, 7, 8, -1, 10, conv1);
    init_max_pool2d(10, 11, pool3);
    init_conv2d(11, 13, 14, -1, 16, conv4);
    init_conv2d(16, 12, 17, 11, 20, conv6);
    init_conv2d(20, 22, 23, -1, 25, conv4);
    init_conv2d(25, 21, 26, 20, 29, conv6);
    init_conv2d(29, 5, 30, -1, 31, conv14);
    init_conv2d(29, 33, 34, -1, 36, conv15);
    init_conv2d(36, 32, 37, 31, 40, conv17);
    init_conv2d(40, 42, 43, -1, 45, conv20);
    init_conv2d(45, 41, 46, 40, 49, conv17);
    init_conv2d(49, 4, 50, -1, 52, conv25);
    init_conv2d(52, 3, 53, -1, 54, conv27);
    init_conv2d(49, 55, 56, 54, 59, conv28);
    init_conv2d(59, 61, 62, -1, 64, conv31);
    init_conv2d(64, 60, 65, 59, 68, conv33);
    init_conv2d(68, 2, 69, -1, 71, conv36);
    init_conv2d(71, 1, 72, -1, 73, conv38);
    init_conv2d(68, 74, 75, 73, 78, conv39);
    init_conv2d(78, 80, 81, -1, 83, conv42);
    init_conv2d(83, 79, 84, 78, 87, conv44);
    init_reduce_mean(87, 88, reduce47);
    init_fc(88, 0, 90, 91, fc49);
}

void ResNet18Global::load_buffers() {
    load_buffer(0, "var0001.dat");
    load_buffer(1, "var0002.dat");
    load_buffer(2, "var0003.dat");
    load_buffer(3, "var0004.dat");
    load_buffer(4, "var0005.dat");
    load_buffer(5, "var0006.dat");
    load_buffer(7, "var0007.dat");
    load_buffer(8, "var0008.dat");
    load_buffer(12, "var0009.dat");
    load_buffer(13, "var0010.dat");
    load_buffer(14, "var0011.dat");
    load_buffer(17, "var0012.dat");
    load_buffer(21, "var0013.dat");
    load_buffer(22, "var0014.dat");
    load_buffer(23, "var0015.dat");
    load_buffer(26, "var0016.dat");
    load_buffer(30, "var0017.dat");
    load_buffer(32, "var0018.dat");
    load_buffer(33, "var0019.dat");
    load_buffer(34, "var0020.dat");
    load_buffer(37, "var0021.dat");
    load_buffer(41, "var0022.dat");
    load_buffer(42, "var0023.dat");
    load_buffer(43, "var0024.dat");
    load_buffer(46, "var0025.dat");
    load_buffer(50, "var0026.dat");
    load_buffer(53, "var0027.dat");
    load_buffer(55, "var0028.dat");
    load_buffer(56, "var0029.dat");
    load_buffer(60, "var0030.dat");
    load_buffer(61, "var0031.dat");
    load_buffer(62, "var0032.dat");
    load_buffer(65, "var0033.dat");
    load_buffer(69, "var0034.dat");
    load_buffer(72, "var0035.dat");
    load_buffer(74, "var0036.dat");
    load_buffer(75, "var0037.dat");
    load_buffer(79, "var0038.dat");
    load_buffer(80, "var0039.dat");
    load_buffer(81, "var0040.dat");
    load_buffer(84, "var0041.dat");
    load_buffer(90, "var0042.dat");
}

void ResNet18Global::init_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const base::Conv2dParam &param) {
    base::init_conv2d(
        this,
        ix, 
        iw, 
        ib, 
        iz, 
        iy, 
        param, 
        m_batch_size);
}

void ResNet18Global::init_fc(
        int ix,
        int iw,
        int ib,
        int iy,
        const base::FCParam &param) {
    base::init_fc(
        this, 
        ix,
        iw,
        ib,
        iy,
        param, 
        m_batch_size);
}

void ResNet18Global::init_max_pool2d(
        int ix,
        int iy,
        const base::Pool2dParam &param) {
    base::init_max_pool2d(this, ix, iy, param, m_batch_size);
}

void ResNet18Global::init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param) {
    base::init_reduce_mean(this, ix, iy, param, m_batch_size);
}

} // namespace tanto
} // namespace resnet18
} // namespace nn
} // namespace ronin

