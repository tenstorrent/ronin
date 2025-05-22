// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/util/transform.hpp"

#include "host/ref/net_ref.hpp"

#include "host/tanto/net_global.hpp"

#include "host/tanto/resnet18_mixed.hpp"

namespace ronin {
namespace nn {
namespace resnet18 {
namespace tanto {

namespace core = ronin::tanto::host;
namespace ref = common::ref;
namespace base = common::tanto;

//
//    Common private resources
//

namespace {

using ronin::op::common::base::PostOp;
using ronin::op::common::base::PostOpSpec;

PostOpSpec noop(PostOp::NONE);
PostOpSpec relu(PostOp::RELU);

void copy(float *dst, const float *src, int count) {
    memcpy(dst, src, count * sizeof(float));
}

} // namespace

//
//    NetRef private resources
//

namespace {

// Conv2dParam: H W C P Q K R S pad_h pad_w stride_h stride_w dilation_h dilation_w post_op
// FCParam: H C K
// Pool2dParam: H W C P Q R S pad_h pad_w stride_h stride_w dilation_h dilation_w
// ReduceParam: H W axis

ref::Conv2dParam conv1 = {224, 224, 3, 112, 112, 64, 7, 7, 3, 3, 2, 2, 1, 1, relu};
ref::Pool2dParam pool3 = {112, 112, 64, 56, 56, 3, 3, 1, 1, 2, 2, 1, 1};

ref::ReduceParam reduce47 = {7 * 7, 512, 1};
ref::FCParam fc49 = {1, 512, 1000};

} // namespace

//
//    ResNet18MixedTail
//

ResNet18MixedTail::ResNet18MixedTail(int N):
        NetRef(N) { }

ResNet18MixedTail::~ResNet18MixedTail() { }

void ResNet18MixedTail::init(const std::string &data_dir) {
    NetRef::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int ResNet18MixedTail::input_count() {
    return 1;
}

void ResNet18MixedTail::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 6;
    int volume = m_N * 224 * 224 * 3;
    assert(int(data.size()) == volume);
    ref::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    copy(buffer->data(), data.data(), volume);
}

int ResNet18MixedTail::output_count() {
    return 1;
}

void ResNet18MixedTail::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 11;
    int volume = m_N * 56 * 56 * 64;
    data.resize(volume);
    ref::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    copy(data.data(), buffer->data(), volume);
}

void ResNet18MixedTail::run() {
    NetRef::run();
}

void ResNet18MixedTail::init_layers() {
    init_conv2d(6, 7, 8, -1, 10, conv1);
    init_max_pool2d(10, 11, pool3);
}

void ResNet18MixedTail::load_buffers() {
    load_buffer(7, "var0007.dat");
    load_buffer(8, "var0008.dat");
}

void ResNet18MixedTail::init_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const ref::Conv2dParam &param) {
    ref::init_conv2d(
        this,
        ix, 
        iw, 
        ib, 
        iz, 
        iy, 
        param);
}

void ResNet18MixedTail::init_max_pool2d(
        int ix,
        int iy,
        const ref::Pool2dParam &param) {
    ref::init_max_pool2d(this, ix, iy, param);
}

//
//    ResNet18MixedHead
//

ResNet18MixedHead::ResNet18MixedHead(int N):
        NetRef(N) { }

ResNet18MixedHead::~ResNet18MixedHead() { }

void ResNet18MixedHead::init(const std::string &data_dir) {
    NetRef::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int ResNet18MixedHead::input_count() {
    return 1;
}

void ResNet18MixedHead::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 87;
    int volume = m_N * 7 * 7 * 512;
    assert(int(data.size()) == volume);
    ref::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    copy(buffer->data(), data.data(), volume);
}

int ResNet18MixedHead::output_count() {
    return 1;
}

void ResNet18MixedHead::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 91;
    int volume = m_N * 1000;
    data.resize(volume);
    ref::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    copy(data.data(), buffer->data(), volume);
}

void ResNet18MixedHead::run() {
    NetRef::run();
}

void ResNet18MixedHead::init_layers() {
    init_reduce_mean(87, 88, reduce47);
    init_fc(88, 0, 90, 91, fc49);
}

void ResNet18MixedHead::load_buffers() {
    load_buffer(0, "var0001.dat");
    load_buffer(90, "var0042.dat");
}

void ResNet18MixedHead::init_fc(
        int ix,
        int iw,
        int ib,
        int iy,
        const ref::FCParam &param) {
    ref::init_fc(
        this, 
        ix,
        iw,
        ib,
        iy,
        param);
}

void ResNet18MixedHead::init_reduce_mean(
        int ix,
        int iy,
        const ref::ReduceParam &param) {
    ref::init_reduce_mean(this, ix, iy, param);
}

//
//    NetGlobal private resources
//

namespace {

using namespace ronin::op::common;

int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

std::vector<uint16_t> transform_input(const std::vector<float> &x, int N) {
    std::vector<float> y = util::pad_hw(x, N, 56, 56, 64);
    return util::float_to_u16b(y);
}

std::vector<float> transform_output(const std::vector<uint16_t> &x, int N) {
    std::vector<float> y = util::u16b_to_float(x);
    return util::unpad_hw(y, N, 7, 7, 512);
}

// Conv2dParam: H W C P Q K R S pad_h pad_w stride_h stride_w dilation_h dilation_w post_op

base::Conv2dParam conv4 = {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv6 = {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv14 = {56, 56, 64, 28, 28, 128, 1, 1, 0, 0, 2, 2, 1, 1, noop};
base::Conv2dParam conv15 = {56, 56, 64, 28, 28, 128, 3, 3, 1, 1, 2, 2, 1, 1, relu};
base::Conv2dParam conv17 = {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv20 = {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv25 = {28, 28, 128, 14, 14, 256, 3, 3, 1, 1, 2, 2, 1, 1, relu};
base::Conv2dParam conv27 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, noop};
base::Conv2dParam conv28 = {28, 28, 128, 14, 14, 256, 1, 1, 0, 0, 2, 2, 1, 1, relu};
base::Conv2dParam conv31 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv33 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv36 = {14, 14, 256, 7, 7, 512, 3, 3, 1, 1, 2, 2, 1, 1, relu};
base::Conv2dParam conv38 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, noop};
base::Conv2dParam conv39 = {14, 14, 256, 7, 7, 512, 1, 1, 0, 0, 2, 2, 1, 1, relu};
base::Conv2dParam conv42 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, relu};
base::Conv2dParam conv44 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, relu};

} // namespace

//
//    ResNet18MixedMain
//

ResNet18MixedMain::ResNet18MixedMain(
        const core::Device &device, 
        int N,
        int batch_size):
            NetGlobal(device, N),
            m_batch_size(batch_size) { }

ResNet18MixedMain::~ResNet18MixedMain() { }

void ResNet18MixedMain::init(const std::string &data_dir) {
    NetGlobal::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int ResNet18MixedMain::input_count() {
    return 1;
}

void ResNet18MixedMain::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    std::vector<uint16_t> input = transform_input(data, N());
    int buffer_index = 11;
    const core::Global &global = get_buffer(buffer_index);
    assert(!global.is_null());
    assert(input.size() * sizeof(uint16_t) == global.bytes());
    core::Queue queue(device(), 0);
    queue.enqueue_write(global, input.data(), false);
}

int ResNet18MixedMain::output_count() {
    return 1;
}

void ResNet18MixedMain::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    int P = 7;
    int Q = 7;
    int K = 512;
    int PQ_rnd = round_up(P * Q, 32);
    std::vector<uint16_t> output(N() * PQ_rnd * K);
    int buffer_index = 87;
    const core::Global &global = get_buffer(buffer_index);
    assert(!global.is_null());
    assert(output.size() * sizeof(uint16_t) == global.bytes());
    core::Queue queue(device(), 0);
    // blocking read
    queue.enqueue_read(global, output.data(), true);
    data = transform_output(output, N());
}

void ResNet18MixedMain::run() {
    NetGlobal::run();
}

void ResNet18MixedMain::init_layers() {
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
}

void ResNet18MixedMain::load_buffers() {
    // SKIPPED (head): load_buffer(0, "var0001.dat");
    load_buffer(1, "var0002.dat");
    load_buffer(2, "var0003.dat");
    load_buffer(3, "var0004.dat");
    load_buffer(4, "var0005.dat");
    load_buffer(5, "var0006.dat");
    // SKIPPED (tail): load_buffer(7, "var0007.dat");
    // SKIPPED (tail): load_buffer(8, "var0008.dat");
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
    // SKIPPED (head): load_buffer(90, "var0042.dat");
}

void ResNet18MixedMain::init_conv2d(
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

//
//    ResNet18Mixed
//

ResNet18Mixed::ResNet18Mixed(
        const core::Device &device, 
        int N,
        int batch_size):
            m_tail(N),
            m_main(device, N, batch_size),
            m_head(N) { }

ResNet18Mixed::~ResNet18Mixed() { }

void ResNet18Mixed::init(const std::string &data_dir) {
    m_tail.init(data_dir);
    m_main.init(data_dir);
    m_head.init(data_dir);
}

int ResNet18Mixed::input_count() {
    return m_tail.input_count();
}

void ResNet18Mixed::set_input(int index, const std::vector<float> &data) {
    m_tail.set_input(index, data);
}

int ResNet18Mixed::output_count() {
    return m_head.output_count();
}

void ResNet18Mixed::get_output(int index, std::vector<float> &data) {
    m_head.get_output(index, data);
}

void ResNet18Mixed::run() {
    run_tail();
    run_main();
    run_head();
}

void ResNet18Mixed::run_tail() {
    std::vector<float> temp;
    m_tail.run();
    m_tail.get_output(0, temp);
    m_main.set_input(0, temp);
}

void ResNet18Mixed::run_main() {
    m_main.run();
}

void ResNet18Mixed::run_head() {
    std::vector<float> temp;
    m_main.get_output(0, temp);
    m_head.set_input(0, temp);
    m_head.run();
}

} // namespace tanto
} // namespace resnet18
} // namespace nn
} // namespace ronin

