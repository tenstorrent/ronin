// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/base/post_op.hpp"

#include "host/ref/net_ref.hpp"

#include "host/ref/mobilenetv2_050_ref.hpp"

namespace ronin {
namespace nn {
namespace mobilenetv2_050 {
namespace ref {

namespace base = common::ref;

namespace {

using ronin::op::common::base::PostOp;
using ronin::op::common::base::PostOpSpec;

using base::Conv2dParam;
using base::FCParam;
using base::GroupConv2dParam;
using base::ReduceParam;

PostOpSpec noop(PostOp::NONE);
PostOpSpec relu6(PostOp::CLIP, 0.0f, 6.0f);

// Conv2dParam: H W C P Q K R S pad_h pad_w stride_h stride_w dilation_h dilation_w post_op
// FCParam: H C K
// GroupConv2dParam: H W C P Q K R S pad_h pad_w stride_h stride_w dilation_h dilation_w groups post_op
// ReduceParam: H W axis

Conv2dParam conv1 = {224, 224, 3, 112, 112, 16, 3, 3, 1, 1, 2, 2, 1, 1, relu6};
GroupConv2dParam conv3 = {112, 112, 16, 112, 112, 16, 3, 3, 1, 1, 1, 1, 1, 1, 16, relu6};
Conv2dParam conv5 = {112, 112, 16, 112, 112, 8, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv6 = {112, 112, 8, 112, 112, 48, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv8 = {112, 112, 48, 56, 56, 48, 3, 3, 1, 1, 2, 2, 1, 1, 48, relu6};
Conv2dParam conv10 = {56, 56, 48, 56, 56, 16, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv11 = {56, 56, 16, 56, 56, 96, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv13 = {56, 56, 96, 56, 56, 96, 3, 3, 1, 1, 1, 1, 1, 1, 96, relu6};
Conv2dParam conv15 = {56, 56, 96, 56, 56, 16, 1, 1, 0, 0, 1, 1, 1, 1, noop};
GroupConv2dParam conv19 = {56, 56, 96, 28, 28, 96, 3, 3, 1, 1, 2, 2, 1, 1, 96, relu6};
Conv2dParam conv21 = {28, 28, 96, 28, 28, 16, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv22 = {28, 28, 16, 28, 28, 96, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv24 = {28, 28, 96, 28, 28, 96, 3, 3, 1, 1, 1, 1, 1, 1, 96, relu6};
Conv2dParam conv26 = {28, 28, 96, 28, 28, 16, 1, 1, 0, 0, 1, 1, 1, 1, noop};
GroupConv2dParam conv36 = {28, 28, 96, 14, 14, 96, 3, 3, 1, 1, 2, 2, 1, 1, 96, relu6};
Conv2dParam conv38 = {14, 14, 96, 14, 14, 32, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv39 = {14, 14, 32, 14, 14, 192, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv41 = {14, 14, 192, 14, 14, 192, 3, 3, 1, 1, 1, 1, 1, 1, 192, relu6};
Conv2dParam conv43 = {14, 14, 192, 14, 14, 32, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv61 = {14, 14, 192, 14, 14, 48, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv62 = {14, 14, 48, 14, 14, 288, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv64 = {14, 14, 288, 14, 14, 288, 3, 3, 1, 1, 1, 1, 1, 1, 288, relu6};
Conv2dParam conv66 = {14, 14, 288, 14, 14, 48, 1, 1, 0, 0, 1, 1, 1, 1, noop};
GroupConv2dParam conv76 = {14, 14, 288, 7, 7, 288, 3, 3, 1, 1, 2, 2, 1, 1, 288, relu6};
Conv2dParam conv78 = {7, 7, 288, 7, 7, 80, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv79 = {7, 7, 80, 7, 7, 480, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
GroupConv2dParam conv81 = {7, 7, 480, 7, 7, 480, 3, 3, 1, 1, 1, 1, 1, 1, 480, relu6};
Conv2dParam conv83 = {7, 7, 480, 7, 7, 80, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv95 = {7, 7, 480, 7, 7, 160, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv96 = {7, 7, 160, 7, 7, 1280, 1, 1, 0, 0, 1, 1, 1, 1, relu6};
ReduceParam reduce98 = {7 * 7, 1280, 1};
FCParam fc100 = {1, 1280, 1000};

} // namespace

//
//    MobileNetV2_050_Ref
//

MobileNetV2_050_Ref::MobileNetV2_050_Ref(int N):
        NetRef(N) { }

MobileNetV2_050_Ref::~MobileNetV2_050_Ref() { }

void MobileNetV2_050_Ref::init(const std::string &data_dir) {
    NetRef::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int MobileNetV2_050_Ref::input_count() {
    return 1;
}

void MobileNetV2_050_Ref::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 15;
    int volume = m_N * 224 * 224 * 3;
    assert(int(data.size()) == volume);
    base::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    memcpy(buffer->data(), data.data(), volume * sizeof(float));
}

int MobileNetV2_050_Ref::output_count() {
    return 1;
}

void MobileNetV2_050_Ref::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 206;
    int volume = m_N * 1000;
    data.resize(volume);
    base::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    memcpy(data.data(), buffer->data(), volume * sizeof(float));
}

void MobileNetV2_050_Ref::run() {
    NetRef::run();
}

void MobileNetV2_050_Ref::init_layers() {
    init_conv2d(15, 16, 17, -1, 19, conv1);
    init_group_conv2d(19, 20, 21, -1, 23, conv3);
    init_conv2d(23, 14, 24, -1, 25, conv5);
    init_conv2d(25, 13, 26, -1, 28, conv6);
    init_group_conv2d(28, 29, 30, -1, 32, conv8);
    init_conv2d(32, 12, 33, -1, 34, conv10);
    init_conv2d(34, 36, 37, -1, 39, conv11);
    init_group_conv2d(39, 40, 41, -1, 43, conv13);
    init_conv2d(43, 35, 44, 34, 46, conv15);
    init_conv2d(46, 11, 47, -1, 49, conv11);
    init_group_conv2d(49, 50, 51, -1, 53, conv19);
    init_conv2d(53, 10, 54, -1, 55, conv21);
    init_conv2d(55, 57, 58, -1, 60, conv22);
    init_group_conv2d(60, 61, 62, -1, 64, conv24);
    init_conv2d(64, 56, 65, 55, 67, conv26);
    init_conv2d(67, 69, 70, -1, 72, conv22);
    init_group_conv2d(72, 73, 74, -1, 76, conv24);
    init_conv2d(76, 68, 77, 67, 79, conv26);
    init_conv2d(79, 9, 80, -1, 82, conv22);
    init_group_conv2d(82, 83, 84, -1, 86, conv36);
    init_conv2d(86, 8, 87, -1, 88, conv38);
    init_conv2d(88, 90, 91, -1, 93, conv39);
    init_group_conv2d(93, 94, 95, -1, 97, conv41);
    init_conv2d(97, 89, 98, 88, 100, conv43);
    init_conv2d(100, 102, 103, -1, 105, conv39);
    init_group_conv2d(105, 106, 107, -1, 109, conv41);
    init_conv2d(109, 101, 110, 100, 112, conv43);
    init_conv2d(112, 114, 115, -1, 117, conv39);
    init_group_conv2d(117, 118, 119, -1, 121, conv41);
    init_conv2d(121, 113, 122, 112, 124, conv43);
    init_conv2d(124, 7, 125, -1, 127, conv39);
    init_group_conv2d(127, 128, 129, -1, 131, conv41);
    init_conv2d(131, 6, 132, -1, 133, conv61);
    init_conv2d(133, 135, 136, -1, 138, conv62);
    init_group_conv2d(138, 139, 140, -1, 142, conv64);
    init_conv2d(142, 134, 143, 133, 145, conv66);
    init_conv2d(145, 147, 148, -1, 150, conv62);
    init_group_conv2d(150, 151, 152, -1, 154, conv64);
    init_conv2d(154, 146, 155, 145, 157, conv66);
    init_conv2d(157, 5, 158, -1, 160, conv62);
    init_group_conv2d(160, 161, 162, -1, 164, conv76);
    init_conv2d(164, 4, 165, -1, 166, conv78);
    init_conv2d(166, 168, 169, -1, 171, conv79);
    init_group_conv2d(171, 172, 173, -1, 175, conv81);
    init_conv2d(175, 167, 176, 166, 178, conv83);
    init_conv2d(178, 180, 181, -1, 183, conv79);
    init_group_conv2d(183, 184, 185, -1, 187, conv81);
    init_conv2d(187, 179, 188, 178, 190, conv83);
    init_conv2d(190, 3, 191, -1, 193, conv79);
    init_group_conv2d(193, 194, 195, -1, 197, conv81);
    init_conv2d(197, 2, 198, -1, 199, conv95);
    init_conv2d(199, 1, 200, -1, 202, conv96);
    init_reduce_mean(202, 203, reduce98);
    init_fc(203, 0, 205, 206, fc100);
}

void MobileNetV2_050_Ref::load_buffers() {
    load_buffer(0, "var0001.dat");
    load_buffer(1, "var0002.dat");
    load_buffer(2, "var0003.dat");
    load_buffer(3, "var0004.dat");
    load_buffer(4, "var0005.dat");
    load_buffer(5, "var0006.dat");
    load_buffer(6, "var0007.dat");
    load_buffer(7, "var0008.dat");
    load_buffer(8, "var0009.dat");
    load_buffer(9, "var0010.dat");
    load_buffer(10, "var0011.dat");
    load_buffer(11, "var0012.dat");
    load_buffer(12, "var0013.dat");
    load_buffer(13, "var0014.dat");
    load_buffer(14, "var0015.dat");
    load_buffer(16, "var0016.dat");
    load_buffer(17, "var0017.dat");
    load_buffer(20, "var0018.dat");
    load_buffer(21, "var0019.dat");
    load_buffer(24, "var0020.dat");
    load_buffer(26, "var0021.dat");
    load_buffer(29, "var0022.dat");
    load_buffer(30, "var0023.dat");
    load_buffer(33, "var0024.dat");
    load_buffer(35, "var0025.dat");
    load_buffer(36, "var0026.dat");
    load_buffer(37, "var0027.dat");
    load_buffer(40, "var0028.dat");
    load_buffer(41, "var0029.dat");
    load_buffer(44, "var0030.dat");
    load_buffer(47, "var0031.dat");
    load_buffer(50, "var0032.dat");
    load_buffer(51, "var0033.dat");
    load_buffer(54, "var0034.dat");
    load_buffer(56, "var0035.dat");
    load_buffer(57, "var0036.dat");
    load_buffer(58, "var0037.dat");
    load_buffer(61, "var0038.dat");
    load_buffer(62, "var0039.dat");
    load_buffer(65, "var0040.dat");
    load_buffer(68, "var0041.dat");
    load_buffer(69, "var0042.dat");
    load_buffer(70, "var0043.dat");
    load_buffer(73, "var0044.dat");
    load_buffer(74, "var0045.dat");
    load_buffer(77, "var0046.dat");
    load_buffer(80, "var0047.dat");
    load_buffer(83, "var0048.dat");
    load_buffer(84, "var0049.dat");
    load_buffer(87, "var0050.dat");
    load_buffer(89, "var0051.dat");
    load_buffer(90, "var0052.dat");
    load_buffer(91, "var0053.dat");
    load_buffer(94, "var0054.dat");
    load_buffer(95, "var0055.dat");
    load_buffer(98, "var0056.dat");
    load_buffer(101, "var0057.dat");
    load_buffer(102, "var0058.dat");
    load_buffer(103, "var0059.dat");
    load_buffer(106, "var0060.dat");
    load_buffer(107, "var0061.dat");
    load_buffer(110, "var0062.dat");
    load_buffer(113, "var0063.dat");
    load_buffer(114, "var0064.dat");
    load_buffer(115, "var0065.dat");
    load_buffer(118, "var0066.dat");
    load_buffer(119, "var0067.dat");
    load_buffer(122, "var0068.dat");
    load_buffer(125, "var0069.dat");
    load_buffer(128, "var0070.dat");
    load_buffer(129, "var0071.dat");
    load_buffer(132, "var0072.dat");
    load_buffer(134, "var0073.dat");
    load_buffer(135, "var0074.dat");
    load_buffer(136, "var0075.dat");
    load_buffer(139, "var0076.dat");
    load_buffer(140, "var0077.dat");
    load_buffer(143, "var0078.dat");
    load_buffer(146, "var0079.dat");
    load_buffer(147, "var0080.dat");
    load_buffer(148, "var0081.dat");
    load_buffer(151, "var0082.dat");
    load_buffer(152, "var0083.dat");
    load_buffer(155, "var0084.dat");
    load_buffer(158, "var0085.dat");
    load_buffer(161, "var0086.dat");
    load_buffer(162, "var0087.dat");
    load_buffer(165, "var0088.dat");
    load_buffer(167, "var0089.dat");
    load_buffer(168, "var0090.dat");
    load_buffer(169, "var0091.dat");
    load_buffer(172, "var0092.dat");
    load_buffer(173, "var0093.dat");
    load_buffer(176, "var0094.dat");
    load_buffer(179, "var0095.dat");
    load_buffer(180, "var0096.dat");
    load_buffer(181, "var0097.dat");
    load_buffer(184, "var0098.dat");
    load_buffer(185, "var0099.dat");
    load_buffer(188, "var0100.dat");
    load_buffer(191, "var0101.dat");
    load_buffer(194, "var0102.dat");
    load_buffer(195, "var0103.dat");
    load_buffer(198, "var0104.dat");
    load_buffer(200, "var0105.dat");
    load_buffer(205, "var0106.dat");
}

void MobileNetV2_050_Ref::init_conv2d(
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
        param);
}

void MobileNetV2_050_Ref::init_fc(
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
        param);
}

void MobileNetV2_050_Ref::init_group_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const base::GroupConv2dParam &param) {
    base::init_group_conv2d(
        this,
        ix, 
        iw, 
        ib, 
        iz, 
        iy, 
        param);
}

void MobileNetV2_050_Ref::init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param) {
    base::init_reduce_mean(this, ix, iy, param);
}

} // namespace ref
} // namespace mobilenetv2_050
} // namespace nn
} // namespace ronin

