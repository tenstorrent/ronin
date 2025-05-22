// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/base/post_op.hpp"

#include "host/ref/net_ref.hpp"

#include "host/ref/resnet50_v1_7_ref.hpp"

namespace ronin {
namespace nn {
namespace resnet50_v1_7 {
namespace ref {

namespace base = common::ref;

namespace {

using ronin::op::common::base::PostOp;
using ronin::op::common::base::PostOpSpec;

using base::BinaryParam;
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
Conv2dParam conv4 = {56, 56, 64, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv6 = {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv8 = {56, 56, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv9 = {56, 56, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv12 = {56, 56, 256, 56, 56, 64, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv16 = {56, 56, 64, 56, 56, 256, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv21 = {56, 56, 64, 28, 28, 64, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv23 = {28, 28, 64, 28, 28, 256, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Pool2dParam pool24 = {56, 56, 256, 28, 28, 1, 1, 0, 0, 2, 2, 1, 1};
BinaryParam binary25 = {28 * 28, 256, relu};
Conv2dParam conv27 = {28, 28, 256, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv29 = {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv31 = {28, 28, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv32 = {28, 28, 256, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv35 = {28, 28, 512, 28, 28, 128, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv39 = {28, 28, 128, 28, 28, 512, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv51 = {28, 28, 128, 14, 14, 128, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv53 = {14, 14, 128, 14, 14, 512, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Pool2dParam pool54 = {28, 28, 512, 14, 14, 1, 1, 0, 0, 2, 2, 1, 1};
BinaryParam binary55 = {14 * 14, 512, relu};
Conv2dParam conv57 = {14, 14, 512, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv59 = {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv61 = {14, 14, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv62 = {14, 14, 512, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv65 = {14, 14, 1024, 14, 14, 256, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv69 = {14, 14, 256, 14, 14, 1024, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv95 = {14, 14, 256, 7, 7, 256, 3, 3, 1, 1, 2, 2, 1, 1, relu};
Conv2dParam conv97 = {7, 7, 256, 7, 7, 1024, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Pool2dParam pool98 = {14, 14, 1024, 7, 7, 1, 1, 0, 0, 2, 2, 1, 1};
BinaryParam binary99 = {7 * 7, 1024, relu};
Conv2dParam conv101 = {7, 7, 1024, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv103 = {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1, 1, 1, relu};
Conv2dParam conv105 = {7, 7, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1, noop};
Conv2dParam conv106 = {7, 7, 1024, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv109 = {7, 7, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, relu};
Conv2dParam conv113 = {7, 7, 512, 7, 7, 2048, 1, 1, 0, 0, 1, 1, 1, 1, relu};
ReduceParam reduce123 = {7 * 7, 2048, 1};
FCParam fc125 = {1, 2048, 1000};

} // namespace

//
//    ResNet50V17Ref
//

ResNet50V17Ref::ResNet50V17Ref(int N):
        NetRef(N) { }

ResNet50V17Ref::~ResNet50V17Ref() { }

void ResNet50V17Ref::init(const std::string &data_dir) {
    NetRef::set_data_dir(data_dir);
    init_layers();
    load_buffers();
}

int ResNet50V17Ref::input_count() {
    return 1;
}

void ResNet50V17Ref::set_input(int index, const std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 0;
    int volume = m_N * 224 * 224 * 3;
    assert(int(data.size()) == volume);
    base::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    memcpy(buffer->data(), data.data(), volume * sizeof(float));
}

int ResNet50V17Ref::output_count() {
    return 1;
}

void ResNet50V17Ref::get_output(int index, std::vector<float> &data) {
    assert(index == 0);
    int buffer_index = 233;
    int volume = m_N * 1000;
    data.resize(volume);
    base::Buffer *buffer = get_buffer(buffer_index);
    assert(buffer != nullptr);
    assert(buffer->volume() >= volume);
    memcpy(data.data(), buffer->data(), volume * sizeof(float));
}

void ResNet50V17Ref::run() {
    NetRef::run();
}

void ResNet50V17Ref::init_layers() {
    init_conv2d(0, 1, 2, -1, 4, conv1);
    init_max_pool2d(4, 5, pool3);
    init_conv2d(5, 6, 7, -1, 9, conv4);
    init_conv2d(9, 10, 11, -1, 13, conv6);
    init_conv2d(13, 14, 15, -1, 16, conv8);
    init_conv2d(5, 17, 18, 16, 21, conv9);
    init_conv2d(21, 22, 23, -1, 25, conv12);
    init_conv2d(25, 26, 27, -1, 29, conv6);
    init_conv2d(29, 30, 31, 21, 34, conv16);
    init_conv2d(34, 35, 36, -1, 38, conv12);
    init_conv2d(38, 39, 40, -1, 42, conv21);
    init_conv2d(42, 43, 44, -1, 45, conv23);
    init_max_pool2d(34, 46, pool24);
    init_add(45, 46, 48, binary25);
    init_conv2d(48, 49, 50, -1, 52, conv27);
    init_conv2d(52, 53, 54, -1, 56, conv29);
    init_conv2d(56, 57, 58, -1, 59, conv31);
    init_conv2d(48, 60, 61, 59, 64, conv32);
    init_conv2d(64, 65, 66, -1, 68, conv35);
    init_conv2d(68, 69, 70, -1, 72, conv29);
    init_conv2d(72, 73, 74, 64, 77, conv39);
    init_conv2d(77, 78, 79, -1, 81, conv35);
    init_conv2d(81, 82, 83, -1, 85, conv29);
    init_conv2d(85, 86, 87, 77, 90, conv39);
    init_conv2d(90, 91, 92, -1, 94, conv35);
    init_conv2d(94, 95, 96, -1, 98, conv51);
    init_conv2d(98, 99, 100, -1, 101, conv53);
    init_max_pool2d(90, 102, pool54);
    init_add(101, 102, 104, binary55);
    init_conv2d(104, 105, 106, -1, 108, conv57);
    init_conv2d(108, 109, 110, -1, 112, conv59);
    init_conv2d(112, 113, 114, -1, 115, conv61);
    init_conv2d(104, 116, 117, 115, 120, conv62);
    init_conv2d(120, 121, 122, -1, 124, conv65);
    init_conv2d(124, 125, 126, -1, 128, conv59);
    init_conv2d(128, 129, 130, 120, 133, conv69);
    init_conv2d(133, 134, 135, -1, 137, conv65);
    init_conv2d(137, 138, 139, -1, 141, conv59);
    init_conv2d(141, 142, 143, 133, 146, conv69);
    init_conv2d(146, 147, 148, -1, 150, conv65);
    init_conv2d(150, 151, 152, -1, 154, conv59);
    init_conv2d(154, 155, 156, 146, 159, conv69);
    init_conv2d(159, 160, 161, -1, 163, conv65);
    init_conv2d(163, 164, 165, -1, 167, conv59);
    init_conv2d(167, 168, 169, 159, 172, conv69);
    init_conv2d(172, 173, 174, -1, 176, conv65);
    init_conv2d(176, 177, 178, -1, 180, conv95);
    init_conv2d(180, 181, 182, -1, 183, conv97);
    init_max_pool2d(172, 184, pool98);
    init_add(183, 184, 186, binary99);
    init_conv2d(186, 187, 188, -1, 190, conv101);
    init_conv2d(190, 191, 192, -1, 194, conv103);
    init_conv2d(194, 195, 196, -1, 197, conv105);
    init_conv2d(186, 198, 199, 197, 202, conv106);
    init_conv2d(202, 203, 204, -1, 206, conv109);
    init_conv2d(206, 207, 208, -1, 210, conv103);
    init_conv2d(210, 211, 212, 202, 215, conv113);
    init_conv2d(215, 216, 217, -1, 219, conv109);
    init_conv2d(219, 220, 221, -1, 223, conv103);
    init_conv2d(223, 224, 225, 215, 228, conv113);
    init_reduce_mean(228, 229, reduce123);
    init_fc(229, 230, 232, 233, fc125); 
}

void ResNet50V17Ref::load_buffers() {
    load_buffer(1, "var0001.dat");
    load_buffer(2, "var0002.dat");
    load_buffer(6, "var0003.dat");
    load_buffer(7, "var0004.dat");
    load_buffer(10, "var0005.dat");
    load_buffer(11, "var0006.dat");
    load_buffer(14, "var0007.dat");
    load_buffer(15, "var0008.dat");
    load_buffer(17, "var0009.dat");
    load_buffer(18, "var0010.dat");
    load_buffer(22, "var0011.dat");
    load_buffer(23, "var0012.dat");
    load_buffer(26, "var0013.dat");
    load_buffer(27, "var0014.dat");
    load_buffer(30, "var0015.dat");
    load_buffer(31, "var0016.dat");
    load_buffer(35, "var0017.dat");
    load_buffer(36, "var0018.dat");
    load_buffer(39, "var0019.dat");
    load_buffer(40, "var0020.dat");
    load_buffer(43, "var0021.dat");
    load_buffer(44, "var0022.dat");
    load_buffer(49, "var0023.dat");
    load_buffer(50, "var0024.dat");
    load_buffer(53, "var0025.dat");
    load_buffer(54, "var0026.dat");
    load_buffer(57, "var0027.dat");
    load_buffer(58, "var0028.dat");
    load_buffer(60, "var0029.dat");
    load_buffer(61, "var0030.dat");
    load_buffer(65, "var0031.dat");
    load_buffer(66, "var0032.dat");
    load_buffer(69, "var0033.dat");
    load_buffer(70, "var0034.dat");
    load_buffer(73, "var0035.dat");
    load_buffer(74, "var0036.dat");
    load_buffer(78, "var0037.dat");
    load_buffer(79, "var0038.dat");
    load_buffer(82, "var0039.dat");
    load_buffer(83, "var0040.dat");
    load_buffer(86, "var0041.dat");
    load_buffer(87, "var0042.dat");
    load_buffer(91, "var0043.dat");
    load_buffer(92, "var0044.dat");
    load_buffer(95, "var0045.dat");
    load_buffer(96, "var0046.dat");
    load_buffer(99, "var0047.dat");
    load_buffer(100, "var0048.dat");
    load_buffer(105, "var0049.dat");
    load_buffer(106, "var0050.dat");
    load_buffer(109, "var0051.dat");
    load_buffer(110, "var0052.dat");
    load_buffer(113, "var0053.dat");
    load_buffer(114, "var0054.dat");
    load_buffer(116, "var0055.dat");
    load_buffer(117, "var0056.dat");
    load_buffer(121, "var0057.dat");
    load_buffer(122, "var0058.dat");
    load_buffer(125, "var0059.dat");
    load_buffer(126, "var0060.dat");
    load_buffer(129, "var0061.dat");
    load_buffer(130, "var0062.dat");
    load_buffer(134, "var0063.dat");
    load_buffer(135, "var0064.dat");
    load_buffer(138, "var0065.dat");
    load_buffer(139, "var0066.dat");
    load_buffer(142, "var0067.dat");
    load_buffer(143, "var0068.dat");
    load_buffer(147, "var0069.dat");
    load_buffer(148, "var0070.dat");
    load_buffer(151, "var0071.dat");
    load_buffer(152, "var0072.dat");
    load_buffer(155, "var0073.dat");
    load_buffer(156, "var0074.dat");
    load_buffer(160, "var0075.dat");
    load_buffer(161, "var0076.dat");
    load_buffer(164, "var0077.dat");
    load_buffer(165, "var0078.dat");
    load_buffer(168, "var0079.dat");
    load_buffer(169, "var0080.dat");
    load_buffer(173, "var0081.dat");
    load_buffer(174, "var0082.dat");
    load_buffer(177, "var0083.dat");
    load_buffer(178, "var0084.dat");
    load_buffer(181, "var0085.dat");
    load_buffer(182, "var0086.dat");
    load_buffer(187, "var0087.dat");
    load_buffer(188, "var0088.dat");
    load_buffer(191, "var0089.dat");
    load_buffer(192, "var0090.dat");
    load_buffer(195, "var0091.dat");
    load_buffer(196, "var0092.dat");
    load_buffer(198, "var0093.dat");
    load_buffer(199, "var0094.dat");
    load_buffer(203, "var0095.dat");
    load_buffer(204, "var0096.dat");
    load_buffer(207, "var0097.dat");
    load_buffer(208, "var0098.dat");
    load_buffer(211, "var0099.dat");
    load_buffer(212, "var0100.dat");
    load_buffer(216, "var0101.dat");
    load_buffer(217, "var0102.dat");
    load_buffer(220, "var0103.dat");
    load_buffer(221, "var0104.dat");
    load_buffer(224, "var0105.dat");
    load_buffer(225, "var0106.dat");
    load_buffer(230, "var0107.dat");
    load_buffer(232, "var0108.dat");
}

void ResNet50V17Ref::init_add(
        int ia,
        int ib,
        int ic,
        const base::BinaryParam &param) {
    base::init_add(this, ia, ib, ic, param);
}

void ResNet50V17Ref::init_conv2d(
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

void ResNet50V17Ref::init_fc(
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

void ResNet50V17Ref::init_max_pool2d(
        int ix,
        int iy,
        const base::Pool2dParam &param) {
    base::init_max_pool2d(this, ix, iy, param);
}

void ResNet50V17Ref::init_reduce_mean(
        int ix,
        int iy,
        const base::ReduceParam &param) {
    base::init_reduce_mean(this, ix, iy, param);
}

} // namespace ref
} // namespace resnet50_v1_7
} // namespace nn
} // namespace ronin

