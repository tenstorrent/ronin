// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary/host/ref/binary_ref.hpp"
#include "conv/host/ref/conv2d_ref.hpp"
#include "fc/host/ref/fc_ref.hpp"
#include "group_conv/host/ref/group_conv2d_ref.hpp"
#include "pool/host/ref/pool2d_ref.hpp"
#include "reduce/host/ref/reduce_ref.hpp"

#include "host/ref/layer_base.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace ref {

//
//    Binary
//

using op::binary::ref::AddRef;
using op::binary::ref::SubRef;
using op::binary::ref::MulRef;

class AddRefLayer: public LayerBase<AddRef> {
public:
    AddRefLayer(int N, const BinaryParam &param);
    ~AddRefLayer();
};

class SubRefLayer: public LayerBase<SubRef> {
public:
    SubRefLayer(int N, const BinaryParam &param);
    ~SubRefLayer();
};

class MulRefLayer: public LayerBase<MulRef> {
public:
    MulRefLayer(int N, const BinaryParam &param);
    ~MulRefLayer();
};

//
//    Conv2d
//

using op::conv::ref::Conv2dRef;

class Conv2dRefLayer: public LayerBase<Conv2dRef> {
public:
    Conv2dRefLayer(int N, const Conv2dParam &param);
    ~Conv2dRefLayer();
};

//
//    FC
//

using op::fc::ref::FCRef;

class FCRefLayer: public LayerBase<FCRef> {
public:
    FCRefLayer(int N, const FCParam &param);
    ~FCRefLayer();
};

//
//    GroupConv2d
//

using op::group_conv::ref::GroupConv2dRef;

class GroupConv2dRefLayer: public LayerBase<GroupConv2dRef> {
public:
    GroupConv2dRefLayer(int N, const GroupConv2dParam &param);
    ~GroupConv2dRefLayer();
};

//
//    Pool2d
//

using op::pool::ref::AvgPool2dRef;
using op::pool::ref::MaxPool2dRef;

class AvgPool2dRefLayer: public LayerBase<AvgPool2dRef> {
public:
    AvgPool2dRefLayer(int N, const Pool2dParam &param);
    ~AvgPool2dRefLayer();
};

class MaxPool2dRefLayer: public LayerBase<MaxPool2dRef> {
public:
    MaxPool2dRefLayer(int N, const Pool2dParam &param);
    ~MaxPool2dRefLayer();
};

//
//    Reduce
//

using op::reduce::ref::ReduceMaxRef;
using op::reduce::ref::ReduceMeanRef;
using op::reduce::ref::ReduceSumRef;

class ReduceMaxRefLayer: public LayerBase<ReduceMaxRef> {
public:
    ReduceMaxRefLayer(int N, const ReduceParam &param);
    ~ReduceMaxRefLayer();
};

class ReduceMeanRefLayer: public LayerBase<ReduceMeanRef> {
public:
    ReduceMeanRefLayer(int N, const ReduceParam &param);
    ~ReduceMeanRefLayer();
};

class ReduceSumRefLayer: public LayerBase<ReduceSumRef> {
public:
    ReduceSumRefLayer(int N, const ReduceParam &param);
    ~ReduceSumRefLayer();
};

} // namespace ref
} // namespace common
} // namespace nn
} // namespace ronin

