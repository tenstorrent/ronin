// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "host/ref/layer_base.hpp"
#include "host/ref/layer_ref.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace ref {

//
//    AddRefLayer
//

AddRefLayer::AddRefLayer(int N, const BinaryParam &param):
        LayerBase(
            N, 
            param.H,
            param.C, 
            param.post_op) { }

AddRefLayer::~AddRefLayer() { }

//
//    SubRefLayer
//

SubRefLayer::SubRefLayer(int N, const BinaryParam &param):
        LayerBase(
            N, 
            param.H,
            param.C, 
            param.post_op) { }

SubRefLayer::~SubRefLayer() { }

//
//    MulRefLayer
//

MulRefLayer::MulRefLayer(int N, const BinaryParam &param):
        LayerBase(
            N, 
            param.H,
            param.C, 
            param.post_op) { }

MulRefLayer::~MulRefLayer() { }

//
//    Conv2dRefLayer
//

Conv2dRefLayer::Conv2dRefLayer(int N, const Conv2dParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.K,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            param.dilation_h,
            param.dilation_w,
            param.post_op) { }

Conv2dRefLayer::~Conv2dRefLayer() { }

//
//    FCRefLayer
//

FCRefLayer::FCRefLayer(int N, const FCParam &param):
        LayerBase(
            N,
            param.H,
            param.C,
            param.K) { }

FCRefLayer::~FCRefLayer() { }

//
//    GroupConv2dRefLayer
//

GroupConv2dRefLayer::GroupConv2dRefLayer(int N, const GroupConv2dParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.K,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            param.dilation_h,
            param.dilation_w,
            param.groups,
            param.post_op) { }

GroupConv2dRefLayer::~GroupConv2dRefLayer() { }

//
//    AvgPool2dRefLayer
//

AvgPool2dRefLayer::AvgPool2dRefLayer(int N, const Pool2dParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            param.dilation_h,
            param.dilation_w) { }

AvgPool2dRefLayer::~AvgPool2dRefLayer() { }

//
//    MaxPool2dRefLayer
//

MaxPool2dRefLayer::MaxPool2dRefLayer(int N, const Pool2dParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            param.dilation_h,
            param.dilation_w) { }

MaxPool2dRefLayer::~MaxPool2dRefLayer() { }

//
//    ReduceMaxRefLayer
//

ReduceMaxRefLayer::ReduceMaxRefLayer(int N, const ReduceParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.axis) { }

ReduceMaxRefLayer::~ReduceMaxRefLayer() { }

//
//    ReduceMeanRefLayer
//

ReduceMeanRefLayer::ReduceMeanRefLayer(int N, const ReduceParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.axis) { }

ReduceMeanRefLayer::~ReduceMeanRefLayer() { }

//
//    ReduceSumRefLayer
//

ReduceSumRefLayer::ReduceSumRefLayer(int N, const ReduceParam &param):
        LayerBase(
            N,
            param.H,
            param.W,
            param.axis) { }

ReduceSumRefLayer::~ReduceSumRefLayer() { }

} // namespace ref
} // namespace common
} // namespace nn
} // namespace ronin

