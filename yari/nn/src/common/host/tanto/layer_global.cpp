// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "host/tanto/layer_base.hpp"
#include "host/tanto/layer_global.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

//
//    AddRefLayer
//

AddBatchLayer::AddBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size):
            LayerBase(
                N, 
                param.H,
                param.C, 
                param.post_op,
                batch_size) { }

AddBatchLayer::~AddBatchLayer() { }

//
//    SubBatchLayer
//

SubBatchLayer::SubBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size):
            LayerBase(
                N, 
                param.H,
                param.C, 
                param.post_op, 
                batch_size) { }

SubBatchLayer::~SubBatchLayer() { }

//
//    MulBatchLayer
//

MulBatchLayer::MulBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size):
            LayerBase(
                N, 
                param.H,
                param.C, 
                param.post_op,
                batch_size) { }

MulBatchLayer::~MulBatchLayer() { }

//
//    Conv2dBasicBatchLayer
//

Conv2dBasicBatchLayer::Conv2dBasicBatchLayer(
        int N,
        const Conv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

Conv2dBasicBatchLayer::~Conv2dBasicBatchLayer() { }

//
//    Conv2dBasicSplitLayer
//

Conv2dBasicSplitLayer::Conv2dBasicSplitLayer(
        int N,
        const Conv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

Conv2dBasicSplitLayer::~Conv2dBasicSplitLayer() { }

//
//    Conv2dBasicSpatialLayer
//

Conv2dBasicSpatialLayer::Conv2dBasicSpatialLayer(
        int N,
        const Conv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

Conv2dBasicSpatialLayer::~Conv2dBasicSpatialLayer() { }

//
//    Conv2dImageBatchLayer
//

Conv2dImageBatchLayer::Conv2dImageBatchLayer(
        int N,
        const Conv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

Conv2dImageBatchLayer::~Conv2dImageBatchLayer() { }

//
//    FCBatchLayer
//

FCBatchLayer::FCBatchLayer(
        int N, 
        const FCParam &param,
        int batch_size):
            LayerBase(
                N,
                param.H,
                param.C,
                param.K,
                batch_size) { }

FCBatchLayer::~FCBatchLayer() { }

//
//    GroupConv2dBasicBatchLayer
//

GroupConv2dBasicBatchLayer::GroupConv2dBasicBatchLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

GroupConv2dBasicBatchLayer::~GroupConv2dBasicBatchLayer() { }

//
//    GroupConv2dDwBatchLayer
//

GroupConv2dDwBatchLayer::GroupConv2dDwBatchLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

GroupConv2dDwBatchLayer::~GroupConv2dDwBatchLayer() { }

//
//    GroupConv2dDwSpatialLayer
//

GroupConv2dDwSpatialLayer::GroupConv2dDwSpatialLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

GroupConv2dDwSpatialLayer::~GroupConv2dDwSpatialLayer() { }

//
//    DSConv2dBatchLayer
//

DSConv2dBatchLayer::DSConv2dBatchLayer(
        int N,
        const DSConv2dParam &param,
        int batch_size):
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
                param.post_op,
                batch_size) { }

DSConv2dBatchLayer::~DSConv2dBatchLayer() { }

//
//    AvgPool2dBatchLayer
//

AvgPool2dBatchLayer::AvgPool2dBatchLayer(
        int N, 
        const Pool2dParam &param,
        int batch_size):
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
                param.dilation_w,
                batch_size) { }

AvgPool2dBatchLayer::~AvgPool2dBatchLayer() { }

//
//    MaxPool2dBatchLayer
//

MaxPool2dBatchLayer::MaxPool2dBatchLayer(
        int N, 
        const Pool2dParam &param,
        int batch_size):
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
                param.dilation_w,
                batch_size) { }

MaxPool2dBatchLayer::~MaxPool2dBatchLayer() { }

//
//    ReduceMaxBatchLayer
//

ReduceMaxBatchLayer::ReduceMaxBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size):
            LayerBase(
                N,
                param.H,
                param.W,
                param.axis,
                batch_size) { }

ReduceMaxBatchLayer::~ReduceMaxBatchLayer() { }

//
//    ReduceMeanBatchLayer
//

ReduceMeanBatchLayer::ReduceMeanBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size):
            LayerBase(
                N,
                param.H,
                param.W,
                param.axis,
                batch_size) { }

ReduceMeanBatchLayer::~ReduceMeanBatchLayer() { }

//
//    ReduceSumBatchLayer
//

ReduceSumBatchLayer::ReduceSumBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size):
            LayerBase(
                N,
                param.H,
                param.W,
                param.axis,
                batch_size) { }

ReduceSumBatchLayer::~ReduceSumBatchLayer() { }

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

