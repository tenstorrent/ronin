// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary/host/tanto/binary_batch.hpp"
#include "conv/host/tanto/conv2d_basic_batch.hpp"
#include "conv/host/tanto/conv2d_basic_spatial.hpp"
#include "conv/host/tanto/conv2d_image_batch.hpp"
#include "fc/host/tanto/fc_batch.hpp"
#include "group_conv/host/tanto/group_conv2d_basic_batch.hpp"
#include "group_conv/host/tanto/group_conv2d_dw_batch.hpp"
#include "group_conv/host/tanto/group_conv2d_dw_spatial.hpp"
#include "group_conv/host/tanto/ds_conv2d_batch.hpp"
#include "pool/host/tanto/pool2d_batch.hpp"
#include "reduce/host/tanto/reduce_batch.hpp"

#include "host/tanto/layer_base.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

//
//    Binary
//

using op::binary::tanto::AddBatch;
using op::binary::tanto::SubBatch;
using op::binary::tanto::MulBatch;

class AddBatchLayer: public LayerBase<AddBatch> {
public:
    AddBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size);
    ~AddBatchLayer();
};

class SubBatchLayer: public LayerBase<SubBatch> {
public:
    SubBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size);
    ~SubBatchLayer();
};

class MulBatchLayer: public LayerBase<MulBatch> {
public:
    MulBatchLayer(
        int N, 
        const BinaryParam &param,
        int batch_size);
    ~MulBatchLayer();
};

//
//    Conv2d
//

using op::conv::tanto::Conv2dBasicBatch;

class Conv2dBasicBatchLayer: public LayerBase<Conv2dBasicBatch> {
public:
    Conv2dBasicBatchLayer(
        int N,
        const Conv2dParam &param,
        int batch_size);
    ~Conv2dBasicBatchLayer();
};

using op::conv::tanto::Conv2dBasicSpatial;

class Conv2dBasicSpatialLayer: public LayerBase<Conv2dBasicSpatial> {
public:
    Conv2dBasicSpatialLayer(
        int N,
        const Conv2dParam &param,
        int batch_size);
    ~Conv2dBasicSpatialLayer();
};

using op::conv::tanto::Conv2dImageBatch;

class Conv2dImageBatchLayer: public LayerBase<Conv2dImageBatch> {
public:
    Conv2dImageBatchLayer(
        int N,
        const Conv2dParam &param,
        int batch_size);
    ~Conv2dImageBatchLayer();
};

//
//    FC
//

using op::fc::tanto::FCBatch;

class FCBatchLayer: public LayerBase<FCBatch> {
public:
    FCBatchLayer(
        int N, 
        const FCParam &param,
        int batch_size);
    ~FCBatchLayer();
};

//
//    GroupConv2d
//

using op::group_conv::tanto::GroupConv2dBasicBatch;

class GroupConv2dBasicBatchLayer: public LayerBase<GroupConv2dBasicBatch> {
public:
    GroupConv2dBasicBatchLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size);
    ~GroupConv2dBasicBatchLayer();
};

using op::group_conv::tanto::GroupConv2dDwBatch;

class GroupConv2dDwBatchLayer: public LayerBase<GroupConv2dDwBatch> {
public:
    GroupConv2dDwBatchLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size);
    ~GroupConv2dDwBatchLayer();
};

using op::group_conv::tanto::GroupConv2dDwSpatial;

class GroupConv2dDwSpatialLayer: public LayerBase<GroupConv2dDwSpatial> {
public:
    GroupConv2dDwSpatialLayer(
        int N,
        const GroupConv2dParam &param,
        int batch_size);
    ~GroupConv2dDwSpatialLayer();
};

using op::group_conv::tanto::DSConv2dBatch;

class DSConv2dBatchLayer: public LayerBase<DSConv2dBatch> {
public:
    DSConv2dBatchLayer(
        int N,
        const DSConv2dParam &param,
        int batch_size);
    ~DSConv2dBatchLayer();
};

//
//    Pool2d
//

using op::pool::tanto::AvgPool2dBatch;
using op::pool::tanto::MaxPool2dBatch;

class AvgPool2dBatchLayer: public LayerBase<AvgPool2dBatch> {
public:
    AvgPool2dBatchLayer(
        int N, 
        const Pool2dParam &param,
        int batch_size);
    ~AvgPool2dBatchLayer();
};

class MaxPool2dBatchLayer: public LayerBase<MaxPool2dBatch> {
public:
    MaxPool2dBatchLayer(
        int N, 
        const Pool2dParam &param,
        int batch_size);
    ~MaxPool2dBatchLayer();
};

//
//    Reduce
//

using op::reduce::tanto::ReduceMaxBatch;
using op::reduce::tanto::ReduceMeanBatch;
using op::reduce::tanto::ReduceSumBatch;

class ReduceMaxBatchLayer: public LayerBase<ReduceMaxBatch> {
public:
    ReduceMaxBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size);
    ~ReduceMaxBatchLayer();
};

class ReduceMeanBatchLayer: public LayerBase<ReduceMeanBatch> {
public:
    ReduceMeanBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size);
    ~ReduceMeanBatchLayer();
};

class ReduceSumBatchLayer: public LayerBase<ReduceSumBatch> {
public:
    ReduceSumBatchLayer(
        int N, 
        const ReduceParam &param,
        int batch_size);
    ~ReduceSumBatchLayer();
};

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

