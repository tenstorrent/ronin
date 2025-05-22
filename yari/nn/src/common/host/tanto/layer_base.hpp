// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <utility>

#include "host/base/post_op.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

namespace base = ronin::op::common::base;

//
//    Layer
//

class Layer {
public:
    Layer() { }
    virtual ~Layer() { }
public:
    virtual int input_volume(int index) = 0;
    virtual int output_volume(int index) = 0;
    virtual std::vector<float> transform_input(int index, const std::vector<float> &x) = 0;
    virtual std::vector<float> transform_output(int index, const std::vector<float> &x) = 0;
    virtual void run() = 0;
};

template<typename OP>
class LayerBase: public Layer {
public:
    template<typename... ARGS>
    LayerBase(ARGS &&...args):
        m_op(std::forward<ARGS>(args)...) { }
    ~LayerBase() { }
public:
    template<typename... ARGS>
    void init(ARGS &&...args) {
        m_op.init(std::forward<ARGS>(args)...);
    }
    int input_volume(int index) override {
        return m_op.input_volume(index);
    }
    int output_volume(int index) override {
        return m_op.output_volume(index);
    }
    std::vector<float> transform_input(int index, const std::vector<float> &x) override {
        return m_op.transform_input(index, x);
    }
    std::vector<float> transform_output(int index, const std::vector<float> &x) override {
        return m_op.transform_output(index, x);
    }
    void run() override {
        m_op.run();
    }
protected:
    OP m_op;
};

//
//    Binary
//

struct BinaryParam {
    int H;
    int C;
    base::PostOpSpec post_op;
};

//
//    Conv2d
//

struct Conv2dParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int K;
    int R;
    int S;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    base::PostOpSpec post_op;
};

//
//    FC
//

struct FCParam {
    int H;
    int C;
    int K;
};

//
//    GroupConv2d
//

struct GroupConv2dParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int K;
    int R;
    int S;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int groups;
    base::PostOpSpec post_op;
};

struct DSConv2dParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int K;
    int R;
    int S;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    base::PostOpSpec post_op;
};

//
//    Pool2d
//

struct Pool2dParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int R;
    int S;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
};

//
//    Reduce
//

struct ReduceParam {
    int H;
    int W;
    int axis;
};

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

