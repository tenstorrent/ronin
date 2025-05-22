// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

enum class ReduceRefOp {
    Max,
    Sum
};

enum class ReduceRefDim {
    Rows,
    Cols,
    Scalar
};

class ReduceRef {
public:
    ReduceRef();
    ~ReduceRef();
public:
    void init(
        ReduceRefOp op, 
        ReduceRefDim dim,
        int N,
        int C,
        int H,
        int W);
    void run(
        const float *x,
        float scaler,
        float *y);
private:
    void max_rows(
        const float *x,
        float scaler,
        float *y);
    void sum_rows(
        const float *x,
        float scaler,
        float *y);
    void max_cols(
        const float *x,
        float scaler,
        float *y);
    void sum_cols(
        const float *x,
        float scaler,
        float *y);
    void max_scalar(
        const float *x,
        float scaler,
        float *y);
    void sum_scalar(
        const float *x,
        float scaler,
        float *y);
private:
    ReduceRefOp m_op = ReduceRefOp(0);
    ReduceRefDim m_dim = ReduceRefDim(0);
    int m_N = 0;
    int m_C = 0;
    int m_H = 0;
    int m_W = 0;
};

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

