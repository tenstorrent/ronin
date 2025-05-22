// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

enum class BcastRefOp {
    Add,
    Sub,
    Mul
};

enum class BcastRefDim {
    Rows,
    Cols,
    Scalar
};

class BcastRef {
public:
    BcastRef();
    ~BcastRef();
public:
    void init(
        BcastRefOp op, 
        BcastRefDim dim,
        int N,
        int C,
        int H,
        int W);
    void run(
        const float *a,
        const float *b,
        float *c);
private:
    void add_rows(
        const float *a,
        const float *b,
        float *c);
    void add_cols(
        const float *a,
        const float *b,
        float *c);
    void add_scalar(
        const float *a,
        const float *b,
        float *c);
    void sub_rows(
        const float *a,
        const float *b,
        float *c);
    void sub_cols(
        const float *a,
        const float *b,
        float *c);
    void sub_scalar(
        const float *a,
        const float *b,
        float *c);
    void mul_rows(
        const float *a,
        const float *b,
        float *c);
    void mul_cols(
        const float *a,
        const float *b,
        float *c);
    void mul_scalar(
        const float *a,
        const float *b,
        float *c);
private:
    BcastRefOp m_op = BcastRefOp(0);
    BcastRefDim m_dim = BcastRefDim(0);
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

