// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

enum class EltwiseBinaryRefOp {
    Add,
    Sub,
    Mul
};

class EltwiseBinaryRef {
public:
    EltwiseBinaryRef();
    ~EltwiseBinaryRef();
public:
    void init(EltwiseBinaryRefOp op, int N);
    void run(
        const float *a,
        const float *b,
        float *c);
private:
    void add(
        const float *a,
        const float *b,
        float *c);
    void sub(
        const float *a,
        const float *b,
        float *c);
    void mul(
        const float *a,
        const float *b,
        float *c);
private:
    EltwiseBinaryRefOp m_op = EltwiseBinaryRefOp(0);
    int m_N = 0;
};

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

