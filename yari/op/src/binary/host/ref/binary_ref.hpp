// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host/base/post_op.hpp"

namespace ronin {
namespace op {
namespace binary {
namespace ref {

namespace base = ronin::op::common::base;

class AddRef {
public:
    AddRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op = base::PostOpSpec());
    ~AddRef();
public:
    void init(const float *a, const float *b, float *c);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_a = nullptr;
    const float *m_b = nullptr;
    float *m_c = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_C = 0;
    base::PostOpSpec m_post_op;
};

class SubRef {
public:
    SubRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op = base::PostOpSpec());
    ~SubRef();
public:
    void init(const float *a, const float *b, float *c);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_a = nullptr;
    const float *m_b = nullptr;
    float *m_c = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_C = 0;
    base::PostOpSpec m_post_op;
};

class MulRef {
public:
    MulRef(
        int N, 
        int H,
        int C, 
        const base::PostOpSpec &post_op = base::PostOpSpec());
    ~MulRef();
public:
    void init(const float *a, const float *b, float *c);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_a = nullptr;
    const float *m_b = nullptr;
    float *m_c = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_C = 0;
    base::PostOpSpec m_post_op;
};

} // namespace ref
} // namespace binary
} // namespace op
} // namespace ronin

