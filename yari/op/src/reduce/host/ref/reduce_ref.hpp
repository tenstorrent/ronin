// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace op {
namespace reduce {
namespace ref {

class ReduceMaxRef {
public:
    ReduceMaxRef(
        int N,
        int H,
        int W,
        int axis);
    ~ReduceMaxRef();
public:
    void init(const float *x, float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_axis = 0;
};

class ReduceMeanRef {
public:
    ReduceMeanRef(
        int N,
        int H,
        int W,
        int axis);
    ~ReduceMeanRef();
public:
    void init(const float *x, float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_axis = 0;
};

class ReduceSumRef {
public:
    ReduceSumRef(
        int N,
        int H,
        int W,
        int axis);
    ~ReduceSumRef();
public:
    void init(const float *x, float *y);
    void run();
    int input_volume(int index);
    int output_volume(int index);
private:
    const float *m_x = nullptr;
    float *m_y = nullptr;
    int m_N = 0;
    int m_H = 0;
    int m_W = 0;
    int m_axis = 0;
};

} // namespace ref
} // namespace reduce
} // namespace op
} // namespace ronin

