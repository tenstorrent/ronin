// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "host/core/api.hpp"

#include "host/ref/net_ref.hpp"

#include "host/tanto/net_global.hpp"

namespace ronin {
namespace nn {
namespace resnet18 {
namespace tanto {

namespace core = ronin::tanto::host;
namespace ref = common::ref;
namespace base = common::tanto;

class ResNet18MixedTail: public ref::NetRef {
public:
    ResNet18MixedTail(int N);
    ~ResNet18MixedTail();
public:
    void init(const std::string &data_dir);
    int input_count();
    void set_input(int index, const std::vector<float> &data);
    int output_count();
    void get_output(int index, std::vector<float> &data);
    void run();
private:
    void init_layers();
    void load_buffers();
    void init_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const ref::Conv2dParam &param);
    void init_max_pool2d(
        int ix,
        int iy,
        const ref::Pool2dParam &param);
};

class ResNet18MixedHead: public ref::NetRef {
public:
    ResNet18MixedHead(int N);
    ~ResNet18MixedHead();
public:
    void init(const std::string &data_dir);
    int input_count();
    void set_input(int index, const std::vector<float> &data);
    int output_count();
    void get_output(int index, std::vector<float> &data);
    void run();
private:
    void init_layers();
    void load_buffers();
    void init_fc(
        int ix,
        int iw,
        int ib,
        int iy,
        const ref::FCParam &param);
    void init_reduce_mean(
        int ix,
        int iy,
        const ref::ReduceParam &param);
};

class ResNet18MixedMain: public base::NetGlobal {
public:
    ResNet18MixedMain(
        const core::Device &device, 
        int N,
        int batch_size);
    ~ResNet18MixedMain();
public:
    void init(const std::string &data_dir);
    int input_count();
    void set_input(int index, const std::vector<float> &data);
    int output_count();
    void get_output(int index, std::vector<float> &data);
    void run();
private:
    void init_layers();
    void load_buffers();
    void init_conv2d(
        int ix,
        int iw,
        int ib,
        int iz,
        int iy,
        const base::Conv2dParam &param);
private:
    int m_batch_size;
};

class ResNet18Mixed {
public:
    ResNet18Mixed(
        const core::Device &device, 
        int N,
        int batch_size);
    ~ResNet18Mixed();
public:
    void init(const std::string &data_dir);
    int input_count();
    void set_input(int index, const std::vector<float> &data);
    int output_count();
    void get_output(int index, std::vector<float> &data);
    void run();
    void run_tail();
    void run_main();
    void run_head();
private:
    ResNet18MixedTail m_tail;
    ResNet18MixedMain m_main;
    ResNet18MixedHead m_head;
};

} // namespace tanto
} // namespace resnet18
} // namespace nn
} // namespace ronin

