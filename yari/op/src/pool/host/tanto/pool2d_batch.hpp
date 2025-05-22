// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

namespace ronin {
namespace op {
namespace pool {
namespace tanto {

namespace core = ronin::tanto::host;

enum class Pool2dBatchOp {
    AVG,
    MAX
};

class Pool2dBatch {
public:
    Pool2dBatch(
        Pool2dBatchOp op,
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size);
    virtual ~Pool2dBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void validate_globals();
    void create_globals();
    void create_locals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_writer();
    void create_1x1_math();
    void create_avg_math();
    void create_max_math();
    void init_locals();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
    void compute_mask(std::vector<uint32_t> &vmask);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
private:
    core::Device m_device;
    uint32_t m_batch_size = 0;
    Pool2dBatchOp m_op = Pool2dBatchOp(0);
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_C = 0;
    uint32_t m_P = 0;
    uint32_t m_Q = 0;
    uint32_t m_R = 0;
    uint32_t m_S = 0;
    uint32_t m_pad_h = 0;
    uint32_t m_pad_w = 0;
    uint32_t m_stride_h = 0;
    uint32_t m_stride_w = 0;
    uint32_t m_dilation_h = 0;
    uint32_t m_dilation_w = 0;
    core::Program m_program;
    uint32_t m_x_start = 0;
    uint32_t m_y_start = 0;
    uint32_t m_x_end = 0;
    uint32_t m_y_end = 0;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gy;
    core::Global m_ginit;
    core::Global m_gmask;
    core::Local m_lx;
    core::Local m_linit;
    core::Local m_lmask;
    core::Pipe m_px;
    core::Pipe m_py;
    core::Pipe m_py_im;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_init_size = 0;
    uint32_t m_mask_size = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    uint32_t m_start_p = 0;
    uint32_t m_start_q = 0;
    uint32_t m_end_q = 0;
    uint32_t m_delta_p = 0;
    uint32_t m_delta_q = 0;
    uint32_t m_delta_r = 0;
    uint32_t m_delta_s = 0;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

class AvgPool2dBatch: public Pool2dBatch {
public:
    AvgPool2dBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size);
    ~AvgPool2dBatch();
};

class MaxPool2dBatch: public Pool2dBatch {
public:
    MaxPool2dBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size);
    ~MaxPool2dBatch();
};

} // namespace tanto
} // namespace pool
} // namespace op
} // namespace ronin

