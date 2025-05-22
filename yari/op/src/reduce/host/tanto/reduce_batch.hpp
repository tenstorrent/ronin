// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <map>

#include "host/core/api.hpp"

namespace ronin {
namespace op {
namespace reduce {
namespace tanto {

namespace core = ronin::tanto::host;

enum class ReduceBatchOp {
    MAX,
    MEAN,
    SUM
};

class ReduceBatch {
public:
    ReduceBatch(
        ReduceBatchOp op,
        int N,
        int H,
        int W,
        int axis,
        int batch_size);
    virtual ~ReduceBatch();
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
    void create_cols_writer();
    void create_rows_writer();
    void create_cols_math();
    void create_rows_math();
    void init_locals();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
    std::string make_math_name();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
private:
    core::Device m_device;
    ReduceBatchOp m_op = ReduceBatchOp(0);
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_axis = 0;
    core::Program m_program;
    uint32_t m_x_start = 0;
    uint32_t m_y_start = 0;
    uint32_t m_x_end = 0;
    uint32_t m_y_end = 0;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gs;
    core::Global m_gy;
    core::Global m_gzero;
    core::Local m_lzero;
    core::Pipe m_px;
    core::Pipe m_ps;
    core::Pipe m_py;
    core::Pipe m_px_im;
    core::Pipe m_py_im;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_zero_size = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_ps_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_px_im_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

class ReduceMaxBatch: public ReduceBatch {
public:
    ReduceMaxBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size);
    ~ReduceMaxBatch();
};

class ReduceMeanBatch: public ReduceBatch {
public:
    ReduceMeanBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size);
    ~ReduceMeanBatch();
};

class ReduceSumBatch: public ReduceBatch {
public:
    ReduceSumBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size);
    ~ReduceSumBatch();
};

} // namespace tanto
} // namespace reduce
} // namespace op
} // namespace ronin

