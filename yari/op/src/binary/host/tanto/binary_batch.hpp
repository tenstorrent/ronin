// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

namespace ronin {
namespace op {
namespace binary {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

enum class BinaryBatchOp {
    ADD,
    SUB,
    MUL
};

class BinaryBatch {
public:
    BinaryBatch(
        BinaryBatchOp op,
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size);
    ~BinaryBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &ga,
        const core::Global &gb,
        const core::Global &gc);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void validate_globals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_writer();
    void create_math();
    void create_unary_math();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
    std::string make_math_name();
    std::string get_unary_kernel_suffix();
    uint32_t encode_unary_param0();
    bool is_unary_relu6();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
private:
    core::Device m_device;
    BinaryBatchOp m_op = BinaryBatchOp(0);
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_C = 0;
    base::PostOpSpec m_post_op;
    uint32_t m_H_arg = 0;
    uint32_t m_C_arg = 0;
    uint32_t m_pipe_frame_size = 0;
    core::Program m_program;
    uint32_t m_x_start = 0;
    uint32_t m_y_start = 0;
    uint32_t m_x_end = 0;
    uint32_t m_y_end = 0;
    core::Grid m_grid;
    core::Global m_ga;
    core::Global m_gb;
    core::Global m_gc;
    core::Pipe m_pa;
    core::Pipe m_pb;
    core::Pipe m_pc;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

class AddBatch: public BinaryBatch {
public:
    AddBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size);
    ~AddBatch();
};

class SubBatch: public BinaryBatch {
public:
    SubBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size);
    ~SubBatch();
};

class MulBatch: public BinaryBatch {
public:
    MulBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size);
    ~MulBatch();
};

} // namespace tanto
} // namespace binary
} // namespace op
} // namespace ronin

