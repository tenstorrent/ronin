// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

namespace core = ronin::tanto::host;

enum class BcastOp {
    Add,
    Sub,
    Mul
};

enum class BcastDim {
    Rows,
    Cols,
    Scalar
};

class Bcast {
public:
    Bcast();
    ~Bcast();
public:
    void init(
        const core::Device &device,
        BcastOp op, 
        BcastDim dim,
        int N,
        int C,
        int H,
        int W);
    void run(
        const void *a,
        const void *b,
        void *c);
private:
    void create_globals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_reader_rows();
    void create_reader_cols();
    void create_reader_scalar();
    void create_writer();
    void create_writer_rows();
    void create_writer_cols();
    void create_writer_scalar();
    void create_math();
    void create_math_rows();
    void create_math_cols();
    void create_math_scalar();
    uint32_t get_bcast_size();
    std::string make_reader_name();
    std::string make_writer_name();
    std::string make_math_name();
    static std::string op_to_str(BcastOp op);
    static std::string dim_to_str(BcastDim dim);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t 
        TILE_HEIGHT = 32,
        TILE_WIDTH = 32,
        TILE_SIZE = 1024;
private:
    core::Device m_device;
    BcastOp m_op = BcastOp(0);
    BcastDim m_dim = BcastDim(0);
    uint32_t m_N = 0;
    uint32_t m_C = 0;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_pipe_frame_size = 0;
    core::Program m_program;
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

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

