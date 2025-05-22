// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/reduce.hpp"

//
//    Basic reduce operations implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/reduce_rows_reader.cpp >metal/reduce_rows_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/reduce_rows_writer.cpp >metal/reduce_rows_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/reduce_rows_math.cpp >metal/reduce_rows_max_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/reduce_rows_math.cpp >metal/reduce_rows_sum_math.cpp
//
//    tanto --mode=read -DT=bfloat16 \
//        tanto/reduce_cols_reader.cpp >metal/reduce_cols_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/reduce_cols_writer.cpp >metal/reduce_cols_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/reduce_cols_math.cpp >metal/reduce_cols_max_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/reduce_cols_math.cpp >metal/reduce_cols_sum_math.cpp
//
//    tanto --mode=read -DT=bfloat16 \
//        tanto/reduce_scalar_reader.cpp >metal/reduce_scalar_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/reduce_scalar_writer.cpp >metal/reduce_scalar_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/reduce_scalar_math.cpp >metal/reduce_scalar_max_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/reduce_scalar_math.cpp >metal/reduce_scalar_sum_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    Reduce
//

Reduce::Reduce() { }

Reduce::~Reduce() { }

void Reduce::init(
        const core::Device &device,
        ReduceOp op, 
        ReduceDim dim,
        int N,
        int C,
        int H,
        int W) {
    assert(H % TILE_HEIGHT == 0);
    assert(W % TILE_WIDTH == 0);

    m_device = device;
    m_op = op;
    m_dim = dim;
    m_N = N;
    m_C = C;
    m_H = H;
    m_W = W;
    m_pipe_frame_size = 1;

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    m_scaler_tile.resize(TILE_SIZE);

    create_globals();
    create_pipes();
    create_kernels();
}

void Reduce::run(
        const void *x,
        float scaler,
        void *y) {
    init_scaler(scaler);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gx, x, false);
    queue.enqueue_write(m_gs, m_scaler_tile.data(), false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gy, y, false);
}

void Reduce::create_globals() {
    uint32_t xsize = m_N * m_C * m_H * m_W;
    uint32_t ysize = get_output_size();
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_gx = core::Global(m_device, T, xsize, log2_page_size);
    m_gs = core::Global(m_device, T, 1024, log2_page_size);
    m_gy = core::Global(m_device, T, ysize, log2_page_size);
}

void Reduce::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_ps =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size,
            m_pipe_frame_size);
    m_py =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void Reduce::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void Reduce::create_reader() {
    switch (m_dim) {
    case ReduceDim::Rows:
        create_reader_rows();
        break;
    case ReduceDim::Cols:
        create_reader_cols();
        break;
    case ReduceDim::Scalar:
        create_reader_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Reduce::create_reader_rows() {
    std::string path = m_kernel_base_path + "/" + make_reader_name() + ".cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gs,
        pipe<T> px,
        pipe<T> ps,
        uint32 gx_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = (m_N * m_C * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gs,
        m_px,
        m_ps,
        uint32_t(0), // gx_pos
        num_tiles
    };
    m_reader.set_args(m_grid, args);
}

void Reduce::create_reader_cols() {
    std::string path = m_kernel_base_path + "/" + make_reader_name() + ".cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gs,
        pipe<T> px,
        pipe<T> ps,
        uint32 gx_pos,
        uint32 N,
        uint32 Ht,
        uint32 Wt,
        uint32 HtWt)
*/
    uint32_t NC = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    uint32_t HtWt = Ht * Wt;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gs,
        m_px,
        m_ps,
        uint32_t(0), // gx_pos
        NC,
        Ht,
        Wt,
        HtWt
    };
    m_reader.set_args(m_grid, args);
}

void Reduce::create_reader_scalar() {
    std::string path = m_kernel_base_path + "/" + make_reader_name() + ".cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gs,
        pipe<T> px,
        pipe<T> ps,
        uint32 gx_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = (m_N * m_C * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gs,
        m_px,
        m_ps,
        uint32_t(0), // gx_pos
        num_tiles
    };
    m_reader.set_args(m_grid, args);
}

void Reduce::create_writer() {
    // kernel construction is identical for all "dim" values
    // still use switch to keep implementation generic
    switch (m_dim) {
    case ReduceDim::Rows:
        create_writer_rows();
        break;
    case ReduceDim::Cols:
        create_writer_cols();
        break;
    case ReduceDim::Scalar:
        create_writer_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Reduce::create_writer_rows() {
    std::string path = m_kernel_base_path + "/" + make_writer_name() + ".cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 gy_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = get_output_size() / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Reduce::create_writer_cols() {
    std::string path = m_kernel_base_path + "/" + make_writer_name() + ".cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 gy_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = get_output_size() / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Reduce::create_writer_scalar() {
    std::string path = m_kernel_base_path + "/" + make_writer_name() + ".cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 gy_pos,
        uint32 num_tiles)
*/
    // reduce scalar collapses H and W dimensions
    uint32_t num_tiles = get_output_size() / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Reduce::create_math() {
    // kernel construction is identical for all "dim" values
    // still use switch to keep implementation generic
    switch (m_dim) {
    case ReduceDim::Rows:
        create_math_rows();
        break;
    case ReduceDim::Cols:
        create_math_cols();
        break;
    case ReduceDim::Scalar:
        create_math_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Reduce::create_math_rows() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> ps,
        pipe<T> py,
        uint32 Ht,
        uint32 Wt,
        uint32 NC)
*/
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    uint32_t NC = m_N * m_C;
    std::vector<core::KernelArg> args{
        m_px,
        m_ps,
        m_py,
        Ht,
        Wt,
        NC
    };
    m_math.set_args(m_grid, args);
}

void Reduce::create_math_cols() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> ps,
        pipe<T> py,
        uint32 Ht,
        uint32 Wt,
        uint32 NC)
*/
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    uint32_t NC = m_N * m_C;
    std::vector<core::KernelArg> args{
        m_px,
        m_ps,
        m_py,
        Ht,
        Wt,
        NC
    };
    m_math.set_args(m_grid, args);
}

void Reduce::create_math_scalar() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> ps,
        pipe<T> py,
        uint32 Ht,
        uint32 Wt,
        uint32 NC)
*/
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    uint32_t NC = m_N * m_C;
    std::vector<core::KernelArg> args{
        m_px,
        m_ps,
        m_py,
        Ht,
        Wt,
        NC
    };
    m_math.set_args(m_grid, args);
}

void Reduce::init_scaler(float scaler) {
    uint16_t s = float_as_u16b(scaler);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        m_scaler_tile[i] = s;
    }
}

uint32_t Reduce::get_output_size() {
    switch (m_dim) {
    case ReduceDim::Rows:
        // reduce rows collapses W dimension
        return m_N * m_C * m_H * TILE_WIDTH;
    case ReduceDim::Cols:
        // reduce cols collapses H dimension
        return m_N * m_C * TILE_HEIGHT * m_W;
    case ReduceDim::Scalar:
        // reduce scalar collapses H and W dimensions
        return m_N * m_C * TILE_SIZE;
    default:
        assert(false);
        return 0;
    }
}

std::string Reduce::make_reader_name() {
    return "reduce_" + dim_to_str(m_dim) + "_reader";
}

std::string Reduce::make_writer_name() {
    return "reduce_" + dim_to_str(m_dim) + "_writer";
}

std::string Reduce::make_math_name() {
    return "reduce_" + dim_to_str(m_dim) + "_" + op_to_str(m_op) + "_math";
}

std::string Reduce::op_to_str(ReduceOp op) {
    switch (op) {
    case ReduceOp::Max:
        return "max";
    case ReduceOp::Sum:
        return "sum";
    default:
        assert(false);
        return "invalid";
    }
}

std::string Reduce::dim_to_str(ReduceDim dim) {
    switch (dim) {
    case ReduceDim::Rows:
        return "rows";
    case ReduceDim::Cols:
        return "cols";
    case ReduceDim::Scalar:
        return "scalar";
    default:
        assert(false);
        return "invalid";
    }
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

