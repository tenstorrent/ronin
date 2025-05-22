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
#include "host/tanto/bcast.hpp"

//
//    Basic broadcast operations implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/bcast_rows_reader.cpp >metal/bcast_rows_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/bcast_rows_writer.cpp >metal/bcast_rows_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/bcast_rows_math.cpp >metal/bcast_rows_add_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/bcast_rows_math.cpp >metal/bcast_rows_sub_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=2 \
//        tanto/bcast_rows_math.cpp >metal/bcast_rows_mul_math.cpp
//
//    tanto --mode=read -DT=bfloat16 \
//        tanto/bcast_cols_reader.cpp >metal/bcast_cols_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/bcast_cols_writer.cpp >metal/bcast_cols_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/bcast_cols_math.cpp >metal/bcast_cols_add_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/bcast_cols_math.cpp >metal/bcast_cols_sub_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=2 \
//        tanto/bcast_cols_math.cpp >metal/bcast_cols_mul_math.cpp
//
//    tanto --mode=read -DT=bfloat16 \
//        tanto/bcast_scalar_reader.cpp >metal/bcast_scalar_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/bcast_scalar_writer.cpp >metal/bcast_scalar_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/bcast_scalar_math.cpp >metal/bcast_scalar_add_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/bcast_scalar_math.cpp >metal/bcast_scalar_sub_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=2 \
//        tanto/bcast_scalar_math.cpp >metal/bcast_scalar_mul_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    Bcast
//

Bcast::Bcast() { }

Bcast::~Bcast() { }

void Bcast::init(
        const core::Device &device,
        BcastOp op, 
        BcastDim dim,
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

    create_globals();
    create_pipes();
    create_kernels();
}

void Bcast::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void Bcast::create_globals() {
    uint32_t asize = m_N * m_C * m_H * m_W;
    uint32_t bsize = get_bcast_size();
    uint32_t csize = asize;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_ga = core::Global(m_device, T, asize, log2_page_size);
    m_gb = core::Global(m_device, T, bsize, log2_page_size);
    m_gc = core::Global(m_device, T, csize, log2_page_size);
}

void Bcast::create_pipes() {
    m_pa =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pc =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void Bcast::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void Bcast::create_reader() {
    // kernel construction is identical for all "dim" values
    // still use switch to keep implementation generic
    switch (m_dim) {
    case BcastDim::Rows:
        create_reader_rows();
        break;
    case BcastDim::Cols:
        create_reader_cols();
        break;
    case BcastDim::Scalar:
        create_reader_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Bcast::create_reader_rows() {
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
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 NC,
        uint32 Ht,
        uint32 Wt,
        uint32 gb_no_nc)
*/
    uint32_t NC = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        NC,
        Ht,
        Wt,
        uint32_t(0)
    };
    m_reader.set_args(m_grid, args);
}

void Bcast::create_reader_cols() {
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
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 NC,
        uint32 Ht,
        uint32 Wt,
        uint32 gb_no_nc)
*/
    uint32_t NC = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        NC,
        Ht,
        Wt,
        uint32_t(0)
    };
    m_reader.set_args(m_grid, args);
}

void Bcast::create_reader_scalar() {
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
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 NC,
        uint32 Ht,
        uint32 Wt,
        uint32 gb_no_nc)
*/
    uint32_t NC = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        NC,
        Ht,
        Wt,
        uint32_t(0)
    };
    m_reader.set_args(m_grid, args);
}

void Bcast::create_writer() {
    // kernel construction is identical for all "dim" values
    // still use switch to keep implementation generic
    switch (m_dim) {
    case BcastDim::Rows:
        create_writer_rows();
        break;
    case BcastDim::Cols:
        create_writer_cols();
        break;
    case BcastDim::Scalar:
        create_writer_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Bcast::create_writer_rows() {
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
        global<T> gc,
        pipe<T> pc,
        uint32 gc_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = (m_N * m_C * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Bcast::create_writer_cols() {
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
        global<T> gc,
        pipe<T> pc,
        uint32 gc_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = (m_N * m_C * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Bcast::create_writer_scalar() {
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
        global<T> gc,
        pipe<T> pc,
        uint32 gc_pos,
        uint32 num_tiles)
*/
    uint32_t num_tiles = (m_N * m_C * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        num_tiles
    };
    m_writer.set_args(m_grid, args);
}

void Bcast::create_math() {
    // kernel construction is identical for all "dim" values
    // still use switch to keep implementation generic
    switch (m_dim) {
    case BcastDim::Rows:
        create_math_rows();
        break;
    case BcastDim::Cols:
        create_math_cols();
        break;
    case BcastDim::Scalar:
        create_math_scalar();
        break;
    default:
        assert(false);
        break;
    }
}

void Bcast::create_math_rows() {
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
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 B,
        uint32 Ht,
        uint32 Wt)
*/
    uint32_t B = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        B,
        Ht,
        Wt
    };
    m_math.set_args(m_grid, args);
}

void Bcast::create_math_cols() {
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
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 B,
        uint32 Ht,
        uint32 Wt)
*/
    uint32_t B = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        B,
        Ht,
        Wt
    };
    m_math.set_args(m_grid, args);
}

void Bcast::create_math_scalar() {
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
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 B,
        uint32 Ht,
        uint32 Wt)
*/
    uint32_t B = m_N * m_C;
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        B,
        Ht,
        Wt
    };
    m_math.set_args(m_grid, args);
}

uint32_t Bcast::get_bcast_size() {
    switch (m_dim) {
    case BcastDim::Rows:
        return m_N * m_C * TILE_HEIGHT * m_W;
    case BcastDim::Cols:
        return m_N * m_C * m_H * TILE_WIDTH;
    case BcastDim::Scalar:
        return m_N * m_C * TILE_SIZE;
    default:
        assert(false);
        return 0;
    }
}

std::string Bcast::make_reader_name() {
    return "bcast_" + dim_to_str(m_dim) + "_reader";
}

std::string Bcast::make_writer_name() {
    return "bcast_" + dim_to_str(m_dim) + "_writer";
}

std::string Bcast::make_math_name() {
    return "bcast_" + dim_to_str(m_dim) + "_" + op_to_str(m_op) + "_math";
}

std::string Bcast::op_to_str(BcastOp op) {
    switch (op) {
    case BcastOp::Add:
        return "add";
    case BcastOp::Sub:
        return "sub";
    case BcastOp::Mul:
        return "mul";
    default:
        assert(false);
        return "invalid";
    }
}

std::string Bcast::dim_to_str(BcastDim dim) {
    switch (dim) {
    case BcastDim::Rows:
        return "rows";
    case BcastDim::Cols:
        return "cols";
    case BcastDim::Scalar:
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

