// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <utility>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/eltwise_binary.hpp"

//
//    Basic elementwise binary operations implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/eltwise_binary_reader.cpp >metal/eltwise_binary_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/eltwise_binary_writer.cpp >metal/eltwise_binary_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/eltwise_binary_math.cpp >metal/eltwise_add_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/eltwise_binary_math.cpp >metal/eltwise_sub_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=2 \
//        tanto/eltwise_binary_math.cpp >metal/eltwise_mul_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    EltwiseBinary
//

EltwiseBinary::EltwiseBinary() { }

EltwiseBinary::~EltwiseBinary() { }

void EltwiseBinary::init(
        const core::Device &device,
        EltwiseBinaryOp op, 
        int N) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_op = op;
    m_N = uint32_t(N);
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    // TODO: Implement switch between Metal and Tanto kernels
    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void EltwiseBinary::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void EltwiseBinary::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_ga = core::Global(m_device, T, m_N, log2_page_size);
    m_gb = core::Global(m_device, T, m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_N, log2_page_size);
}

void EltwiseBinary::create_pipes() {
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

void EltwiseBinary::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void EltwiseBinary::create_reader() {
    std::string path = m_kernel_base_path + "/eltwise_binary_reader.cpp";
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
        uint32 num_blocks,
        uint32 block_tiles) 
*/
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        m_num_blocks,
        m_block_tiles
    };
    m_reader.set_args(m_grid, args);
}

void EltwiseBinary::create_writer() {
    std::string path = m_kernel_base_path + "/eltwise_binary_writer.cpp";
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
        uint32_t gc_pos,
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        m_num_blocks,
        m_block_tiles
    };
    m_writer.set_args(m_grid, args);
}

void EltwiseBinary::create_math() {
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
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        m_num_blocks,
        m_block_tiles
    };
    m_math.set_args(m_grid, args);
}

std::string EltwiseBinary::make_math_name() {
    // makes name of precompiled kernel
    switch (m_op) {
    case EltwiseBinaryOp::Add:
        return "eltwise_add_math";
    case EltwiseBinaryOp::Sub:
        return "eltwise_sub_math";
    case EltwiseBinaryOp::Mul:
        return "eltwise_mul_math";
    default:
        assert(false);
        return "invalid";
    }
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

