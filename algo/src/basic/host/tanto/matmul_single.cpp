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
#include "host/tanto/matmul_single.hpp"

//
//    Basic single core matrix multiplication implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/matmul_single_reader.cpp >metal/matmul_single_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/matmul_single_writer.cpp >metal/matmul_single_writer.cpp
//    tanto --mode=compute -DT=bfloat16 \
//        tanto/matmul_single_math.cpp >metal/matmul_single_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    MatmulSingle
//

MatmulSingle::MatmulSingle() { }

MatmulSingle::~MatmulSingle() { }

void MatmulSingle::init(
        const core::Device &device,
        int batch,
        int M,
        int N,
        int K) {
    assert(M % TILE_DIM == 0);
    assert(N % TILE_DIM == 0);
    assert(K % TILE_DIM == 0);

    m_device = device;
    m_batch = uint32_t(batch);
    m_M = uint32_t(M);
    m_N = uint32_t(N);
    m_K = uint32_t(K);
    m_pipe_frame_size = 1;

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void MatmulSingle::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void MatmulSingle::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_ga = core::Global(m_device, T, m_batch * m_M * m_K, log2_page_size);
    m_gb = core::Global(m_device, T, m_batch * m_K * m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_batch * m_M * m_N, log2_page_size);
}

void MatmulSingle::create_pipes() {
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

void MatmulSingle::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void MatmulSingle::create_reader() {
    std::string path = m_kernel_base_path + "/matmul_single_reader.cpp";
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
        uint32 batch,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt,
        uint32 MtKt,
        uint32 KtNt,
        uint32 bcast_b)
*/
    uint32_t Mt = m_M / TILE_DIM;
    uint32_t Nt = m_N / TILE_DIM;
    uint32_t Kt = m_K / TILE_DIM;
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        uint32_t(0), // ga_pos
        uint32_t(0), // gb_pos
        m_batch,
        Mt,
        Kt,
        Nt,
        Mt * Kt,
        Kt * Nt,
        uint32_t(0)  // bcast_b
    };
    m_reader.set_args(m_grid, args);
}

void MatmulSingle::create_writer() {
    std::string path = m_kernel_base_path + "/matmul_single_writer.cpp";
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
        uint32 batch,
        uint32 Mt,
        uint32 Nt)
*/
    uint32_t Mt = m_M / TILE_DIM;
    uint32_t Nt = m_N / TILE_DIM;
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        uint32_t(0), // gc_pos
        m_batch,
        Mt,
        Nt
    };
    m_writer.set_args(m_grid, args);
}

void MatmulSingle::create_math() {
    std::string path = m_kernel_base_path + "/matmul_single_math.cpp";
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
        uint32 batch,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt)
*/
    uint32_t Mt = m_M / TILE_DIM;
    uint32_t Nt = m_N / TILE_DIM;
    uint32_t Kt = m_K / TILE_DIM;
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        m_batch,
        Mt,
        Kt,
        Nt
    };
    m_math.set_args(m_grid, args);
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

