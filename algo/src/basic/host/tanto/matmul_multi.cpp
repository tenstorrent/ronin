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
#include "host/tanto/matmul_split.hpp"
#include "host/tanto/matmul_multi.hpp"

//
//    Basic multi core matrix multiplication implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/matmul_multi_reader.cpp >metal/matmul_multi_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/matmul_multi_writer.cpp >metal/matmul_multi_writer.cpp
//    tanto --mode=compute -DT=bfloat16 \
//        tanto/matmul_multi_math.cpp >metal/matmul_multi_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

namespace {

// introduce method in Grid ?
bool grid_contains(const core::Grid &grid, uint32_t x, uint32_t y) {
    uint32_t count = grid.range_count();
    for (uint32_t i = 0; i < count; i++) {
        core::Range range = grid.range_at(i);
        if (x >= range.x_start && x <= range.x_end && 
                y >= range.y_start && y <= range.y_end) {
            return true;
        }
    }
    return false;
}

} // namespace

//
//    MatmulMulti
//

MatmulMulti::MatmulMulti() { }

MatmulMulti::~MatmulMulti() { }

void MatmulMulti::init(
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
    setup_grids();

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void MatmulMulti::run(
        const void *a,
        const void *b,
        void *c) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ga, a, false);
    queue.enqueue_write(m_gb, b, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gc, c, false);
}

void MatmulMulti::setup_grids() {
    m_device.worker_grid_size(m_workers_x, m_workers_y);
    uint32_t out_tiles = (m_batch * m_M * m_N) / TILE_SIZE;
    matmul_split(
        m_program,
        m_workers_x,
        m_workers_y,
        out_tiles,
        false, // row_wise,
        m_all_cores,
        m_grid,
        m_grid1,
        m_grid2,
        m_grid1_out_tiles,
        m_grid2_out_tiles);
}

void MatmulMulti::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_ga = core::Global(m_device, T, m_batch * m_M * m_K, log2_page_size);
    m_gb = core::Global(m_device, T, m_batch * m_K * m_N, log2_page_size);
    m_gc = core::Global(m_device, T, m_batch * m_M * m_N, log2_page_size);
}

void MatmulMulti::create_pipes() {
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

void MatmulMulti::create_kernels() {
    create_reader_writer();
    create_math();
}

void MatmulMulti::create_reader_writer() {
    std::string reader_path = m_kernel_base_path + "/matmul_multi_reader.cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            reader_path, 
            {}, 
            m_defines);
    std::string writer_path = m_kernel_base_path + "/matmul_multi_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            writer_path, 
            {}, 
            m_defines);
    uint32_t out_tile_pos = 0;
    for (uint32_t i = 0; i < m_all_cores; i++) {
        uint32_t x = i / m_workers_y;
        uint32_t y = i % m_workers_y;
        uint32_t out_num_tiles = 0;
        if (grid_contains(m_grid1, x, y)) {
            out_num_tiles = m_grid1_out_tiles;
        } else if (grid_contains(m_grid2, x, y)) {
            out_num_tiles = m_grid2_out_tiles;
        } else {
            assert(false && "Core not in specified core ranges");
        }
        setup_reader(x, y, out_tile_pos, out_num_tiles);
        setup_writer(x, y, out_tile_pos, out_num_tiles);
        out_tile_pos += out_num_tiles;
    }
}

void MatmulMulti::setup_reader(
        uint32_t x,
        uint32_t y,
        uint32_t out_tile_pos,
        uint32_t out_num_tiles) {
/*
void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt,
        uint32 bcast_b,
        uint32 out_tile_pos,
        uint32 out_num_tiles)
*/
    uint32_t Mt = m_M / TILE_DIM;
    uint32_t Nt = m_N / TILE_DIM;
    uint32_t Kt = m_K / TILE_DIM;
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb,
        Mt,
        Kt,
        Nt,
        uint32_t(0), // bcast_b
        out_tile_pos,
        out_num_tiles
    };
    m_reader.set_args(x, y, args);
}

void MatmulMulti::setup_writer(
        uint32_t x,
        uint32_t y,
        uint32_t out_tile_pos,
        uint32_t out_num_tiles) {
/*
void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 tile_pos,
        uint32 num_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        out_tile_pos,
        out_num_tiles
    };
    m_writer.set_args(x, y, args);
}

void MatmulMulti::create_math() {
    std::string path = m_kernel_base_path + "/matmul_multi_math.cpp";
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
    uint32_t Kt = m_K / TILE_DIM;
    std::vector<core::KernelArg> grid1_args{
        m_pa,
        m_pb,
        m_pc,
        uint32_t(1), // batch
        uint32_t(1), // Mt
        Kt,
        m_grid1_out_tiles // Nt
    };
    m_math.set_args(m_grid1, grid1_args);
    if (m_grid2.range_count() != 0) {
        std::vector<core::KernelArg> grid2_args{
            m_pa,
            m_pb,
            m_pc,
            uint32_t(1), // batch
            uint32_t(1), // Mt
            Kt,
            m_grid2_out_tiles // Nt
        };
        m_math.set_args(m_grid2, grid2_args);
    }
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

