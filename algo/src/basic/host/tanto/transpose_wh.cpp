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
#include "host/tanto/transpose_wh.hpp"

//
//    Basic transpose W/H operation implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/transpose_wh_reader.cpp >metal/transpose_wh_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/transpose_wh_writer.cpp >metal/transpose_wh_writer.cpp
//    tanto --mode=compute -DT=bfloat16 \
//        tanto/transpose_wh_math.cpp >metal/transpose_wh_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    TransposeWh
//

TransposeWh::TransposeWh() { }

TransposeWh::~TransposeWh() { }

void TransposeWh::init(
        const core::Device &device,
        int N,
        int H,
        int W) {
    assert(H % TILE_HEIGHT == 0);
    assert(W % TILE_WIDTH == 0);

    m_device = device;
    m_N = uint32_t(N);
    m_H = uint32_t(H);
    m_W = uint32_t(W);
    m_pipe_frame_size = 1;

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void TransposeWh::run(const void *x, void *y) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gx, x, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gy, y, false);
}

void TransposeWh::create_globals() {
    uint32_t size = m_N * m_H * m_W;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_gx = core::Global(m_device, T, size, log2_page_size);
    m_gy = core::Global(m_device, T, size, log2_page_size);
}

void TransposeWh::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
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

void TransposeWh::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void TransposeWh::create_reader() {
    std::string path = m_kernel_base_path + "/transpose_wh_reader.cpp";
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
        pipe<T> px,
        uint32 gx_pos,
        uint32 N,
        uint32 Ht,
        uint32 Wt,
        uint32 HtWt)
*/
    uint32_t Ht = m_H / TILE_HEIGHT;
    uint32_t Wt = m_W / TILE_WIDTH;
    uint32_t HtWt = Ht * Wt;
    std::vector<core::KernelArg> args{
        m_gx,
        m_px,
        uint32_t(0), // gx_pos
        m_N,
        Ht,
        Wt,
        HtWt
    };
    m_reader.set_args(m_grid, args);
}

void TransposeWh::create_writer() {
    std::string path = m_kernel_base_path + "/transpose_wh_writer.cpp";
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
    uint32_t num_tiles = (m_N * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        num_tiles,
    };
    m_writer.set_args(m_grid, args);
}

void TransposeWh::create_math() {
    std::string path = m_kernel_base_path + "/transpose_wh_math.cpp";
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
void kernel(pipe<T> px, pipe<T> py, uint32 NHtWt)
*/
    uint32_t NHtWt = (m_N * m_H * m_W) / TILE_SIZE;
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        NHtWt
    };
    m_math.set_args(m_grid, args);
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

