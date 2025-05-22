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
#include "host/tanto/unpack_untilize.hpp"

//
//    Basic unpack untilize operation implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/unpack_untilize_reader.cpp >metal/unpack_untilize_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/unpack_untilize_writer.cpp >metal/unpack_untilize_writer.cpp
//    tanto --mode=compute -DT=bfloat16 \
//        tanto/unpack_untilize_math.cpp >metal/unpack_untilize_math.cpp
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    UnpackUntilize
//

UnpackUntilize::UnpackUntilize() { }

UnpackUntilize::~UnpackUntilize() { }

void UnpackUntilize::init(
        const core::Device &device,
        int H,
        int W) {
    assert(H % TILE_HEIGHT == 0);
    assert(W % TILE_WIDTH == 0);

    m_device = device;
    m_H = uint32_t(H);
    m_W = uint32_t(W);
    m_num_blocks = m_H / TILE_HEIGHT;
    m_block_tiles = m_W / TILE_WIDTH;
    m_pipe_frame_size = m_block_tiles;

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void UnpackUntilize::run(const void *x, void *y) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gx, x, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gy, y, false);
}

void UnpackUntilize::create_globals() {
    uint32_t size = m_H * m_W;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_gx = core::Global(m_device, T, size, log2_page_size);
    m_gy = core::Global(m_device, T, size, log2_page_size);
}

void UnpackUntilize::create_pipes() {
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

void UnpackUntilize::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void UnpackUntilize::create_reader() {
    std::string path = m_kernel_base_path + "/unpack_untilize_reader.cpp";
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
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_px,
        uint32_t(0), // gx_pos
        m_num_blocks,
        m_block_tiles,    
    };
    m_reader.set_args(m_grid, args);
}

void UnpackUntilize::create_writer() {
    std::string path = m_kernel_base_path + "/unpack_untilize_writer.cpp";
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
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        m_num_blocks,
        m_block_tiles
    };
    m_writer.set_args(m_grid, args);
}

void UnpackUntilize::create_math() {
    std::string path = m_kernel_base_path + "/unpack_untilize_math.cpp";
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
        pipe<T> py,
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        m_num_blocks,
        m_block_tiles
    };
    m_math.set_args(m_grid, args);
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

