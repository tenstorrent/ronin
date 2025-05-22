// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/load_dist.hpp"

namespace ronin {
namespace op {
namespace move {
namespace tanto {

namespace core = ronin::tanto::host;
namespace util = ronin::op::common::util;

//
//    LoadDist
//

LoadDist::LoadDist(
        int N,
        int H,
        int C,
        int batch_size):
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_C(uint32_t(C)),
            m_batch_size(uint32_t(batch_size)) { 
    m_H_arg = m_H;
    m_C_arg = m_C;
    m_H = u32_align(m_H, 32);
    m_C = u32_align(m_C, 32);
}

LoadDist::~LoadDist() { }

void LoadDist::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Local &ly) {
    m_device = device;
    m_gx = gx;
    m_ly = ly;

    // ACHTUNG: Supported batch sizes are Wormhole-specific
    assert(m_batch_size == 8 || m_batch_size == 16);
    assert(m_N == m_batch_size);

    assert(m_H % 32 == 0);
    assert(m_C % 32 == 0);

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    compute_grid_dims(m_grid_x, m_grid_y);

    m_grid = core::Grid(m_program, 0, 0, m_grid_x - 1, m_grid_y - 1);
    m_block_size = (m_grid_x * m_grid_y) / m_batch_size;

    assert(m_N % m_block_size == 0);

    m_kernel_base_path = "op/move/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_kernels();
}

void LoadDist::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int LoadDist::input_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

int LoadDist::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

std::vector<float> LoadDist::transform_input(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::pad(x, m_N, m_H_arg, m_C_arg, m_N, m_H, m_C);
}

std::vector<float> LoadDist::transform_output(int index, const std::vector<float> &x) {
    // UNUSED
    assert(index == 0);
    return util::unpad(x, m_N, m_H, m_C, m_N, m_H_arg, m_C_arg);
}

void LoadDist::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_ly.is_null());
    assert(m_gx.bytes() <= input_volume(0) * item_bytes);
    assert(m_ly.size() == output_volume(0) / m_block_size);
}

void LoadDist::create_kernels() {
    create_reader();
}

void LoadDist::create_reader() {
    std::string path = m_kernel_base_path + "/load_dist_reader.cpp";
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
        local<T> ly,
        uint32 xpos,
        uint32 ypos,
        uint32 H,
        uint32 C,
        uint32 Cb)
*/
    uint32_t Cb = m_C / m_block_size;
    std::vector<core::KernelArg> args{
        m_gx,
        m_ly,
        uint32_t(0), // [2] xpos
        uint32_t(0), // [3] ypos (reserved)
        m_H,
        m_C,
        Cb
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_block_size;
            uint32_t qy = y % m_block_size;
            uint32_t xpos = (py * m_grid_x + x) * m_H * m_C + qy * Cb;
            args[2] = xpos;
            m_reader.set_args(x, y, args);
        }
    }
}

void LoadDist::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    x = 8;
    y = 8;
}

} // namespace tanto
} // namespace move
} // namespace op
} // namespace ronin

