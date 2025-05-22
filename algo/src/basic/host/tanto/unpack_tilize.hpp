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

class UnpackTilize {
public:
    UnpackTilize();
    ~UnpackTilize();
public:
    void init(
        const core::Device &device,
        int H,
        int W);
    void run(const void *x, void *y);
private:
    void create_globals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_writer();
    void create_math();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t 
        TILE_HEIGHT = 32,
        TILE_WIDTH = 32;
private:
    core::Device m_device;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_num_blocks = 0;
    uint32_t m_block_tiles = 0;
    uint32_t m_pipe_frame_size = 0;
    core::Program m_program;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gy;
    core::Pipe m_px;
    core::Pipe m_py;
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

