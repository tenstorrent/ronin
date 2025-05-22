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

class MatmulMulti {
public:
    MatmulMulti();
    ~MatmulMulti();
public:
    void init(
        const core::Device &device,
        int batch,
        int M,
        int N,
        int K);
    void run(
        const void *a,
        const void *b,
        void *c);
private:
    void setup_grids();
    void create_globals();
    void create_pipes();
    void create_kernels();
    void create_reader_writer();
    void setup_reader(
        uint32_t x,
        uint32_t y,
        uint32_t out_tile_pos,
        uint32_t out_num_tiles);
    void setup_writer(
        uint32_t x,
        uint32_t y,
        uint32_t out_tile_pos,
        uint32_t out_num_tiles);
    void create_math();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t 
        TILE_SIZE = 1024,
        TILE_DIM = 32;
private:
    core::Device m_device;
    uint32_t m_batch = 0;
    uint32_t m_M = 0;
    uint32_t m_N = 0;
    uint32_t m_K = 0;
    uint32_t m_pipe_frame_size = 0;
    core::Program m_program;
    uint32_t m_workers_x = 0;
    uint32_t m_workers_y = 0;
    core::Grid m_grid;
    core::Grid m_grid1;
    core::Grid m_grid2;
    uint32_t m_all_cores = 0;
    uint32_t m_grid1_out_tiles = 0;
    uint32_t m_grid2_out_tiles = 0;
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

