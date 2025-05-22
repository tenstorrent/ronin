// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

namespace ronin {
namespace op {
namespace move {
namespace tanto {

namespace core = ronin::tanto::host;

class StoreDist {
public:
    StoreDist(
        int N,
        int H,
        int C,
        int batch_size);
    ~StoreDist();
public:
    void init(
        const core::Device &device,
        const core::Local &lx,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void validate_globals();
    void create_kernels();
    void create_writer();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
private:
    core::Device m_device;
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_C = 0;
    core::Program m_program;
    uint32_t m_grid_x = 0;
    uint32_t m_grid_y = 0;
    uint32_t m_block_size = 0;
    core::Grid m_grid;
    core::Local m_lx;
    core::Global m_gy;
    core::Kernel m_writer;
    uint32_t m_H_arg = 0;
    uint32_t m_C_arg = 0;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace move
} // namespace op
} // namespace ronin

