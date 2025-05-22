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
namespace fc {
namespace tanto {

namespace core = ronin::tanto::host;

class FCBatch {
public:
    FCBatch(
        int N,
        int H,
        int C,
        int K,
        int batch_size);
    ~FCBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void validate_globals();
    void create_globals();
    void create_locals();
    void create_pipes();
    void create_semaphores();
    void create_kernels();
    void create_bias_reader();
    void create_writer();
    void create_mcast_writer();
    void create_bias_math();
    void init_locals();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t DEST_TILES = 8;
private:
    core::Device m_device;
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_C = 0;
    uint32_t m_K = 0;
    core::Program m_program;
    uint32_t m_x_start = 0;
    uint32_t m_y_start = 0;
    uint32_t m_x_end = 0;
    uint32_t m_y_end = 0;
    uint32_t m_x_start_phy = 0;
    uint32_t m_y_start_phy = 0;
    uint32_t m_x_end_phy = 0;
    uint32_t m_y_end_phy = 0;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gw;
    core::Global m_gb;
    core::Global m_gy;
    core::Global m_gzero;
    core::Local m_lzero;
    core::Pipe m_px;
    core::Pipe m_pw;
    core::Pipe m_pb;
    core::Pipe m_py;
    core::Pipe m_px_im;
    core::Pipe m_py_im;
    core::Semaphore m_sem_send;
    core::Semaphore m_sem_recv;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_zero_size = 0;
    uint32_t m_H_arg = 0;
    uint32_t m_K_arg = 0;
    uint32_t m_Ko = 0;
    uint32_t m_Ki = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_pw_frame_size = 0;
    uint32_t m_pb_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_px_im_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace fc
} // namespace op
} // namespace ronin

