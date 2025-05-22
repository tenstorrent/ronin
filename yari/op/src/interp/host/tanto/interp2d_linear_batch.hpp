// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/tanto/interp_common.hpp"

namespace ronin {
namespace op {
namespace interp {
namespace tanto {

namespace core = ronin::tanto::host;

class Interp2dLinearBatch {
public:
    Interp2dLinearBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        float scale_h,
        float scale_w,
        CoordTransformMode coord_transform_mode,
        int batch_size);
    ~Interp2dLinearBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
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
    void create_kernels();
    void create_reader();
    void create_writer();
    void create_math();
    void init_locals();
    void compute_locals(std::vector<float> &vw, std::vector<uint32_t> &vp);
    void compute_grid_dims(uint32_t &x, uint32_t &y);
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t DEST_TILES = 8;
private:
    core::Device m_device;
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_C = 0;
    uint32_t m_P = 0;
    uint32_t m_Q = 0;
    float m_scale_h = 0.0f;
    float m_scale_w = 0.0f;
    CoordTransformMode m_coord_transform_mode = CoordTransformMode(0);
    core::Program m_program;
    uint32_t m_x_start = 0;
    uint32_t m_y_start = 0;
    uint32_t m_x_end = 0;
    uint32_t m_y_end = 0;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gw;
    core::Global m_gp;
    core::Global m_gy;
    core::Global m_gzero;
    core::Local m_lx;
    core::Local m_lp;
    core::Local m_lzero;
    core::Pipe m_px;
    core::Pipe m_pw;
    core::Pipe m_py;
    core::Pipe m_px_im;
    core::Pipe m_pw_im;
    core::Pipe m_pt_im;
    core::Pipe m_py_im;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_C_arg = 0;
    uint32_t m_Co = 0;
    uint32_t m_Ci = 0;
    uint32_t m_zero_size = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_pw_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_px_im_frame_size = 0;
    uint32_t m_pw_im_frame_size = 0;
    uint32_t m_pt_im_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace interp
} // namespace op
} // namespace ronin

