// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

class Conv2dImageBatch {
public:
    Conv2dImageBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int K,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        const base::PostOpSpec &post_op,
        int batch_size);
    ~Conv2dImageBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gz,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void init_options();
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
    void create_bias_unary_math();
    void create_param_bias_reader();
    void create_param_writer();
    void create_param_mcast_writer();
    void create_param_bias_math();
    void create_param_bias_unary_math();
    void init_locals();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
    std::string get_unary_kernel_suffix();
    uint32_t get_unary_op_code();
    uint32_t encode_unary_param0();
    bool is_unary_relu6();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t DEST_TILES = 8;
    static const uint32_t 
        OPT_BIAS = 0x1,
        OPT_ADD = 0x2,
        OPT_UNARY = 0x4;
private:
    core::Device m_device;
    uint32_t m_batch_size = 0;
    uint32_t m_N = 0;
    uint32_t m_H = 0;
    uint32_t m_W = 0;
    uint32_t m_C = 0;
    uint32_t m_P = 0;
    uint32_t m_Q = 0;
    uint32_t m_K = 0;
    uint32_t m_R = 0;
    uint32_t m_S = 0;
    uint32_t m_pad_h = 0;
    uint32_t m_pad_w = 0;
    uint32_t m_stride_h = 0;
    uint32_t m_stride_w = 0;
    uint32_t m_dilation_h = 0;
    uint32_t m_dilation_w = 0;
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
    core::Global m_gz;  // reserved
    core::Global m_gy;
    core::Global m_gzero;
    core::Local m_lx;
    core::Local m_lzero;
    core::Pipe m_px;
    core::Pipe m_pw;
    core::Pipe m_pb;
    core::Pipe m_pz;    // reserved
    core::Pipe m_py;
    core::Pipe m_px_im;
    core::Pipe m_pz_im; // reserved
    core::Pipe m_py_im;
    core::Semaphore m_sem_send;
    core::Semaphore m_sem_recv;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_before_h = 0;
    uint32_t m_after_h = 0;
    uint32_t m_before_w = 0;
    uint32_t m_after_w = 0;
    uint32_t m_offset_w = 0;
    uint32_t m_W_arg = 0;
    uint32_t m_C_arg = 0;
    uint32_t m_K_arg = 0;
    uint32_t m_S_arg = 0;
    uint32_t m_RSC_rnd = 0;
    uint32_t m_Ko = 0;
    uint32_t m_Ki = 0;
    uint32_t m_zero_size = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_pw_frame_size = 0;
    uint32_t m_pb_frame_size = 0;
    uint32_t m_pz_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_px_im_frame_size = 0;
    uint32_t m_pz_im_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    uint32_t m_delta_p = 0;
    uint32_t m_delta_q = 0;
    uint32_t m_delta_r = 0;
    uint32_t m_end_q = 0;
    base::PostOpSpec m_post_op;
    uint32_t m_options = 0;
    uint32_t m_enable_c4 = false;
    bool m_enable_param_kernels = false;
    std::string m_metal_kernel_base_path;
    std::string m_param_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

