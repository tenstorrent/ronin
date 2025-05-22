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
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

//
//    Depthwise separable convolution
//    Depthwise and pointwise convolutions fused in one operation
//

class DSConv2dBatch {
public:
    DSConv2dBatch(
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
    ~DSConv2dBatch();
public:
    void init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gw2,
        const core::Global &gb2,
        const core::Global &gz,
        const core::Global &gy);
    void run();
    int input_volume(int index);
    int output_volume(int index);
    std::vector<float> transform_input(int index, const std::vector<float> &x);
    std::vector<float> transform_output(int index, const std::vector<float> &x);
private:
    void init_config();
    void validate_globals();
    void create_globals();
    void create_locals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_lx_reader();
    void create_writer();
    void create_add_writer();
    void create_math();
    void create_add_math();
    void init_locals();
    void compute_grid_dims(uint32_t &x, uint32_t &y);
    void compute_mask(std::vector<uint32_t> &vmask);
    std::string get_unary_kernel_suffix();
    uint32_t encode_unary_param0();
    bool is_unary_relu6();
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
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gw;
    core::Global m_gb;
    core::Global m_gw2;
    core::Global m_gb2;
    core::Global m_gz;
    core::Global m_gy;
    core::Global m_gzero;
    core::Global m_gmask;
    core::Local m_lx;
    core::Local m_lzero;
    core::Local m_lmask;
    core::Pipe m_px;
    core::Pipe m_pw;
    core::Pipe m_pb;
    core::Pipe m_pw2;
    core::Pipe m_pb2;
    core::Pipe m_pz;
    core::Pipe m_py;
    core::Pipe m_pu_im;
    core::Pipe m_pt_im;
    core::Pipe m_pz_im;
    core::Pipe m_py_im;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    uint32_t m_C_arg = 0;
    uint32_t m_K_arg = 0;
    uint32_t m_Co = 0;
    uint32_t m_Ci = 0;
    uint32_t m_Ko = 0;
    uint32_t m_Ki = 0;
    uint32_t m_zero_size = 0;
    uint32_t m_mask_size = 0;
    uint32_t m_px_frame_size = 0;
    uint32_t m_pw_frame_size = 0;
    uint32_t m_pb_frame_size = 0;
    uint32_t m_pw2_frame_size = 0;
    uint32_t m_pb2_frame_size = 0;
    uint32_t m_pz_frame_size = 0;
    uint32_t m_py_frame_size = 0;
    uint32_t m_pu_im_frame_size = 0;
    uint32_t m_pt_im_frame_size = 0;
    uint32_t m_pz_im_frame_size = 0;
    uint32_t m_py_im_frame_size = 0;
    uint32_t m_start_p = 0;
    uint32_t m_start_q = 0;
    uint32_t m_end_q = 0;
    uint32_t m_delta_p = 0;
    uint32_t m_delta_q = 0;
    uint32_t m_delta_r = 0;
    uint32_t m_delta_s = 0;
    base::PostOpSpec m_post_op;
    bool m_cache_lx = false;
    bool m_opt_add = false;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

