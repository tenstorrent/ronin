// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/conv2d_basic_split.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;

//
//    Conv2dBasicSplit
//

void Conv2dBasicSplit::create_param_bias_reader() {
    std::string path = m_param_kernel_base_path + "/basic_split_bias_reader.cpp";
/*
param<uint32> C;
param<uint32> R;
param<uint32> S;
param<uint32> PQ;
param<uint32> Kb;
param<uint32> start_p;
param<uint32> start_q;
param<uint32> delta_p;
param<uint32> delta_q;
param<uint32> delta_r;
param<uint32> delta_s;
param<uint32> end_q;
param<uint32> zero_size;
param<uint32> mask_size;
param<uint32> x_stride;
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<uint32_t> params{
        m_C,
        m_R,
        m_S,
        m_P * m_Q,
        Kb,
        m_start_p,
        m_start_q,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_delta_s,
        m_end_q,
        m_zero_size,
        m_mask_size,
        x_stride
    };
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 x_pos,
        uint32 b_pos)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_gmask,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_sem_send_x,
        m_sem_recv_x,
        uint32_t(0), // [10] send_mode
        uint32_t(0), // [11] x0
        uint32_t(0), // [12] y0
        uint32_t(0), // [13] x1
        uint32_t(0), // [14] y1
        m_block_size - 1,
        m_N / m_batch_size,
        uint32_t(0), // [17] x_pos
        uint32_t(0)  // [18] b_pos
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_block_size;
            uint32_t qy = y % m_block_size;
            uint32_t y0 = y - qy;
            uint32_t y1 = y0 + m_block_size - 1;
            uint32_t send_mode = (qy == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, y0, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, y1, x1_phy, y1_phy);
            uint32_t x_pos = (py * m_grid_x + x) * HW_rnd * m_C;
            uint32_t b_pos = qy * Kb * 32;
            args[10] = send_mode;
            args[11] = x0_phy;
            args[12] = y0_phy;
            args[13] = x1_phy;
            args[14] = y1_phy;
            args[17] = x_pos;
            args[18] = b_pos;
            m_reader.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_param_writer() {
    std::string path = m_param_kernel_base_path + "/basic_split_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> KC;
param<uint32> Kb;
param<uint32> KbC;
param<uint32> RSKbC;
param<uint32> y_stride;
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S,
        m_K * m_C,
        Kb,
        Kb * m_C,
        m_R * m_S * Kb * m_C,
        y_stride
    };
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 w_pos,
        uint32 y_pos)
*/
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_sem_send_w,
        m_sem_recv_w,
        uint32_t(0), // [6] send_mode
        uint32_t(0), // [7] x0
        uint32_t(0), // [8] y0
        uint32_t(0), // [9] x1
        uint32_t(0), // [10] y1
        m_grid_x - 1,
        m_N / m_batch_size,
        uint32_t(0), // [13] w_pos
        uint32_t(0)  // [14] y_pos
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_block_size;
            uint32_t qy = y % m_block_size;
            uint32_t send_mode = (x == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(0, y, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(m_grid_x - 1, y, x1_phy, y1_phy);
            uint32_t w_pos = qy * Kb * m_C;
            uint32_t y_pos = (py * m_grid_x + x) * PQ_rnd * m_K + qy * Kb;
            args[6] = send_mode;
            args[7] = x0_phy;
            args[8] = y0_phy;
            args[9] = x1_phy;
            args[10] = y1_phy;
            args[13] = w_pos;
            args[14] = y_pos;
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_param_add_writer() {
    std::string path = m_param_kernel_base_path + "/basic_split_add_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> KC;
param<uint32> Kb;
param<uint32> KbC;
param<uint32> RSKbC;
param<uint32> y_stride;
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S,
        m_K * m_C,
        Kb,
        Kb * m_C,
        m_R * m_S * Kb * m_C,
        y_stride
    };
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gz,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pz,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 w_pos,
        uint32 y_pos)
*/
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_sem_send_w,
        m_sem_recv_w,
        uint32_t(0), // [8] send_mode
        uint32_t(0), // [9] x0
        uint32_t(0), // [10] y0
        uint32_t(0), // [11] x1
        uint32_t(0), // [12] y1
        m_grid_x - 1,
        m_N / m_batch_size,
        uint32_t(0), // [15] w_pos
        uint32_t(0)  // [16] y_pos
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_block_size;
            uint32_t qy = y % m_block_size;
            uint32_t send_mode = (x == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(0, y, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(m_grid_x - 1, y, x1_phy, y1_phy);
            uint32_t w_pos = qy * Kb * m_C;
            uint32_t y_pos = (py * m_grid_x + x) * PQ_rnd * m_K + qy * Kb;
            args[8] = send_mode;
            args[9] = x0_phy;
            args[10] = y0_phy;
            args[11] = x1_phy;
            args[12] = y1_phy;
            args[15] = w_pos;
            args[16] = y_pos;
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_param_bias_math() {
    std::string path = m_param_kernel_base_path + "/basic_split_bias_math.cpp";
/*
param<uint32> C;
param<uint32> PQ;
param<uint32> RS;
param<uint32> Kb;
param<uint32> RSKbC;
*/
    uint32_t Kb = m_K / m_block_size;
    std::vector<uint32_t> params{
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C
    };
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_param_bias_add_math() {
    std::string path = m_param_kernel_base_path + "/basic_split_bias_add_math.cpp";
/*
param<uint32> C;
param<uint32> PQ;
param<uint32> RS;
param<uint32> Kb;
param<uint32> RSKbC;
*/
    uint32_t Kb = m_K / m_block_size;
    std::vector<uint32_t> params{
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C
    };
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_param_bias_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_split_bias_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> PQ;
param<uint32> RS;
param<uint32> Kb;
param<uint32> RSKbC;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    uint32_t Kb = m_K / m_block_size;
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C,
        unary_param0
    };
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_param_bias_add_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_split_bias_add_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> PQ;
param<uint32> RS;
param<uint32> Kb;
param<uint32> RSKbC;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    uint32_t Kb = m_K / m_block_size;
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C,
        unary_param0
    };
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::TANTO,
            path, 
            params, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size
    };
    m_math.set_args(m_grid, args);
}

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

