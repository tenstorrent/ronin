// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/conv2d_image_batch.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;

//
//    Conv2dImageBatch
//

void Conv2dImageBatch::create_param_bias_reader() {
    std::string path = m_param_kernel_base_path + "/image_batch_bias_reader.cpp";
/*
param<uint32> H;
param<uint32> K;
param<uint32> R;
param<uint32> WC;
param<uint32> PQ;
param<uint32> SC;
param<uint32> RSC_rnd;
param<uint32> before_h;
param<uint32> after_h;
param<uint32> before_wc;
param<uint32> after_wc;
param<uint32> offset_wc;
param<uint32> before_hwc;
param<uint32> delta_p;
param<uint32> delta_q;
param<uint32> delta_r;
param<uint32> end_q;
param<uint32> x_stride;
param<uint32> zero_size;
*/
    uint32_t before_wc = m_before_w * m_C;
    uint32_t after_wc = m_after_w * m_C;
    uint32_t offset_wc = m_offset_w * m_C;
    uint32_t before_hwc = m_before_h * (m_before_w + m_W + m_after_w) * m_C;
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<uint32_t> params{
        m_H,
        m_K,
        m_R,
        m_W * m_C,
        m_P * m_Q,
        m_S * m_C,
        m_RSC_rnd,
        m_before_h,
        m_after_h,
        before_wc,
        after_wc,
        offset_wc,
        before_hwc,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_end_q,
        x_stride,
        m_zero_size
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
        local<T> lx,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 x_pos)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_lx,
        m_lzero,
        m_px,
        m_pb,
        m_N / m_batch_size,
        uint32_t(0)  // [8] x_pos
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[8] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Conv2dImageBatch::create_param_writer() {
    std::string path = m_param_kernel_base_path + "/image_batch_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> KRSC_rnd;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_K * m_RSC_rnd,
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
        uint32 N,
        uint32 y_pos)
*/
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_N / m_batch_size,
        uint32_t(0)  // [5] y_pos
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[5] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dImageBatch::create_param_mcast_writer() {
    std::string path = m_param_kernel_base_path + "/image_batch_mcast_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> KRSC_rnd;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_K * m_RSC_rnd,
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
        uint32 y_pos)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [6] send_mode
        uint32_t(0), // [7] x0
        uint32_t(0), // [8] y0
        uint32_t(0), // [9] x1
        uint32_t(0), // [10] y1
        num_dests,
        m_N / m_batch_size,
        uint32_t(0)  // [13] y_pos
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            uint32_t send_mode = (y == m_y_start) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, m_y_start, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_y_end, x1_phy, y1_phy);
            args[6] = send_mode;
            args[7] = x0_phy;
            args[8] = y0_phy;
            args[9] = x1_phy;
            args[10] = y1_phy;
            args[13] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dImageBatch::create_param_bias_math() {
    std::string path = m_param_kernel_base_path + "/image_batch_bias_math.cpp";
/*
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RSC_rnd;
*/
    std::vector<uint32_t> params{
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_RSC_rnd
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

void Conv2dImageBatch::create_param_bias_unary_math() {
    std::string path = m_param_kernel_base_path + "/image_batch_bias_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RSC_rnd;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<uint32_t> params{
        unary_op_code,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_RSC_rnd,
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

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

