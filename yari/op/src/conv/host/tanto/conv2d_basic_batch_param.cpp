// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/conv2d_basic_batch.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;

//
//    Conv2dBasicBatch
//

void Conv2dBasicBatch::create_param_bias_reader() {
    std::string path = m_param_kernel_base_path + "/basic_batch_bias_reader.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> R;
param<uint32> S;
param<uint32> PQ;
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
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_R,
        m_S,
        m_P * m_Q,
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
        uint32 N,
        uint32 x_pos)
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
        m_N / m_batch_size,
        uint32_t(0)  // [9] x_pos
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[9] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_lx_bias_reader() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lx_bias_reader.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> R;
param<uint32> S;
param<uint32> HWC;
param<uint32> PQ;
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
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_R,
        m_S,
        m_H * m_W * m_C,
        m_P * m_Q,
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
        local<T> lx,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 x_pos)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_gmask,
        m_lx,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_N / m_batch_size,
        uint32_t(0)  // [10] x_pos
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[10] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_pw_bias_reader() {
    std::string path = m_param_kernel_base_path + "/basic_batch_pw_bias_reader.cpp";
/*
param<uint32> HW;
param<uint32> C;
param<uint32> K;
param<uint32> zero_size;
param<uint32> x_stride;
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<uint32_t> params{
        m_H * m_W,
        m_C,
        m_K,
        m_zero_size,
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
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 HW,
        uint32 C,
        uint32 K,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_lzero,
        m_px,
        m_pb,
        m_N / m_batch_size,
        uint32_t(0)  // [7] x_pos
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[7] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_writer.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_P * m_Q,
        m_R * m_S,
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

void Conv2dBasicBatch::create_param_mcast_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_mcast_writer.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_P * m_Q,
        m_R * m_S,
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

void Conv2dBasicBatch::create_param_lw_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RSKC;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S * m_K * m_C,
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
        uint32_t(0) // [5] y_pos
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

void Conv2dBasicBatch::create_param_lw_mcast_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_mcast_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RSKC;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S * m_K * m_C,
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

void Conv2dBasicBatch::create_param_add_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_add_writer.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_P * m_Q,
        m_R * m_S,
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
        uint32 N,
        uint32 y_pos)
*/
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_N / m_batch_size,
        uint32_t(0)  // [7] y_pos
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[7] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_mcast_add_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_mcast_add_writer.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_P * m_Q,
        m_R * m_S,
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
        uint32 y_pos)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [8] send_mode
        uint32_t(0), // [9] x0
        uint32_t(0), // [10] y0
        uint32_t(0), // [11] x1
        uint32_t(0), // [12] y1
        num_dests,
        m_N / m_batch_size,
        uint32_t(0)  // [15] y_pos
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
            args[8] = send_mode;
            args[9] = x0_phy;
            args[10] = y0_phy;
            args[11] = x1_phy;
            args[12] = y1_phy;
            args[15] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_lw_add_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_add_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RSKC;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S * m_K * m_C,
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
        uint32 N,
        uint32 y_pos)
*/
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_N / m_batch_size,
        uint32_t(0)  // [7] y_pos
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[7] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_lw_mcast_add_writer() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_mcast_add_writer.cpp";
/*
param<uint32> K;
param<uint32> PQ;
param<uint32> RSKC;
param<uint32> y_stride;
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<uint32_t> params{
        m_K,
        m_P * m_Q,
        m_R * m_S * m_K * m_C,
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
        uint32 y_pos)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [8] send_mode
        uint32_t(0), // [9] x0
        uint32_t(0), // [10] y0
        uint32_t(0), // [11] x1
        uint32_t(0), // [12] y1
        num_dests,
        m_N / m_batch_size,
        uint32_t(0)  // [15] y_pos
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
            args[8] = send_mode;
            args[9] = x0_phy;
            args[10] = y0_phy;
            args[11] = x1_phy;
            args[12] = y1_phy;
            args[15] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dBasicBatch::create_param_bias_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_bias_math.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
*/
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S
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

void Conv2dBasicBatch::create_param_bias_add_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_bias_add_math.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
*/
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S
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

void Conv2dBasicBatch::create_param_bias_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_bias_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
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

void Conv2dBasicBatch::create_param_bias_add_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_bias_add_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
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

void Conv2dBasicBatch::create_param_lw_bias_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_bias_math.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> RSKC;
*/
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C
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

void Conv2dBasicBatch::create_param_lw_bias_add_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_bias_add_math.cpp";
/*
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> RSKC;
*/
    std::vector<uint32_t> params{
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C
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

void Conv2dBasicBatch::create_param_lw_bias_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_bias_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> RSKC;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C,
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

void Conv2dBasicBatch::create_param_lw_bias_add_unary_math() {
    std::string path = m_param_kernel_base_path + "/basic_batch_lw_bias_add_unary_math.cpp";
/*
param<uint32> unary_op_code;
param<uint32> C;
param<uint32> K;
param<uint32> Ko;
param<uint32> Ki;
param<uint32> PQ;
param<uint32> RS;
param<uint32> RSKC;
param<uint32> unary_param0;
*/
    uint32_t unary_op_code = get_unary_op_code();
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<uint32_t> params{
        unary_op_code,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C,
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

