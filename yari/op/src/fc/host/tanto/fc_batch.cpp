// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/fc_batch.hpp"

namespace ronin {
namespace op {
namespace fc {
namespace tanto {

namespace core = ronin::tanto::host;
namespace util = ronin::op::common::util;

//
//    FCBatch
//

FCBatch::FCBatch(
        int N,
        int H,
        int C,
        int K,
        int batch_size):
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_C(uint32_t(C)),
            m_K(uint32_t(K)),
            m_batch_size(uint32_t(batch_size)) { 
    // these values are required to compute input/output volumes and shapes
    m_H_arg = m_H;
    m_K_arg = m_K;
    m_H = u32_align(m_H, 32);
    if (m_K > DEST_TILES * 32) {
        m_Ki = DEST_TILES;
        m_Ko = (m_K + DEST_TILES * 32 - 1) / (DEST_TILES * 32);
    } else {
        m_Ki = (m_K + 31) / 32;
        m_Ko = 1;
    }
    m_K = m_Ki * m_Ko * 32;            
}

FCBatch::~FCBatch() { }

void FCBatch::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gy) {
    m_device = device;
    m_gx = gx;
    m_gw = gw;
    m_gb = gb;
    m_gy = gy;

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    // TODO: Implement alignment of m_C as in conv2d?
    assert(m_C % 32 == 0);

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    m_x_start = 0;
    m_y_start = 0;
    m_x_end = grid_x - 1;
    m_y_end = grid_y - 1;

    m_device.worker_core_from_logical_core(
        m_x_start, 
        m_y_start, 
        m_x_start_phy, 
        m_y_start_phy);
    m_device.worker_core_from_logical_core(
        m_x_end, 
        m_y_end,
        m_x_end_phy,
        m_y_end_phy);

    m_grid = core::Grid(m_program, m_x_start, m_y_start, m_x_end, m_y_end);

    m_zero_size = m_C;

    uint32_t Ct = m_C / 32;
    uint32_t Kt = m_K / 32;    

    m_px_frame_size = Ct;
    m_pw_frame_size = Ct;
    m_pb_frame_size = Kt;
    m_py_frame_size = Kt;
    m_px_im_frame_size = Ct;
    m_py_im_frame_size = Kt;

    m_kernel_base_path = "op/fc/device/metal";
    m_defines = {{"T", "bfloat16"}};

    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_semaphores();
    create_kernels();

    init_locals();
}

void FCBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int FCBatch::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * m_H * m_C;
    case 1:
        return m_K * m_C;
    case 2:
        return m_K * 32;
    default:
        assert(false);
        return 0;
    }
}

int FCBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_K;
}

std::vector<float> FCBatch::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        y = util::pad(x, m_N, m_H_arg, m_C, m_N, m_H, m_C);
        break;
    case 1:
        y = util::pad(x, m_K_arg, m_C, m_K, m_C);
        y = util::tilize(y, m_K, m_C);
        y = util::make_faces(y);
        break;
    case 2:
        y = util::pad(x, 1, m_K_arg, 32, m_K);
        y = util::tilize(y, 32, m_K);
        y = util::make_faces(y);
        break;
    default:
        assert(false);
        break;
    }
    return y;
}

std::vector<float> FCBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, m_H, m_K, m_N, m_H_arg, m_K_arg);
}

void FCBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gw.is_null());
    // bias is required in this implementation
    assert(!m_gb.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gw.bytes() == input_volume(1) * item_bytes);
    assert(m_gb.bytes() == input_volume(2) * item_bytes);
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void FCBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
}

void FCBatch::create_locals() {
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
}

void FCBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    m_pw =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pw_frame_size * 2,
            m_pw_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pb_frame_size * 2,
            m_pb_frame_size);
    m_py =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_py_frame_size * 2,
            m_py_frame_size);
    m_px_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_px_im_frame_size * 2,
            m_px_im_frame_size);
    m_py_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_py_im_frame_size * 2,
            m_py_im_frame_size);
}

void FCBatch::create_semaphores() {
    m_sem_send = core::Semaphore(m_program, m_grid, 0);
    m_sem_recv = core::Semaphore(m_program, m_grid, 0);
}

void FCBatch::create_kernels() {
    create_bias_reader();
    if (m_batch_size <= 8) {
        create_writer();
    } else {
        create_mcast_writer();
    }
    create_bias_math();
}

void FCBatch::create_bias_reader() {
    std::string path = m_kernel_base_path + "/fc_batch_bias_reader.cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
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
        uint32 H,
        uint32 C,
        uint32 K,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    uint32_t x_stride = m_batch_size * m_H * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_lzero,
        m_px,
        m_pb,
        m_N / m_batch_size,
        m_H_arg,
        m_C,
        m_K,
        m_zero_size,
        uint32_t(0), // [11] x_pos
        x_stride
    };
    uint32_t x_inc = m_H * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[11] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void FCBatch::create_writer() {
    std::string path = m_kernel_base_path + "/fc_batch_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        uint32 N,
        uint32 H,
        uint32 C,
        uint32 K,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t y_stride = (m_batch_size - 1) * m_H * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_N / m_batch_size,
        m_H_arg,
        m_C,
        m_K,
        uint32_t(0), // [8] y_pos
        y_stride
    };
    uint32_t y_inc = m_H * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[8] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void FCBatch::create_mcast_writer() {
    std::string path = m_kernel_base_path + "/fc_batch_mcast_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel_send(
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
        uint32 H,
        uint32 C,
        uint32 K,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    uint32_t y_stride = (m_batch_size - 1) * m_H * m_K;
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
        m_H_arg,
        m_C,
        m_K,
        uint32_t(0), // [16] y_pos
        y_stride
    };
    uint32_t y_inc = m_H * m_K;
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
            args[16] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void FCBatch::create_bias_math() {
    std::string path = m_kernel_base_path + "/fc_batch_bias_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 H,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_H_arg,
        m_C,
        m_K,
        m_Ko,
        m_Ki
    };
    m_math.set_args(m_grid, args);
}

void FCBatch::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
}

void FCBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

} // namespace tanto
} // namespace fc
} // namespace op
} // namespace ronin

