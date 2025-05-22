// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>

#include "host/core/api.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/interp_common.hpp"
#include "host/tanto/interp2d_linear_batch.hpp"

namespace ronin {
namespace op {
namespace interp {
namespace tanto {

namespace core = ronin::tanto::host;
namespace util = ronin::op::common::util;

namespace {

uint32_t u32_align(uint32_t a, uint32_t b) {
    return ((a + b - 1) / b) * b;
}

uint32_t get_item_bytes(core::DataFormat data_format) {
    switch (data_format) {
    case core::DataFormat::UINT32:
        return 4;
    case core::DataFormat::FLOAT32:
        return 4;
    case core::DataFormat::BFLOAT16:
        return 2;
    default:
        assert(false);
        return 0;
    }
}

} // namespace

//
//    Interp2dLinearBatch
//

Interp2dLinearBatch::Interp2dLinearBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        float scale_h,
        float scale_w,
        CoordTransformMode coord_transform_mode,
        int batch_size):
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_W(uint32_t(W)),
            m_C(uint32_t(C)),
            m_P(uint32_t(P)),
            m_Q(uint32_t(Q)),
            m_scale_h(scale_h),
            m_scale_w(scale_w),
            m_coord_transform_mode(coord_transform_mode),
            m_batch_size(uint32_t(batch_size)) { 
    // these values are required to compute input/output volumes and shapes
    m_C_arg = m_C;
    m_C = u32_align(m_C, 32);            
}

Interp2dLinearBatch::~Interp2dLinearBatch() { }

void Interp2dLinearBatch::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gy) {
    m_device = device;
    m_gx = gx;
    m_gy = gy;

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    assert(m_C % 32 == 0);

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    m_x_start = 0;
    m_y_start = 0;
    m_x_end = grid_x - 1;
    m_y_end = grid_y - 1;

    m_grid = core::Grid(m_program, m_x_start, m_y_start, m_x_end, m_y_end);

    m_zero_size = m_C;

    uint32_t Ct = m_C / 32;
    if (Ct <= DEST_TILES) {
        m_Ci = Ct;
    } else if (Ct % DEST_TILES == 0) {
        m_Ci = DEST_TILES;
    } else {
        // TODO: Implement better rule?
        m_Ci = 1;
    }
    m_Co = Ct / m_Ci;
    assert(m_Ci * m_Co == Ct);

    m_px_frame_size = Ct;
    m_pw_frame_size = 4;
    m_py_frame_size = Ct;
    m_px_im_frame_size = Ct;
    m_pw_im_frame_size = 4;
    m_pt_im_frame_size = Ct;
    m_py_im_frame_size = Ct;

    m_kernel_base_path = "op/interp/device/metal";
    m_defines = {{"T", "bfloat16"}};

    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_kernels();

    init_locals();
}

void Interp2dLinearBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int Interp2dLinearBatch::input_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_H * m_W, 32) * m_C;
}

int Interp2dLinearBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_C;
}

std::vector<float> Interp2dLinearBatch::transform_input(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
}

std::vector<float> Interp2dLinearBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_C, m_N, m_P * m_Q, m_C_arg);
}

void Interp2dLinearBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void Interp2dLinearBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    m_gw = core::Global(m_device, T, PQ_rnd * 4, log2_page_size);
    m_gp = core::Global(m_device, core::DataFormat::UINT32, PQ_rnd * 4, log2_page_size);
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
}

void Interp2dLinearBatch::create_locals() {
    m_lx = core::Local(m_program, m_grid, T, u32_align(m_H * m_W, 32) * m_C);
    m_lp = core::Local(m_program, m_grid, core::DataFormat::UINT32, 128);
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
}

void Interp2dLinearBatch::create_pipes() {
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
    m_pw_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_pw_im_frame_size * 2,
            m_pw_im_frame_size);
    m_pt_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_pt_im_frame_size * 2,
            m_pt_im_frame_size);
    m_py_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_py_im_frame_size * 2,
            m_py_im_frame_size);
}

void Interp2dLinearBatch::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void Interp2dLinearBatch::create_reader() {
    std::string path = m_kernel_base_path + "/interp2d_linear_batch_reader.cpp";
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
        global<T> gw,
        global<uint32> gp,
        global<T> gzero,
        local<T> lx,
        local<uint32> lp,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pw,
        uint32 N,
        uint32 C,
        uint32 HWC,
        uint32 PQ,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gw,
        m_gp,
        m_gzero,
        m_lx,
        m_lp,
        m_lzero,
        m_px,
        m_pw,
        m_N / m_batch_size,
        m_C,
        HW_rnd * m_C,
        m_P * m_Q,
        m_zero_size,
        uint32_t(0), // [14] x_pos
        x_stride
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[14] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Interp2dLinearBatch::create_writer() {
    std::string path = m_kernel_base_path + "/interp2d_linear_batch_writer.cpp";
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
        global<T> gy,
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        uint32_t(0), // [5] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_C;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[5] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Interp2dLinearBatch::create_math() {
    std::string path = m_kernel_base_path + "/interp2d_linear_batch_math.cpp";
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
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pw_im,
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 Co,
        uint32 Ci,
        uint32 PQ)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_py,
        m_px_im,
        m_pw_im,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_Co,
        m_Ci,
        m_P * m_Q
    };
    m_math.set_args(m_grid, args);
}

void Interp2dLinearBatch::init_locals() {
    std::vector<float> vw(m_gw.bytes() / get_item_bytes(T), 0.0f);
    std::vector<uint32_t> vp(m_gp.bytes() / sizeof(uint32_t), 0);
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    compute_locals(vw, vp);
    std::vector<uint16_t> tvw = util::float_to_u16b(vw);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
    queue.enqueue_write(m_gw, tvw.data(), true);
    queue.enqueue_write(m_gp, vp.data(), true);
}

void Interp2dLinearBatch::compute_locals(std::vector<float> &vw, std::vector<uint32_t> &vp) {
    std::vector<uint32_t> in_h1(m_P);
    std::vector<uint32_t> in_h2(m_P);
    std::vector<uint32_t> in_w1(m_Q);
    std::vector<uint32_t> in_w2(m_Q);
    std::vector<float> dh1(m_P);
    std::vector<float> dh2(m_P);
    std::vector<float> dw1(m_Q);
    std::vector<float> dw2(m_Q);

    for (uint32_t p = 0; p < m_P; p++) {
        float in_h = get_input_coord(m_coord_transform_mode, p, m_scale_h, m_P, m_H);
        in_h = std::max(0.0f, std::min(in_h, float(m_H - 1)));
        in_h1[p] = std::min(uint32_t(in_h), m_H - 1);
        in_h2[p] = std::min(in_h1[p] + 1, m_H - 1);
        dh1[p] = std::abs(in_h - in_h1[p]);
        dh2[p] = std::abs(in_h - in_h2[p]);
        if (in_h1[p] == in_h2[p]) {
            dh1[p] = 0.5f;
            dh2[p] = 0.5f;
        }
    }

    for (uint32_t q = 0; q < m_Q; q++) {
        float in_w = get_input_coord(m_coord_transform_mode, q, m_scale_w, m_Q, m_W);
        in_w = std::max(0.0f, std::min(in_w, float(m_W - 1)));
        in_w1[q] = std::min(uint32_t(in_w), m_W - 1);
        in_w2[q] = std::min(in_w1[q] + 1, m_W - 1);
        dw1[q] = std::abs(in_w - in_w1[q]);
        dw2[q] = std::abs(in_w - in_w2[q]);
        if (in_w1[q] == in_w2[q]) {
            dw1[q] = 0.5f;
            dw2[q] = 0.5f;
        }
    }

    uint32_t WC = m_W * m_C;
    uint32_t PQ = m_P * m_Q;

    uint32_t iw = 0;
    for (uint32_t pq_start = 0; pq_start < PQ; pq_start += 32) {
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t pq = pq_start + i;
            uint32_t p = pq / m_Q;
            uint32_t q = pq % m_Q;
            float w11, w21, w12, w22;
            if (p < m_P) {
                w11 = dh2[p] * dw2[q];
                w21 = dh1[p] * dw2[q];
                w12 = dh2[p] * dw1[q];
                w22 = dh1[p] * dw1[q];
            } else {
                w11 = 0.0f;
                w21 = 0.0f;
                w12 = 0.0f;
                w22 = 0.0f;
            }
            vw[iw] = w11;
            vw[iw + 32] = w21;
            vw[iw + 2 * 32] = w12;
            vw[iw + 3 * 32] = w22;
            iw++;
        }
        iw += 3 * 32;
    }

    uint32_t pq_tail_mark = ~uint32_t(0);
    uint32_t ip = 0;
    for (uint32_t pq_start = 0; pq_start < PQ; pq_start += 32) {
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t pq = pq_start + i;
            uint32_t p = pq / m_Q;
            uint32_t q = pq % m_Q;
            uint32_t ix11, ix21, ix12, ix22;
            if (p < m_P) {
                ix11 = in_h1[p] * WC + in_w1[q] * m_C;
                ix21 = in_h1[p] * WC + in_w2[q] * m_C;
                ix12 = in_h2[p] * WC + in_w1[q] * m_C;
                ix22 = in_h2[p] * WC + in_w2[q] * m_C;
            } else {
                ix11 = pq_tail_mark;
                ix21 = pq_tail_mark;
                ix12 = pq_tail_mark;
                ix22 = pq_tail_mark;
            }
            vp[ip] = ix11;
            vp[ip + 32] = ix21;
            vp[ip + 2 * 32] = ix12;
            vp[ip + 3 * 32] = ix22;
            ip++;
        }
        ip += 3 * 32;
    }
}

void Interp2dLinearBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
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
} // namespace interp
} // namespace op
} // namespace ronin

