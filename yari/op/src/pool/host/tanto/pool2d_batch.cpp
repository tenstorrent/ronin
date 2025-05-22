// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <limits>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/pool2d_batch.hpp"

namespace ronin {
namespace op {
namespace pool {
namespace tanto {

namespace core = ronin::tanto::host;
namespace util = ronin::op::common::util;

namespace {

uint16_t float_to_u16b(float x) {
    union U32 {
        float f;
        uint32_t i;
    } u32;
    u32.f = x;
    return uint16_t(u32.i >> 16);
}

uint32_t float_to_u32(float x) {
    union U32 {
        float f;
        uint32_t i;
    } u32;
    u32.f = x;
    return u32.i;
}

} // namespace

//
//    Pool2dBatch
//

Pool2dBatch::Pool2dBatch(
        Pool2dBatchOp op,
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size):
            m_op(op),
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_W(uint32_t(W)),
            m_C(uint32_t(C)),
            m_P(uint32_t(P)),
            m_Q(uint32_t(Q)),
            m_R(uint32_t(R)),
            m_S(uint32_t(S)),
            m_pad_h(uint32_t(pad_h)),
            m_pad_w(uint32_t(pad_w)),
            m_stride_h(uint32_t(stride_h)),
            m_stride_w(uint32_t(stride_w)),
            m_dilation_h(uint32_t(dilation_h)),
            m_dilation_w(uint32_t(dilation_w)),
            m_batch_size(uint32_t(batch_size)) { }

Pool2dBatch::~Pool2dBatch() { }

void Pool2dBatch::init(
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

    m_init_size = m_C;
    m_mask_size = (m_P * m_Q + 31) / 32;
    m_mask_size *= m_R * m_S;

    uint32_t Ct = m_C / 32;

    m_px_frame_size = Ct;
    m_py_frame_size = Ct;
    m_py_im_frame_size = Ct;

    m_start_p = uint32_t(-int32_t(m_pad_h * m_W * m_C));
    m_start_q = uint32_t(-int32_t(m_pad_w * m_C));
    m_delta_p = m_stride_h * m_W * m_C;
    m_delta_q = m_stride_w * m_C;
    m_delta_r = m_dilation_h * m_W * m_C;
    m_delta_s = m_dilation_w * m_C;
    m_end_q = m_start_q + m_Q * m_delta_q;

    m_kernel_base_path = "op/pool/device/metal";
    m_defines = {{"T", "bfloat16"}};

    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_kernels();

    init_locals();
}

void Pool2dBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int Pool2dBatch::input_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_H * m_W, 32) * m_C;
}

int Pool2dBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_C;
}

std::vector<float> Pool2dBatch::transform_input(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::pad(x, m_N, m_H * m_W, m_C, m_N, u32_align(m_H * m_W, 32), m_C);
}

std::vector<float> Pool2dBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_C, m_N, m_P * m_Q, m_C);
}

void Pool2dBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void Pool2dBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    m_ginit = core::Global(m_device, T, m_init_size, log2_page_size);
    m_gmask = core::Global(m_device, core::DataFormat::UINT32, m_mask_size, log2_page_size);
}

void Pool2dBatch::create_locals() {
    m_linit = core::Local(m_program, m_grid, T, m_init_size);
    m_lmask = core::Local(m_program, m_grid, core::DataFormat::UINT32, m_mask_size);
}

void Pool2dBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    m_py =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_py_frame_size * 2,
            m_py_frame_size);
    if (!(m_R == 1 && m_S == 1)) {
        m_py_im =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INTERMED,
                T,
                m_py_im_frame_size * 2,
                m_py_im_frame_size);
    }
}

void Pool2dBatch::create_kernels() {
    create_reader();
    create_writer();
    if (m_R == 1 && m_S == 1) {
        create_1x1_math();
    } else if (m_op == Pool2dBatchOp::AVG) {
        create_avg_math();
    } else if (m_op == Pool2dBatchOp::MAX) {
        create_max_math();
    } else {
        assert(false);
    }
}

void Pool2dBatch::create_reader() {
    std::string path = m_kernel_base_path + "/pool2d_batch_reader.cpp";
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
        global<T> ginit,
        global<uint32> gmask,
        local<T> linit,
        local<uint32> lmask,
        pipe<T> px,
        uint32 N,
        uint32 C,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 init_size,
        uint32 mask_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_ginit,
        m_gmask,
        m_linit,
        m_lmask,
        m_px,
        m_N / m_batch_size,
        m_C,
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
        m_init_size,
        m_mask_size,
        uint32_t(0), // [20] x_pos
        x_stride
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[20] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Pool2dBatch::create_writer() {
    std::string path = m_kernel_base_path + "/pool2d_batch_writer.cpp";
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

void Pool2dBatch::create_1x1_math() {
    std::string path = m_kernel_base_path + "/pool2d_batch_1x1_math.cpp";
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
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 PQ)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q
    };
    m_math.set_args(m_grid, args);
}

void Pool2dBatch::create_avg_math() {
    std::string path = m_kernel_base_path + "/pool2d_batch_avg_math.cpp";
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
        pipe<T> py,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 scale)
*/
    uint32_t scale = float_to_u32(1.0f / float(m_R * m_S));
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        scale
    };
    m_math.set_args(m_grid, args);
}

void Pool2dBatch::create_max_math() {
    std::string path = m_kernel_base_path + "/pool2d_batch_max_math.cpp";
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
        pipe<T> py,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S
    };
    m_math.set_args(m_grid, args);
}

void Pool2dBatch::init_locals() {
    float init_value = 
        (m_op == Pool2dBatchOp::MAX) ? 
            std::numeric_limits<float>::lowest() : 
            0.0f;
    std::vector<uint16_t> vinit(m_ginit.bytes() / sizeof(uint16_t), float_to_u16b(init_value));
    std::vector<uint32_t> vmask(m_gmask.bytes() / sizeof(uint32_t), 0);
    compute_mask(vmask);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_ginit, vinit.data(), true);
    queue.enqueue_write(m_gmask, vmask.data(), true);
}

void Pool2dBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

void Pool2dBatch::compute_mask(std::vector<uint32_t> &vmask) {
    int PQ = m_P * m_Q;
    int size = (PQ + 31) / 32;
    size *= m_R * m_S;
    assert(size <= vmask.size());
    int k = 0;
    for (int pq_start = 0; pq_start < PQ; pq_start += 32) {
        for (int r = 0; r < int(m_R); r++) {
            for (int s = 0; s < int(m_S); s++) {
                uint32_t mask = 0;
                uint32_t flag = 1;
                for (int i = 0; i < 32; i++) {
                    bool valid = true;
                    int pq = pq_start + i;
                    if (pq >= PQ) {
                        valid = false;
                    } else {
                        int p = pq / m_Q;
                        int q = pq - p * m_Q;
                        int h = p * m_stride_h - m_pad_h + r * m_dilation_h;
                        int w = q * m_stride_w - m_pad_w + s * m_dilation_w;
                        if (h < 0 || h >= int(m_H) || w < 0 || w >= int(m_W)) {
                            valid = false;
                        }
                    }
                    if (valid) {
                        mask |= flag;
                    }
                    flag <<= 1;
                }
                vmask[k] = mask;
                k++;
            }
        }
    }
}

//
//    AvgPool2dBatch
//

AvgPool2dBatch::AvgPool2dBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size):
            Pool2dBatch(
                Pool2dBatchOp::AVG,
                N,
                H,
                W,
                C,
                P,
                Q,
                R,
                S,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                batch_size) { }

AvgPool2dBatch::~AvgPool2dBatch() { }

//
//    MaxPool2dBatch
//

MaxPool2dBatch::MaxPool2dBatch(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_size):
            Pool2dBatch(
                Pool2dBatchOp::MAX,
                N,
                H,
                W,
                C,
                P,
                Q,
                R,
                S,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                batch_size) { }

MaxPool2dBatch::~MaxPool2dBatch() { }

} // namespace tanto
} // namespace pool
} // namespace op
} // namespace ronin

