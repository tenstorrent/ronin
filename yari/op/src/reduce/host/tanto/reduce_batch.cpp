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
#include "host/tanto/reduce_batch.hpp"

namespace ronin {
namespace op {
namespace reduce {
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

} // namespace

//
//    ReduceBatch
//

ReduceBatch::ReduceBatch(
        ReduceBatchOp op,
        int N,
        int H,
        int W,
        int axis,
        int batch_size):
            m_op(op),
            m_N(N),
            m_H(H),
            m_W(W),
            m_axis(axis),
            m_batch_size(batch_size) { }

ReduceBatch::~ReduceBatch() { }

void ReduceBatch::init(
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

    assert(m_W % 32 == 0);
    assert(m_axis == 1 || m_axis == 2);

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    m_x_start = 0;
    m_y_start = 0;
    m_x_end = grid_x - 1;
    m_y_end = grid_y - 1;

    m_grid = core::Grid(m_program, m_x_start, m_y_start, m_x_end, m_y_end);

    m_zero_size = m_W;

    uint32_t Ht = (m_H + 31) / 32;
    uint32_t Wt = m_W / 32;
    m_px_frame_size = Wt;
    m_ps_frame_size = 1;
    m_py_frame_size = (m_axis == 1) ? Wt : Ht;
    m_px_im_frame_size = Wt;
    m_py_im_frame_size = (m_axis == 1) ? Wt : Ht;

    m_kernel_base_path = "op/reduce/device/metal";
    m_defines = {{"T", "bfloat16"}};

    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_kernels();

    init_locals();
}

void ReduceBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int ReduceBatch::input_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_H, 32) * m_W;
}

int ReduceBatch::output_volume(int index) {
    assert(index == 0);
    if (m_axis == 1) {
        return m_N * 32 * m_W;
    } else {
        return m_N * u32_align(m_H, 32) * 32;
    }
}

std::vector<float> ReduceBatch::transform_input(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::pad(x, m_N, m_H, m_W, m_N, u32_align(m_H, 32), m_W);
}

std::vector<float> ReduceBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    if (m_axis == 1) {
        return util::unpad(x, m_N, 32, m_W, m_N, 1, m_W);
    } else {
        return util::unpad(x, m_N, u32_align(m_H, 32), 32, m_N, m_H, 1);
    }
}

void ReduceBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void ReduceBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    m_gs = core::Global(m_device, T, 1024, log2_page_size);
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
}

void ReduceBatch::create_locals() {
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
}

void ReduceBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    // no double buffering for scale
    m_ps =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_ps_frame_size,
            m_ps_frame_size);
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

void ReduceBatch::create_kernels() {
    create_reader();
    if (m_axis == 1) {
        create_cols_writer();
        create_cols_math();
    } else {
        create_rows_writer();
        create_rows_math();
    }
}

void ReduceBatch::create_reader() {
    std::string path = m_kernel_base_path + "/reduce_batch_reader.cpp";
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
        global<T> gs,
        global<T> gzero,
        local<T> lzero,
        pipe<T> px,
        pipe<T> ps,
        uint32 N,
        uint32 H,
        uint32 W,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    uint32_t H_rnd = u32_align(m_H, 32);
    uint32_t x_stride = (m_batch_size - 1) * H_rnd * m_W;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gs,
        m_gzero,
        m_lzero,
        m_px,
        m_ps,
        m_N / m_batch_size,
        m_H,
        m_W,
        m_zero_size,
        uint32_t(0), // [10] x_pos
        x_stride
    };
    uint32_t x_inc = H_rnd * m_W;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[10] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void ReduceBatch::create_cols_writer() {
    std::string path = m_kernel_base_path + "/reduce_batch_cols_writer.cpp";
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
        uint32 W,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t y_stride = m_batch_size * 32 * m_W;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        m_N / m_batch_size,
        m_W,
        uint32_t(0), // [4] y_pos
        y_stride
    };
    uint32_t y_inc = 32 * m_W;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[4] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void ReduceBatch::create_rows_writer() {
    std::string path = m_kernel_base_path + "/reduce_batch_rows_writer.cpp";
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
        uint32 H,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t H_rnd = u32_align(m_H, 32);
    uint32_t y_stride = m_batch_size * 32 * H_rnd;
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        m_N / m_batch_size,
        H_rnd,
        uint32_t(0), // [4] y_pos
        y_stride
    };
    uint32_t y_inc = 32 * H_rnd;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[4] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void ReduceBatch::create_cols_math() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
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
        pipe<T> ps,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 H,
        uint32 W)
*/
    uint32_t H_rnd = u32_align(m_H, 32);
    std::vector<core::KernelArg> args{
        m_px,
        m_ps,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        H_rnd,
        m_W
    };
    m_math.set_args(m_grid, args);
}

void ReduceBatch::create_rows_math() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
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
        pipe<T> ps,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 H,
        uint32 W)
*/
    uint32_t H_rnd = u32_align(m_H, 32);
    std::vector<core::KernelArg> args{
        m_px,
        m_ps,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        H_rnd,
        m_W
    };
    m_math.set_args(m_grid, args);
}

void ReduceBatch::init_locals() {
    float scale = 1.0f;
    if (m_op == ReduceBatchOp::MEAN) {
        if (m_axis == 1) {
            scale = 1.0f / float(m_H);
        } else {
            scale = 1.0f / float(m_W);
        }
    }
    std::vector<uint16_t> vscale(1024, float_to_u16b(scale));
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gs, vscale.data(), true);
    queue.enqueue_write(m_gzero, vzero.data(), true);
}

void ReduceBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

std::string ReduceBatch::make_math_name() {
    std::string str_op = (m_op == ReduceBatchOp::MAX) ? "max" : "sum";
    std::string str_dim = (m_axis == 1) ? "cols" : "rows";
    return "reduce_batch_" + str_op + "_" + str_dim + "_math";
}

//
//    ReduceMaxBatch
//

ReduceMaxBatch::ReduceMaxBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size):
            ReduceBatch(
                ReduceBatchOp::MAX,
                N,
                H,
                W,
                axis,
                batch_size) { }

ReduceMaxBatch::~ReduceMaxBatch() { }

//
//    ReduceMeanBatch
//

ReduceMeanBatch::ReduceMeanBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size):
            ReduceBatch(
                ReduceBatchOp::MEAN,
                N,
                H,
                W,
                axis,
                batch_size) { }

ReduceMeanBatch::~ReduceMeanBatch() { }

//
//    ReduceSumBatch
//

ReduceSumBatch::ReduceSumBatch(
        int N,
        int H,
        int W,
        int axis,
        int batch_size):
            ReduceBatch(
                ReduceBatchOp::SUM,
                N,
                H,
                W,
                axis,
                batch_size) { }

ReduceSumBatch::~ReduceSumBatch() { }

} // namespace tanto
} // namespace reduce
} // namespace op
} // namespace ronin

