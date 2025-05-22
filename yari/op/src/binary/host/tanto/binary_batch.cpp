// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/binary_batch.hpp"

namespace ronin {
namespace op {
namespace binary {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

//
//    BinaryBatch
//

BinaryBatch::BinaryBatch(
        BinaryBatchOp op,
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size):
            m_op(op),
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_C(uint32_t(C)),
            m_post_op(post_op),
            m_batch_size(uint32_t(batch_size)) {
    m_H_arg = m_H;
    m_C_arg = m_C;
    m_H = u32_align(m_H, 32);
    m_C = u32_align(m_C, 32);
}

BinaryBatch::~BinaryBatch() { }

void BinaryBatch::init(
        const core::Device &device,
        const core::Global &ga,
        const core::Global &gb,
        const core::Global &gc) {
    m_device = device;
    m_ga = ga;
    m_gb = gb;
    m_gc = gc;

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    m_x_start = 0;
    m_y_start = 0;
    m_x_end = grid_x - 1;
    m_y_end = grid_y - 1;

    m_grid = core::Grid(m_program, m_x_start, m_y_start, m_x_end, m_y_end);

    m_pipe_frame_size = 1;

    m_kernel_base_path = "op/binary/device/metal";
    m_defines = {{"T", "bfloat16"}};

    validate_globals();

    create_pipes();
    create_kernels();
}

void BinaryBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int BinaryBatch::input_volume(int index) {
    assert(index == 0 || index == 1);
    return m_N * m_H * m_C;
}

int BinaryBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_C;
}

std::vector<float> BinaryBatch::transform_input(int index, const std::vector<float> &x) {
    return util::pad(x, m_N, m_H_arg, m_C_arg, m_N, m_H, m_C);
}

std::vector<float> BinaryBatch::transform_output(int index, const std::vector<float> &x) {
    return util::unpad(x, m_N, m_H, m_C, m_N, m_H_arg, m_C_arg);
}

void BinaryBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_ga.is_null());
    assert(!m_gb.is_null());
    assert(!m_gc.is_null());
    assert(m_ga.bytes() >= input_volume(0) * item_bytes);
    assert(m_gb.bytes() >= input_volume(1) * item_bytes);
    assert(m_gc.bytes() >= output_volume(0) * item_bytes);
}

void BinaryBatch::create_pipes() {
    m_pa =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_pc =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void BinaryBatch::create_kernels() {
    create_reader();
    create_writer();
    if (m_post_op.op() != base::PostOp::NONE) {
        create_unary_math();
    } else {
        create_math();
    }
}

void BinaryBatch::create_reader() {
    std::string path = m_kernel_base_path + "/binary_batch_reader.cpp";
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
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride)
*/
    uint32_t HC = m_H * m_C;
    uint32_t stride = m_batch_size * HC;
    uint32_t num_frames = HC / (m_pipe_frame_size * 1024);
    std::vector<core::KernelArg> args{
        m_ga,
        m_gb,
        m_pa,
        m_pb, 
        m_N / m_batch_size,
        num_frames,
        m_pipe_frame_size,
        uint32_t(0), // [7] start
        stride
    };
    uint32_t start = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[7] = start;
            m_reader.set_args(x, y, args);
            start += HC;
        }
    }
}

void BinaryBatch::create_writer() {
    std::string path = m_kernel_base_path + "/binary_batch_writer.cpp";
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
        global<T> gc,
        pipe<T> pc,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride)
*/
    uint32_t HC = m_H * m_C;
    uint32_t stride = m_batch_size * HC;
    uint32_t num_frames = HC / (m_pipe_frame_size * 1024);
    std::vector<core::KernelArg> args{
        m_gc,
        m_pc,
        m_N / m_batch_size,
        num_frames,
        m_pipe_frame_size,
        uint32_t(0), // [5] start
        stride
    };
    uint32_t start = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[5] = start;
            m_writer.set_args(x, y, args);
            start += HC;
        }
    }
}

void BinaryBatch::create_math() {
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
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_frames,
        uint32 frame_tiles)
*/
    uint32_t num_frames = (m_N / m_batch_size) * (m_H * m_C) / (m_pipe_frame_size * 1024); 
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        num_frames,
        m_pipe_frame_size
    };
    m_math.set_args(m_grid, args);
}

void BinaryBatch::create_unary_math() {
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
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 unary_param0)
*/
    uint32_t num_frames = (m_N / m_batch_size) * (m_H * m_C) / (m_pipe_frame_size * 1024); 
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_pa,
        m_pb,
        m_pc,
        num_frames,
        m_pipe_frame_size,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void BinaryBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

std::string BinaryBatch::make_math_name() {
    std::string name;
    switch (m_op) {
    case BinaryBatchOp::ADD:
        name = "binary_batch_add";
        break;
    case BinaryBatchOp::SUB:
        name = "binary_batch_sub";
        break;
    case BinaryBatchOp::MUL:
        name = "binary_batch_mul";
        break;
    default:
        assert(false);
        break;
    }
    if (m_post_op.op() != base::PostOp::NONE) {
        name += "_" + get_unary_kernel_suffix();
    }
    name += "_math";
    return name;
}

std::string BinaryBatch::get_unary_kernel_suffix() {
    base::PostOp op = m_post_op.op();
    switch (op) {
    case base::PostOp::RELU:
        return "relu";
    case base::PostOp::CLIP:
        if (is_unary_relu6()) {
            return "relu6";
        }
        // generic clip is not yet implemented
        assert(false);
        return "<?>";
    default:
        assert(false);
        return "<?>";
    }
}

uint32_t BinaryBatch::encode_unary_param0() {
    base::PostOp op = m_post_op.op();
    switch (op) {
    case base::PostOp::RELU:
        return 0;
    case base::PostOp::CLIP:
        if (is_unary_relu6()) {
            // param0 hardcoded in kernels
            return 0;
        }
        // generic clip is not yet implemented
        assert(false);
        return 0;
    default:
        assert(false);
        return 0;
    }
}

bool BinaryBatch::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

//
//    AddBatch
//

AddBatch::AddBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size):
            BinaryBatch(
                BinaryBatchOp::ADD,
                N,
                H,
                C,
                post_op,
                batch_size) { }

AddBatch::~AddBatch() { }

//
//    SubBatch
//

SubBatch::SubBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size):
            BinaryBatch(
                BinaryBatchOp::SUB,
                N,
                H,
                C,
                post_op,
                batch_size) { }

SubBatch::~SubBatch() { }

//
//    MulBatch
//

MulBatch::MulBatch(
        int N,
        int H,
        int C,
        const base::PostOpSpec &post_op,
        int batch_size):
            BinaryBatch(
                BinaryBatchOp::MUL,
                N,
                H,
                C,
                post_op,
                batch_size) { }

MulBatch::~MulBatch() { }

} // namespace tanto
} // namespace binary
} // namespace op
} // namespace ronin

