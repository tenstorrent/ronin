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
#include "host/tanto/conv2d_basic_split.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

//
//    Conv2dBasicSplit
//

Conv2dBasicSplit::Conv2dBasicSplit(
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
        int batch_size):
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_W(uint32_t(W)),
            m_C(uint32_t(C)),
            m_P(uint32_t(P)),
            m_Q(uint32_t(Q)),
            m_K(uint32_t(K)),
            m_R(uint32_t(R)),
            m_S(uint32_t(S)),
            m_pad_h(uint32_t(pad_h)),
            m_pad_w(uint32_t(pad_w)),
            m_stride_h(uint32_t(stride_h)),
            m_stride_w(uint32_t(stride_w)),
            m_dilation_h(uint32_t(dilation_h)),
            m_dilation_w(uint32_t(dilation_w)),
            m_post_op(post_op),
            m_batch_size(uint32_t(batch_size)) { 
    // these values are required to compute input/output volumes and shapes
    m_C_arg = m_C;
    m_K_arg = m_K;
    m_C = u32_align(m_C, 32);
    m_K = u32_align(m_K, 32);
}

Conv2dBasicSplit::~Conv2dBasicSplit() { }

void Conv2dBasicSplit::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gz,
        const core::Global &gy) {
    m_device = device;
    m_gx = gx;
    m_gw = gw;
    m_gb = gb;
    m_gz = gz;
    m_gy = gy;

    // ACHTUNG: Supported batch sizes are Wormhole-specific
    assert(m_batch_size == 8 || m_batch_size == 16);
    assert(m_N % m_batch_size == 0);

    assert(m_C % 32 == 0);
    assert(m_K % 32 == 0);

    m_program = core::Program(m_device);

    compute_grid_dims(m_grid_x, m_grid_y);

    m_grid = core::Grid(m_program, 0, 0, m_grid_x - 1, m_grid_y - 1);
    m_block_size = (m_grid_x * m_grid_y) / m_batch_size;

    m_zero_size = m_C;
    m_mask_size = (m_P * m_Q + 31) / 32;
    m_mask_size *= m_R * m_S;

    assert(m_K % (m_block_size * 32) == 0);
    uint32_t Ct = m_C / 32;
    uint32_t Kbt = m_K / (m_block_size * 32);

    m_px_frame_size = Ct;
    m_pw_frame_size = m_R * m_S * Kbt * Ct;
    if (!m_gb.is_null()) {
        m_pb_frame_size = Kbt;
    }
    if (!m_gz.is_null()) {
        m_pz_frame_size = Kbt;
    }
    m_py_frame_size = Kbt;
    m_px_im_frame_size = Ct;
    m_pz_im_frame_size = Kbt;
    m_py_im_frame_size = Kbt;

    m_start_p = uint32_t(-int32_t(m_pad_h * m_W * m_C));
    m_start_q = uint32_t(-int32_t(m_pad_w * m_C));
    m_delta_p = m_stride_h * m_W * m_C;
    m_delta_q = m_stride_w * m_C;
    m_delta_r = m_dilation_h * m_W * m_C;
    m_delta_s = m_dilation_w * m_C;
    m_end_q = m_start_q + m_Q * m_delta_q;

    m_kernel_base_path = "op/conv/device/metal";
    m_defines = {{"T", "bfloat16"}};

    init_options();
    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_semaphores();
    create_kernels();

    init_locals();
}

void Conv2dBasicSplit::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int Conv2dBasicSplit::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * u32_align(m_H * m_W, 32) * m_C;
    case 1:
        return m_R * m_S * m_K * m_C;
    case 2:
        return m_K * 32;
    case 3:
        return m_N * u32_align(m_P * m_Q, 32) * m_K;
    default:
        assert(false);
        return 0;
    }
}

int Conv2dBasicSplit::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_K;
}

std::vector<float> Conv2dBasicSplit::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        y = util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        break;
    case 1:
        y = util::pad(x, m_R * m_S, m_K_arg, m_C_arg, m_R * m_S, m_K, m_C);
        y = util::tilize(y, m_R * m_S * m_K, m_C);
        y = util::make_faces(y);
        break;
    case 2:
        y = util::pad(x, 1, m_K_arg, 32, m_K);
        y = util::tilize(y, 32, m_K);
        y = util::make_faces(y);
        break;
    case 3:
        y = util::pad(x, m_N, m_P * m_Q, m_K_arg, m_N, u32_align(m_P * m_Q, 32), m_K);
        break;
    default:
        assert(false);
        break;
    }
    return y;
}

std::vector<float> Conv2dBasicSplit::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_K, m_N, m_P * m_Q, m_K_arg);
}

void Conv2dBasicSplit::init_options() {
    m_options = 0;
    if (!m_gb.is_null()) {
        m_options |= OPT_BIAS;
    }
    if (!m_gz.is_null()) {
        m_options |= OPT_ADD;
    }
    // TODO: Support all post ops
    base::PostOp op = m_post_op.op();
    if (op != base::PostOp::NONE) {
        m_options |= OPT_UNARY;
    }
}

void Conv2dBasicSplit::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gw.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gw.bytes() == input_volume(1) * item_bytes);
    if (!m_gb.is_null()) {
        assert(m_gb.bytes() == input_volume(2) * item_bytes);
    }
    if (!m_gz.is_null()) {
        assert(m_gz.bytes() >= input_volume(3) * item_bytes);
    }
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void Conv2dBasicSplit::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
    m_gmask = core::Global(m_device, core::DataFormat::UINT32, m_mask_size, log2_page_size);
}

void Conv2dBasicSplit::create_locals() {
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
    m_lmask = core::Local(m_program, m_grid, core::DataFormat::UINT32, m_mask_size);
}

void Conv2dBasicSplit::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    // no double buffering: weights are always preloaded
    m_pw =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pw_frame_size,
            m_pw_frame_size);
    if (!m_gb.is_null()) {
        m_pb =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INPUT,
                T,
                m_pb_frame_size * 2,
                m_pb_frame_size);
    }
    if (!m_gz.is_null()) {
        m_pz =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INPUT,
                T,
                m_pz_frame_size * 2,
                m_pz_frame_size);
    }
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
    if (!m_gz.is_null()) {
        m_pz_im =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INTERMED,
                T,
                m_pz_im_frame_size * 2,
                m_pz_im_frame_size);
    }
    m_py_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_py_im_frame_size * 2,
            m_py_im_frame_size);
}

void Conv2dBasicSplit::create_semaphores() {
    m_sem_send_x = core::Semaphore(m_program, m_grid, 0);
    m_sem_recv_x = core::Semaphore(m_program, m_grid, 0);
    m_sem_send_w = core::Semaphore(m_program, m_grid, 0);
    m_sem_recv_w = core::Semaphore(m_program, m_grid, 0);
}

void Conv2dBasicSplit::create_kernels() {
    switch (m_options) {
    case OPT_BIAS:
        create_bias_reader();
        create_writer();
        create_bias_math();
        break;
    case OPT_BIAS | OPT_ADD:
        create_bias_reader();
        create_add_writer();
        create_bias_add_math();
        break;
    case OPT_BIAS | OPT_UNARY:
        create_bias_reader();
        create_writer();
        create_bias_unary_math();
        break;
    case OPT_BIAS | OPT_ADD | OPT_UNARY:
        create_bias_reader();
        create_add_writer();
        create_bias_add_unary_math();
        break;
    default:
        // not yet implemented
        assert(false);
        break;
    }
}

void Conv2dBasicSplit::create_bias_reader() {
    std::string path = m_kernel_base_path + "/basic_split_bias_reader.cpp";
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
        uint32 C,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 Kb,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 zero_size,
        uint32 mask_size,
        uint32 x_pos,
        uint32 x_stride,
        uint32 b_pos)
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
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
        uint32_t(0), // [31] x_pos
        x_stride,
        uint32_t(0)  // [33] b_pos
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
            args[31] = x_pos;
            args[33] = b_pos;
            m_reader.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_writer() {
    std::string path = m_kernel_base_path + "/basic_split_writer.cpp";
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
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 RS,
        uint32 KC,
        uint32 Kb,
        uint32 KbC,
        uint32 RSKbC,
        uint32 w_pos,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
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
        m_K,
        m_P * m_Q,
        m_R * m_S,
        m_K * m_C,
        Kb,
        Kb * m_C,
        m_R * m_S * Kb * m_C,
        uint32_t(0), // [20] w_pos
        uint32_t(0), // [21] y_pos
        y_stride
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
            args[20] = w_pos;
            args[21] = y_pos;
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_add_writer() {
    std::string path = m_kernel_base_path + "/basic_split_add_writer.cpp";
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
        uint32 K,
        uint32 PQ,
        uint32 RS,
        uint32 KC,
        uint32 Kb,
        uint32 KbC,
        uint32 RSKbC,
        uint32 w_pos,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
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
        m_K,
        m_P * m_Q,
        m_R * m_S,
        m_K * m_C,
        Kb,
        Kb * m_C,
        m_R * m_S * Kb * m_C,
        uint32_t(0), // [22] w_pos
        uint32_t(0), // [23] y_pos
        y_stride
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
            args[22] = w_pos;
            args[23] = y_pos;
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSplit::create_bias_math() {
    std::string path = m_kernel_base_path + "/basic_split_bias_math.cpp";
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
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 Kb,
        uint32 RSKbC)
*/
    uint32_t Kb = m_K / m_block_size;
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_bias_add_math() {
    std::string path = m_kernel_base_path + "/basic_split_bias_add_math.cpp";
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
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 Kb,
        uint32 RSKbC)
*/
    uint32_t Kb = m_K / m_block_size;
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_bias_unary_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/basic_split_bias_" + suffix + "_math.cpp";
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
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 Kb,
        uint32 RSKbC,
        unary_param0)
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::create_bias_add_unary_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/basic_split_bias_add_" + suffix + "_math.cpp";
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
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS,
        uint32 Kb,
        uint32 RSKbC,
        unary_param0)
*/
    uint32_t Kb = m_K / m_block_size;
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_P * m_Q,
        m_R * m_S,
        Kb,
        m_R * m_S * Kb * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSplit::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    std::vector<uint32_t> vmask(m_gmask.bytes() / sizeof(uint32_t), 0);
    compute_mask(vmask);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
    queue.enqueue_write(m_gmask, vmask.data(), true);
}

void Conv2dBasicSplit::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    x = 8;
    y = 8;
}

void Conv2dBasicSplit::compute_mask(std::vector<uint32_t> &vmask) {
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

std::string Conv2dBasicSplit::get_unary_kernel_suffix() {
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

uint32_t Conv2dBasicSplit::encode_unary_param0() {
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

bool Conv2dBasicSplit::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

