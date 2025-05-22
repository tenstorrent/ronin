// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/conv2d_image_batch.hpp"

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

constexpr bool ENABLE_C4 = true;

namespace {

void copy(float *dst, const float *src, int count) {
    memcpy(dst, src, count * sizeof(float));
}

std::vector<float> pad_w(
        const std::vector<float> &x,
        int N,
        int H,
        int W,
        int C,
        int pad_left,
        int pad_right) {
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }
    int NH = N * H;
    int WC = W * C;
    int WCy = (pad_left + W + pad_right) * C;
    std::vector<float> y(NH * WCy, 0.0f);
    const float *px = x.data();
    float *py = y.data() + pad_left * C;
    for (int nh = 0; nh < NH; nh++) {
        copy(py, px, WC);
        px += WC;
        py += WCy;
    }
    return y;
}

} // namespace

//
//    Conv2dImageBatch
//

Conv2dImageBatch::Conv2dImageBatch(
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
    m_W_arg = m_W;
    m_C_arg = m_C;
    m_K_arg = m_K;
    m_S_arg = m_S;
    m_enable_c4 = ENABLE_C4 && (m_stride_w % 2 == 0);
    if (m_enable_c4) {
        // ACHTUNG: DRAM alignment is Wormhole-specific and assumes 16-bit floats
        uint32_t dram_align = 4;
        m_W = u32_align(m_W + 2 * pad_w, dram_align);
        m_C = 4;
        m_S = u32_align(m_S, 2);
    } else {
        // keep original W and S arguments
        m_C = 8;
    }
    m_K = u32_align(m_K, 32);
    m_RSC_rnd = u32_align(m_R * m_S * m_C, 32);            
}

Conv2dImageBatch::~Conv2dImageBatch() { }

void Conv2dImageBatch::init(
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

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    assert(m_C_arg <= m_C);
    assert(m_dilation_h == 1);
    assert(m_dilation_w == 1);

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

    // unit dilation assumed
    uint32_t P_rnd = (u32_align(m_P * m_Q, 32) + m_Q - 1) / m_Q;
    m_before_h = m_pad_h;
    m_after_h = (P_rnd - 1) * m_stride_h - m_pad_h + (m_R - 1) - m_H + 1;
    if (int32_t(m_after_h) < 0) {
        m_after_h = 0;
    }
    if (m_enable_c4) {
        // pad_w vanishes in this case (already included in W)
        m_before_w = 0;
        m_after_w = (m_Q - 1) * m_stride_w - (m_S - 1) - m_W + 1;
        if (int32_t(m_after_w) < 0) {
            m_after_w = 0;
        }
        assert(m_after_w % 2 == 0);
        m_offset_w = 0;
    } else {
        m_before_w = m_pad_w;
        m_after_w = (m_Q - 1) * m_stride_w - m_pad_w + (m_S - 1) - m_W + 1;
        if (int32_t(m_after_w) < 0) {
            m_after_w = 0;
        }
        // DRAM -> L1 reads require L1 to be also DRAM-aligned
        // ACHTUNG: This is Wormhole-specific (align by 4 for Blackhole)
        m_before_w = u32_align(m_before_w, 2);
        m_after_w = u32_align(m_after_w, 2);
        m_offset_w = m_before_w - m_pad_w;
    }

    uint32_t RSCt = m_RSC_rnd / 32;
    uint32_t Kt = m_K / 32;

    m_Ki = (Kt > DEST_TILES) ? DEST_TILES : Kt;
    m_Ko = Kt / m_Ki;
    assert(m_Ki * m_Ko == Kt);

    uint32_t W_full = m_before_w + m_W + m_after_w;
    m_zero_size = W_full * m_C;

    m_px_frame_size = RSCt;
    m_pw_frame_size = RSCt;
    m_pb_frame_size = Kt;
    m_pz_frame_size = Kt;
    m_py_frame_size = Kt;
    m_px_im_frame_size = RSCt;
    m_pz_im_frame_size = Kt;
    m_py_im_frame_size = Kt;

    m_delta_p = m_stride_h * W_full * m_C;
    m_delta_q = m_stride_w * m_C;
    m_delta_r = W_full * m_C;
    m_end_q = m_Q * m_delta_q;

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

void Conv2dImageBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int Conv2dImageBatch::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * u32_align(m_H * m_W, 32) * m_C;
    case 1:
        return m_K * m_RSC_rnd;
    case 2:
        return m_K * 32;
    case 3:
        return m_N * u32_align(m_P * m_Q, 32) * m_K;
    default:
        assert(false);
        return 0;
    }
}

int Conv2dImageBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_K;
}

std::vector<float> Conv2dImageBatch::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        if (m_enable_c4) {
            y = pad_w(x, m_N, m_H, m_W_arg, m_C_arg, m_pad_w, m_W - m_W_arg - m_pad_w);
            y = util::pad(y, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        } else {
            y = util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        }
        break;
    case 1:
        y = util::pad(x, m_K_arg, m_R, m_S_arg, m_C_arg, m_K, m_R, m_S, m_C);
        y = util::pad(y, m_K, m_R * m_S * m_C, m_K, m_RSC_rnd);
        y = util::tilize(y, m_K, m_RSC_rnd);
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

std::vector<float> Conv2dImageBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_K, m_N, m_P * m_Q, m_K_arg);
}

void Conv2dImageBatch::init_options() {
    m_options = 0;
    if (!m_gb.is_null()) {
        m_options |= OPT_BIAS;
    }
    if (!m_gz.is_null()) {
        m_options |= OPT_ADD;
    }
    base::PostOp op = m_post_op.op();
    if (op != base::PostOp::NONE) {
        m_options |= OPT_UNARY;
    }
}

void Conv2dImageBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gw.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gw.bytes() == input_volume(1) * item_bytes);
    if (!m_gb.is_null()) {
        assert(m_gb.bytes() == input_volume(2) * item_bytes);
    }
    assert(m_gz.is_null());
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void Conv2dImageBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
}

void Conv2dImageBatch::create_locals() {
    uint32_t H_full = m_before_h + m_H + m_after_h;
    uint32_t W_full = m_before_w + m_W + m_after_w;
    m_lx = core::Local(m_program, m_grid, T, H_full * W_full * m_C);
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
}

void Conv2dImageBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    uint32_t pw_size = m_K * m_RSC_rnd / 1024;
    m_pw =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            pw_size,
            m_pw_frame_size);
    if (!m_gb.is_null()) {
        m_pb =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INPUT,
                T,
                m_pb_frame_size,
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

void Conv2dImageBatch::create_semaphores() {
    m_sem_send = core::Semaphore(m_program, m_grid, 0);
    m_sem_recv = core::Semaphore(m_program, m_grid, 0);
}

void Conv2dImageBatch::create_kernels() {
    uint32_t reader_options = m_options & OPT_BIAS;
    uint32_t writer_options = m_options & OPT_ADD;
    uint32_t math_options = m_options;
    if (reader_options == OPT_BIAS) {
        create_bias_reader();
    } else {
        // not yet implemented
        assert(false);
    }
    if (writer_options == OPT_ADD) {
        // not yet implemented
        assert(false);
    } else {
        if (m_batch_size <= 8) {
            create_writer();
        } else {
            create_mcast_writer();
        }
    }
    if (math_options == OPT_BIAS) {
        create_bias_math();
    } else if (math_options == (OPT_BIAS | OPT_UNARY)) {
        create_bias_unary_math();
    } else {
        // not yet implemented
        assert(false);
    }
}

void Conv2dImageBatch::create_bias_reader() {
    std::string path = m_kernel_base_path + "/image_batch_bias_reader.cpp";
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
        local<T> lx,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 H,
        uint32 K,
        uint32 R,
        uint32 WC,
        uint32 PQ,
        uint32 SC,
        uint32 RSC_rnd,
        uint32 before_h,
        uint32 after_h,
        uint32 before_wc,
        uint32 after_wc,
        uint32 offset_wc,
        uint32 before_hwc,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 end_q,
        uint32 x_pos,
        uint32 x_stride,
        uint32 zero_size)
*/
    uint32_t before_wc = m_before_w * m_C;
    uint32_t after_wc = m_after_w * m_C;
    uint32_t offset_wc = m_offset_w * m_C;
    uint32_t before_hwc = m_before_h * (m_before_w + m_W + m_after_w) * m_C;
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_lx,
        m_lzero,
        m_px,
        m_pb,
        m_N / m_batch_size,
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
        uint32_t(0), // [25] x_pos
        x_stride,
        m_zero_size
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[25] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void Conv2dImageBatch::create_writer() {
    std::string path = m_kernel_base_path + "/image_batch_writer.cpp";
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
        uint32 K,
        uint32 PQ,
        uint32 KRSC_rnd,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_N / m_batch_size,
        m_K,
        m_P * m_Q,
        m_K * m_RSC_rnd,
        uint32_t(0), // [8] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[8] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dImageBatch::create_mcast_writer() {
    std::string path = m_kernel_base_path + "/image_batch_mcast_writer.cpp";
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
        uint32 KRSC_rnd,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
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
        m_K,
        m_P * m_Q,
        m_K * m_RSC_rnd,
        uint32_t(0), // [16] y_pos
        y_stride
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
            args[16] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void Conv2dImageBatch::create_bias_math() {
    std::string path = m_kernel_base_path + "/image_batch_bias_math.cpp";
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
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RSC_rnd)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_RSC_rnd
    };
    m_math.set_args(m_grid, args);
}

void Conv2dImageBatch::create_bias_unary_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/image_batch_bias_" + suffix + "_math.cpp";
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
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RSC_rnd,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_RSC_rnd,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dImageBatch::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
}

void Conv2dImageBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

std::string Conv2dImageBatch::get_unary_kernel_suffix() {
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

uint32_t Conv2dImageBatch::encode_unary_param0() {
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

bool Conv2dImageBatch::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

